# ingest_incremental.py
# 用法示例见文末
import os, re, glob, json, argparse, hashlib
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm

from pymongo import MongoClient, UpdateOne
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# ----------------- Utils -----------------
def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def load_jsonl(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def iter_files(input_dir: str, pattern: str, recursive: bool = False, extra_files: List[str] = None):
    files = []
    if input_dir:
        pat = os.path.join(input_dir, pattern)
        files.extend(glob.glob(pat, recursive=recursive))
    if extra_files:
        for p in extra_files:
            if os.path.isdir(p):
                files.extend(glob.glob(os.path.join(p, pattern), recursive=recursive))
            elif os.path.isfile(p):
                files.append(p)
    # 去重 + 排序
    return sorted(list(dict.fromkeys(files)))

# ----------------- Embedding -----------------
def load_embedder(model_name: str, device: str = None):
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)

def encode_batches(model, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        sub = texts[i:i+batch_size]
        vec = model.encode(sub, normalize_embeddings=True, batch_size=len(sub), show_progress_bar=False)
        embs.extend(v.tolist() for v in vec)
    return embs

# ----------------- Milvus -----------------
def ensure_milvus_collection(col_name: str, dim: int,
                             host: str = None, port: str = None,
                             token: str = None, uri: str = None):
    if uri:
        connections.connect("default", uri=uri, token=token)
    else:
        connections.connect("default", host=host, port=port, token=token)
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="case_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="court", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="cause", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="contract_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="law_name", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="article_no", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="version_date", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="validity_status", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="LawLLM KB chunks")
        coll = Collection(name=col_name, schema=schema, using="default")
        coll.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 48, "efConstruction": 200}},
        )
    coll = Collection(col_name)
    coll.load()
    return coll

def delete_milvus(coll: Collection, chunk_ids: List[str]):
    if not chunk_ids: return
    expr = f"chunk_id in {json.dumps([str(x) for x in chunk_ids], ensure_ascii=False)}"
    coll.delete(expr)

def insert_milvus(coll: Collection, batch_rows: Dict[str, List[Any]]):
    data = [
        batch_rows["chunk_id"],
        batch_rows["doc_id"],
        batch_rows["doc_type"],
        batch_rows["case_type"],
        batch_rows["court"],
        batch_rows["cause"],
        batch_rows["contract_type"],
        batch_rows["section"],
        batch_rows["law_name"],
        batch_rows["article_no"],
        batch_rows["version_date"],
        batch_rows["validity_status"],
        batch_rows["chunk_index"],
        batch_rows["text"],
        batch_rows["embedding"],
    ]
    coll.insert(data)

# ----------------- Mongo -----------------
def ensure_mongo(uri: str, db: str, col: str):
    mc = MongoClient(uri)
    c = mc[db][col]
    c.create_index("chunk_id", unique=True)
    c.create_index([("doc_id", 1), ("chunk_index", 1)])
    c.create_index([("doc_type", 1), ("section", 1)])
    c.create_index([("version_date", 1), ("validity_status", 1)])
    c.create_index([("doc_type", 1), ("law_name", 1), ("article_no", 1), ("version_date", 1)])
    c.create_index([("doc_type", 1), ("case_type", 1), ("court", 1), ("cause", 1)])
    c.create_index([("doc_type", 1), ("contract_type", 1), ("section", 1)])
    c.create_index([("updated_at", -1)], sparse=True)
    return c

def fetch_existing_meta(col, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids: return {}
    cur = col.find({"chunk_id": {"$in": chunk_ids}}, {"chunk_id":1, "text_sha1":1, "text":1, "embedding_dim":1})
    return {d["chunk_id"]: d for d in cur}

def upsert_mongo(col, docs: List[Dict[str, Any]]):
    if not docs: return
    ops = []
    now = now_iso()
    for d in docs:
        d["updated_at"] = now
        if "created_at" not in d:
            d["created_at"] = now
        ops.append(UpdateOne({"chunk_id": d["chunk_id"]}, {"$set": d}, upsert=True))
    col.bulk_write(ops, ordered=False)

# ----------------- Core -----------------
def decide_status(in_rows: List[Dict[str, Any]], existing: Dict[str, Dict[str, Any]], skip_existing: bool) -> Tuple[List[int], List[int], List[int]]:
    """返回 (new_idx_list, changed_idx_list, same_idx_list) —— 均为 in_rows 的下标"""
    new_idx, chg_idx, same_idx = [], [], []
    for i, r in enumerate(in_rows):
        cid = r["chunk_id"]
        txt = r.get("text") or ""
        tsha = sha1_text(txt)
        r["text_sha1"] = tsha  # 写回，方便后续存库
        old = existing.get(cid)
        if not old:
            new_idx.append(i)
        else:
            if skip_existing:
                same_idx.append(i)
            else:
                old_sha = old.get("text_sha1")
                if not old_sha:
                    # 老数据没sha，回退对比文本
                    if old.get("text") == txt:
                        same_idx.append(i)
                    else:
                        chg_idx.append(i)
                else:
                    if old_sha == tsha:
                        same_idx.append(i)
                    else:
                        chg_idx.append(i)
    return new_idx, chg_idx, same_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="", help="目录（可选）")
    ap.add_argument("--files", nargs="*", help="指定一个或多个文件（可与 --input_dir 并用）")
    ap.add_argument("--pattern", default="**/*.jsonl")
    ap.add_argument("--recursive", type=int, default=1)

    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    ap.add_argument("--mongo_db", default="lawKB")
    ap.add_argument("--mongo_col", default="law_kb_chunks")

    ap.add_argument("--milvus_host", default="127.0.0.1")
    ap.add_argument("--milvus_port", default="19530")
    ap.add_argument("--milvus_token", default="")
    ap.add_argument("--milvus_uri", default="")
    ap.add_argument("--collection", default="law_kb_chunks")

    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)

    ap.add_argument("--skip_existing", type=int, default=0, help="1=库内已存在的 chunk 一律跳过（不比对文本）")
    ap.add_argument("--dry_run", type=int, default=0, help="1=只做对比与统计，不落库")
    ap.add_argument("--filter_doc_type", default="", help='只入库某类文档，如 "statute|judicial_interpretation|judgment"，可用正则')

    args = ap.parse_args()

    files = iter_files(args.input_dir, args.pattern, recursive=bool(args.recursive), extra_files=args.files)
    if not files:
        print("[INFO] No files found.")
        return
    print(f"[INFO] Files: {len(files)}")

    # init
    embedder = load_embedder(args.model)
    try:
        args.dim = embedder.get_sentence_embedding_dimension()
    except Exception:
        pass
    milvus_col = ensure_milvus_collection(
        args.collection, args.dim,
        host=args.milvus_host, port=args.milvus_port,
        token=(args.milvus_token or None), uri=(args.milvus_uri or None)
    )
    mongo_col = ensure_mongo(args.mongo_uri, args.mongo_db, args.mongo_col)

    total_new = total_chg = total_same = 0

    for fp in files:
        rows = list(load_jsonl(fp))
        if not rows:
            print(f"[SKIP] Empty file: {fp}")
            continue

        if args.filter_doc_type:
            pat = re.compile(args.filter_doc_type)
            rows = [r for r in rows if pat.search(str(r.get("doc_type","")))]
            if not rows:
                print(f"[SKIP] {fp} filtered out by doc_type")
                continue

        # 1) 对比 Mongo（按本文件的 chunk_id 范围批量查）
        chunk_ids = [r["chunk_id"] for r in rows]
        exist_map = fetch_existing_meta(mongo_col, chunk_ids)
        new_idx, chg_idx, same_idx = decide_status(rows, exist_map, skip_existing=bool(args.skip_existing))

        total_new  += len(new_idx)
        total_chg  += len(chg_idx)
        total_same += len(same_idx)

        print(f"\n[FILE] {fp}")
        print(f"  rows={len(rows)}  NEW={len(new_idx)}  CHANGED={len(chg_idx)}  SAME={len(same_idx)}")

        if args.dry_run:
            continue

        # 2) 仅对 NEW+CHANGED 计算嵌入
        work_idx = new_idx + chg_idx
        if not work_idx:
            print("  [OK] nothing to do.")
            continue
        texts = [ (rows[i].get("text") or "") for i in work_idx ]
        embs  = encode_batches(embedder, texts, batch_size=args.batch)

        # 3) Mongo upsert
        mongo_docs = []
        for i, emb in zip(work_idx, embs):
            r = dict(rows[i])
            r["embedding_dim"] = args.dim
            r["text_sha1"] = r.get("text_sha1") or sha1_text(r.get("text",""))
            mongo_docs.append(r)
        upsert_mongo(mongo_col, mongo_docs)

        # 4) Milvus：先删 CHANGED 的 PK，再插入 NEW+CHANGED
        if chg_idx:
            del_ids = [rows[i]["chunk_id"] for i in chg_idx]
            delete_milvus(milvus_col, del_ids)

        batch_rows = {
            "chunk_id":      [rows[i]["chunk_id"] for i in work_idx],
            "doc_id":        [rows[i].get("doc_id", "") for i in work_idx],
            "doc_type":      [rows[i].get("doc_type", "") for i in work_idx],
            "case_type":     [rows[i].get("case_type") or "" for i in work_idx],
            "court":         [rows[i].get("court") or "" for i in work_idx],
            "cause":         [rows[i].get("cause") or "" for i in work_idx],
            "contract_type": [rows[i].get("contract_type") or "" for i in work_idx],
            "section":       [rows[i].get("section") or "" for i in work_idx],
            "law_name":      [rows[i].get("law_name") or "" for i in work_idx],
            "article_no":    [rows[i].get("article_no") or "" for i in work_idx],
            "version_date":  [rows[i].get("version_date") or "" for i in work_idx],
            "validity_status":[rows[i].get("validity_status", "valid") or "valid" for i in work_idx],
            "chunk_index":   [int(rows[i].get("chunk_index", 0)) for i in work_idx],
            "text":          [rows[i].get("text") or "" for i in work_idx],
            "embedding":      embs,
        }
        insert_milvus(milvus_col, batch_rows)
        print(f"  [OK] Mongo upsert: {len(mongo_docs)}  Milvus insert: {len(work_idx)}")

    print(f"\n[DONE] NEW={total_new}  CHANGED={total_chg}  SAME={total_same}")

if __name__ == "__main__":
    main()
