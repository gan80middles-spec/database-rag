# -*- coding: utf-8 -*-
"""
读取切块 JSONL → 计算嵌入 → 写入 Mongo + Milvus
每行 JSON 至少包含：
chunk_id, doc_id, doc_type, law_name, article_no, version_date,
validity_status, chunk_index, text, （可选）path

用法示例：
python ingest_kb.py \
  --input_dir 20_chunks \
  --pattern "*.jsonl" \
  --mongo_uri "mongodb://localhost:27017" --mongo_db lawkb --mongo_col law_kb_chunks \
  --milvus_host 127.0.0.1 --milvus_port 19530 --collection law_kb_v1 \
  --model BAAI/bge-m3 --batch 64 --recreate 0
"""
import os, glob, json, argparse, math
from typing import List, Dict, Any
from tqdm import tqdm
import json
from pymongo import MongoClient, UpdateOne
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from sentence_transformers import SentenceTransformer

# ---------- Embedding ----------
def load_embedder(model_name: str, device: str = None):
    # trust_remote_code=True 以支持 bge-m3
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    return model

def encode_batches(model, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        sub = texts[i:i+batch_size]
        vec = model.encode(sub, normalize_embeddings=True, batch_size=len(sub), show_progress_bar=False)
        embs.extend(v.tolist() for v in vec)
    return embs

# ---------- Milvus ----------
def ensure_milvus_collection(col_name: str, dim: int,
                             host: str = None, port: str = None,
                             recreate: bool = False,
                             token: str = None, uri: str = None):
    if uri:
        connections.connect("default", uri = uri, token = token)
    else:
        connections.connect("default", host=host, port=port, token=token)

    if recreate and utility.has_collection(col_name):
        utility.drop_collection(col_name)
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),      # statute/judgement/contract/faq/case
            FieldSchema(name="case_type", dtype=DataType.VARCHAR, max_length=32),     # 民事/刑事/行政/赔偿/执行（judgement用）
            FieldSchema(name="court", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="cause", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="contract_type", dtype=DataType.VARCHAR, max_length=64), # 劳动/劳务/买卖/租赁/服务…
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=64),       # facts/opinion/result/holding/条款-xxx/qa 等
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
    
    try:
        coll.insert(data)
    except Exception as e:
        pks = [str(x) for x in batch_rows["chunk_id"]]  # 确保都是字符串
        if pks:
            expr = f"chunk_id in {json.dumps(pks, ensure_ascii=False)}"
            coll.delete(expr)
            coll.insert(data)

# ---------- Mongo ----------
def ensure_mongo(uri: str, db: str, col: str):
    mc = MongoClient(uri)
    c = mc[db][col]
    # 主键 & 顺序
    c.create_index("chunk_id", unique=True)
    c.create_index([("doc_id", 1), ("chunk_index", 1)])

    # 通用过滤
    c.create_index([("doc_type", 1), ("section", 1)])
    c.create_index([("version_date", 1), ("validity_status", 1)])

    # statutes：按法名/条号/版本
    c.create_index([("doc_type", 1), ("law_name", 1), ("article_no", 1), ("version_date", 1)])

    # judgments：按类型/法院/案由
    c.create_index([("doc_type", 1), ("case_type", 1), ("court", 1), ("cause", 1)])

    # contracts：按合同类型/条款分类
    c.create_index([("doc_type", 1), ("contract_type", 1), ("section", 1)])

    # 可选：更新时间排序
    c.create_index([("updated_at", -1)], sparse=True)
    return c


def upsert_mongo_chunks(col, docs: List[Dict[str, Any]]):
    ops = []
    for d in docs:
        ops.append(UpdateOne({"chunk_id": d["chunk_id"]}, {"$set": d}, upsert=True))
    if ops:
        col.bulk_write(ops, ordered=False)

# ---------- IO ----------
def iter_jsonl_files(input_dir: str, pattern: str):
    for fp in sorted(glob.glob(os.path.join(input_dir, pattern))):
        yield fp

def load_jsonl(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    ap.add_argument("--mongo_db", default="lawkb")
    ap.add_argument("--mongo_col", default="law_kb_chunks")
    ap.add_argument("--milvus_host", default="127.0.0.1")
    ap.add_argument("--milvus_port", default="19530")
    ap.add_argument("--milvus_token", default="", help="e.g. root:Milvus")
    ap.add_argument("--milvus_uri", default="", help="optional: http(s)://host:19530")
    ap.add_argument("--collection", default="law_kb_v1")
    ap.add_argument("--model", default="BAAI/bge-large-zh-v1.5")  # 或 BAAI/bge-m3
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--recreate", type=int, default=0, help="1=重建 Milvus 集合")
    args = ap.parse_args()

    # init
    embedder = load_embedder(args.model)
    try:
        args.dim = embedder.get_sentence_embedding_dimension()
    except Exception:
        pass
    milvus_col = ensure_milvus_collection(
        args.collection, args.dim,
        host=args.milvus_host, port=args.milvus_port,
        recreate=bool(args.recreate),
        token=(args.milvus_token or None),
        uri=(args.milvus_uri or None),
    )
    mongo_col = ensure_mongo(args.mongo_uri, args.mongo_db, args.mongo_col)

    files = list(iter_jsonl_files(args.input_dir, args.pattern))
    print(f"[INFO] Files: {len(files)}")

    for fp in files:
        print(f"\n[FILE] {fp}")
        # 先把文件全部读入内存（分批嵌入 & 入库）
        rows = list(load_jsonl(fp))
        print(f"[INFO] chunks: {len(rows)}")

        # 计算嵌入
        texts = [r["text"] for r in rows]
        embs = encode_batches(embedder, texts, batch_size=args.batch)

        # 分批写入（避免一次插太多）
        B = 2000
        for i in range(0, len(rows), B):
            sub = rows[i:i+B]
            sub_emb = embs[i:i+B]

            # --- Mongo 文档（保留 path 等扩展字段）---
            mongo_docs = []
            for r, v in zip(sub, sub_emb):
                doc = dict(r)
                doc["embedding_dim"] = args.dim
                mongo_docs.append(doc)
            upsert_mongo_chunks(mongo_col, mongo_docs)

            # --- Milvus 行列 ---
            batch_rows = {
                "chunk_id":      [r["chunk_id"] for r in sub],
                "doc_id":        [r.get("doc_id", "") for r in sub],
                "doc_type":      [r.get("doc_type", "") for r in sub],
                "case_type":     [r.get("case_type") or "" for r in sub],
                "court":         [r.get("court") or "" for r in sub],
                "cause":         [r.get("cause") or "" for r in sub],
                "contract_type": [r.get("contract_type") or "" for r in sub],
                "section":       [r.get("section") or "" for r in sub],
                "law_name":      [r.get("law_name") or "" for r in sub],
                "article_no":    [r.get("article_no") or "" for r in sub],
                "version_date":  [r.get("version_date") or "" for r in sub],
                "validity_status":[r.get("validity_status", "valid") or "valid" for r in sub],
                "chunk_index":   [int(r.get("chunk_index", 0)) for r in sub],
                "text":          [r.get("text") or "" for r in sub],
                "embedding":      sub_emb,
            }

            insert_milvus(milvus_col, batch_rows)

        print(f"[OK] wrote Mongo+Milvus for {fp}")

    print("\n[DONE] All files ingested.")

if __name__ == "__main__":
    main()
