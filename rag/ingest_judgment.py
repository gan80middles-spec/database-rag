# -*- coding: utf-8 -*-
"""将判决文书切块入库到 MongoDB 与 Milvus。

工作流程：

1. 读取 judgment_chunker.py 生成的 JSONL 文件；
2. 为每个 chunk 计算向量，并写入：
   - MongoDB（law_kb_chunks）：保存文本与 chunk 级元数据；
   - Milvus（law_kb_chunk）：保存向量与轻量标量字段；
3. 汇总 judgment_info，按 doc_id 写入 MongoDB（law_kb_docs）。

命令示例：

python ingest_judgment.py \
  --input_dir chunks/judgments \
  --pattern "*.jsonl" \
  --mongo_uri "mongodb://localhost:27017" --mongo_db lawkb \
  --milvus_host 127.0.0.1 --milvus_port 19530 --collection law_kb_chunk \
  --model BAAI/bge-m3 --batch 64
"""

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------- Embedding ----------


def load_embedder(model_name: str, device: Optional[str] = None) -> SentenceTransformer:
    """Load a sentence-transformers model (with trust_remote_code enabled)."""

    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


def encode_batches(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Encode texts in batches to avoid OOM."""

    embeddings: List[List[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        sub = texts[start : start + batch_size]
        vec = model.encode(
            sub,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        embeddings.extend(v.tolist() for v in vec)
    return embeddings


# ---------- Milvus ----------


def ensure_milvus_collection(
    col_name: str,
    dim: int,
    host: Optional[str] = None,
    port: Optional[str] = None,
    recreate: bool = False,
    token: Optional[str] = None,
    uri: Optional[str] = None,
) -> Collection:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

    if uri:
        connections.connect("default", uri=uri, token=token)
    else:
        connections.connect("default", host=host, port=port, token=token)

    if recreate and utility.has_collection(col_name):
        utility.drop_collection(col_name)

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
            # ⚠️ 删除了 text
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="LawLLM judgment chunks (no raw text)")
        coll = Collection(name=col_name, schema=schema, using="default")
        coll.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 48, "efConstruction": 200}},
        )
    coll = Collection(col_name)
    coll.load()
    return coll


def insert_milvus(coll: Collection, batch_rows: Dict[str, List[Any]]) -> None:
    schema_fields = [f.name for f in coll.schema.fields]  # 以集合的字段顺序为准
    missing = [name for name in schema_fields if name not in batch_rows]
    extra = [name for name in batch_rows.keys() if name not in schema_fields]

    if missing:
        # 如果缺的是 text，基本就是没重建成功
        hint = ""
        if "text" in missing:
            hint = "（集合里仍含 text，说明你没 --recreate 1 重建；请先 drop 再建）"
        raise ValueError(f"[SchemaMismatch] 缺少列: {missing} {hint}")

    if extra:
        # 允许日志提示但不中断，也可以选择严格抛错
        print(f"[WARN] 这些列在集合中不存在，将忽略: {extra}")

    data = [batch_rows[name] for name in schema_fields]  # 严格按顺序组装
    try:
        coll.insert(data)
    except Exception:
        import json as _json

        pk_list = [str(x) for x in batch_rows["chunk_id"]]
        coll.delete(f"chunk_id in {_json.dumps(pk_list, ensure_ascii=False)}")
        coll.insert(data)


# ---------- Mongo ----------


def ensure_mongo_collection(uri: str, db: str, col: str):
    client = MongoClient(uri)
    return client[db][col]


def ensure_mongo_chunks(uri: str, db: str, col: str):
    c = ensure_mongo_collection(uri, db, col)
    c.create_index("chunk_id", unique=True)
    c.create_index([("doc_id", 1), ("chunk_index", 1)])
    c.create_index([("doc_type", 1), ("section", 1)])
    c.create_index([("case_system", 1), ("case_subtype", 1)])
    c.create_index([("court", 1), ("judgment_date", -1)])
    return c


def ensure_mongo_docs(uri: str, db: str, col: str):
    c = ensure_mongo_collection(uri, db, col)
    c.create_index("doc_id", unique=True)
    c.create_index([("doc_type", 1), ("case_system", 1)])
    c.create_index([("court", 1), ("judgment_date", -1)])
    return c


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def upsert_mongo_chunks(col, docs: Iterable[Dict[str, Any]]) -> None:
    ops: List[UpdateOne] = []
    timestamp = now_iso()
    for d in docs:
        payload = dict(d)
        payload["updated_at"] = timestamp
        if "created_at" not in payload:
            payload["created_at"] = timestamp
        ops.append(UpdateOne({"chunk_id": payload["chunk_id"]}, {"$set": payload}, upsert=True))
    if ops:
        col.bulk_write(ops, ordered=False)


def upsert_mongo_docs(col, docs: Iterable[Dict[str, Any]]) -> None:
    ops: List[UpdateOne] = []
    timestamp = now_iso()
    for d in docs:
        doc_id = d.get("doc_id")
        if not doc_id:
            continue
        payload = dict(d)
        payload["updated_at"] = timestamp
        update = {"$set": payload, "$setOnInsert": {"created_at": timestamp}}
        ops.append(UpdateOne({"doc_id": doc_id}, update, upsert=True))
    if ops:
        col.bulk_write(ops, ordered=False)


# ---------- IO ----------


def iter_jsonl_files(input_dir: str, pattern: str) -> Iterable[str]:
    search_path = os.path.join(input_dir, "**", pattern)
    for fp in sorted(glob.glob(search_path, recursive=True)):
        if os.path.isfile(fp):
            yield fp


def load_jsonl(fp: str) -> Iterable[Dict[str, Any]]:
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------- Main ----------


def build_doc_record(row: Dict[str, Any]) -> Dict[str, Any]:
    doc_info = row.get("judgment_info") or {}
    return {
        "doc_id": row.get("doc_id", ""),
        "doc_type": row.get("doc_type", ""),
        "doctype_detail": row.get("doctype_detail", ""),
        "case_system": row.get("case_system", ""),
        "case_subtype": row.get("case_subtype", ""),
        "trial_level": row.get("trial_level", ""),
        "court": row.get("court", ""),
        "case_number": row.get("case_number", ""),
        "judgment_date": row.get("judgment_date", ""),
        "statutes": row.get("statutes", []) or [],
        "judgment_info": doc_info,
    }


def prepare_chunk_doc(row: Dict[str, Any], embedding_dim: int) -> Dict[str, Any]:
    doc = dict(row)
    doc.pop("judgment_info", None)
    doc["embedding_dim"] = embedding_dim
    doc.setdefault("case_system", doc.get("case_type", ""))
    doc.setdefault("case_subtype", doc.get("cause", ""))
    return doc


def build_milvus_rows(sub_rows: List[Dict[str, Any]], sub_emb: List[List[float]]) -> Dict[str, List[Any]]:
    return {
        "chunk_id": [r["chunk_id"] for r in sub_rows],
        "doc_id": [r.get("doc_id", "") for r in sub_rows],
        "doc_type": [r.get("doc_type", "") for r in sub_rows],
        "case_type": [r.get("case_system") or r.get("case_type") or "" for r in sub_rows],
        "court": [r.get("court") or "" for r in sub_rows],
        "cause": [r.get("case_subtype") or r.get("cause") or "" for r in sub_rows],
        "contract_type": ["" for _ in sub_rows],
        "section": [r.get("section") or "" for r in sub_rows],
        "law_name": [r.get("law_name") or "" for r in sub_rows],
        "article_no": [r.get("article_no") or "" for r in sub_rows],
        "version_date": [r.get("version_date") or "" for r in sub_rows],
        "validity_status": [r.get("validity_status") or "valid" for r in sub_rows],
        "chunk_index": [int(r.get("chunk_index", 0)) for r in sub_rows],
        # ⚠️ 删除了 "text"
        "embedding": sub_emb,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", default="*.jsonl")

    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    ap.add_argument("--mongo_db", default="lawkb")
    ap.add_argument("--mongo_chunk_col", default="law_kb_chunks")
    ap.add_argument("--mongo_doc_col", default="law_kb_docs")

    ap.add_argument("--milvus_host", default="127.0.0.1")
    ap.add_argument("--milvus_port", default="19530")
    ap.add_argument("--milvus_token", default="")
    ap.add_argument("--milvus_uri", default="")
    ap.add_argument("--collection", default="law_kb_chunk")

    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--recreate", type=int, default=0, help="1 = 重建 Milvus 集合")

    args = ap.parse_args()

    embedder = load_embedder(args.model)
    try:
        args.dim = embedder.get_sentence_embedding_dimension()
    except Exception:
        pass

    milvus_col = ensure_milvus_collection(
        args.collection,
        args.dim,
        host=args.milvus_host,
        port=args.milvus_port,
        recreate=bool(args.recreate),
        token=args.milvus_token or None,
        uri=args.milvus_uri or None,
    )

    mongo_chunks = ensure_mongo_chunks(args.mongo_uri, args.mongo_db, args.mongo_chunk_col)
    mongo_docs = ensure_mongo_docs(args.mongo_uri, args.mongo_db, args.mongo_doc_col)

    files = list(iter_jsonl_files(args.input_dir, args.pattern))
    print(f"[INFO] Files: {len(files)}")

    for fp in files:
        print(f"\n[FILE] {fp}")
        rows = list(load_jsonl(fp))
        if not rows:
            print("[WARN] empty file, skip")
            continue

        texts = [r.get("text", "") for r in rows]
        embs = encode_batches(embedder, texts, batch_size=args.batch)

        doc_records: Dict[str, Dict[str, Any]] = {}
        chunk_docs: List[Dict[str, Any]] = []
        for r in rows:
            doc_id = r.get("doc_id", "")
            if doc_id and doc_id not in doc_records:
                doc_records[doc_id] = build_doc_record(r)
            chunk_docs.append(prepare_chunk_doc(r, args.dim))

        BATCH = 2000
        for start in range(0, len(chunk_docs), BATCH):
            sub_rows = chunk_docs[start : start + BATCH]
            sub_emb = embs[start : start + BATCH]

            upsert_mongo_chunks(mongo_chunks, sub_rows)
            batch_rows = build_milvus_rows(sub_rows, sub_emb)
            insert_milvus(milvus_col, batch_rows)

        upsert_mongo_docs(mongo_docs, doc_records.values())

        print(f"[OK] wrote {len(rows)} chunks (doc_ids: {len(doc_records)})")

    print("\n[DONE] All files ingested.")


if __name__ == "__main__":
    main()

