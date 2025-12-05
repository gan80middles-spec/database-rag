# -*- coding: utf-8 -*-
"""批量将合同模板切块数据写入 MongoDB 与 Milvus。

参考 `ingest_judgment.py`，支持从切块 JSONL 计算向量、写入 Milvus，
并将 chunk 与文档级元数据写入 MongoDB。
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
            FieldSchema(name="business_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="legal_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="order", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="Contract template chunks with scalar metadata")
        coll = Collection(name=col_name, schema=schema, using="default")
        coll.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 48, "efConstruction": 200}},
        )
    coll = Collection(col_name)
    coll.load()
    return coll


def insert_milvus(coll: Collection, batch_rows: Dict[str, List[Any]]) -> None:
    schema_fields = [f.name for f in coll.schema.fields]
    missing = [name for name in schema_fields if name not in batch_rows]
    extra = [name for name in batch_rows.keys() if name not in schema_fields]

    if missing:
        hint = ""
        if "text" in missing:
            hint = "（集合里仍含 text，说明你没 --recreate 1 重建；请先 drop 再建）"
        raise ValueError(f"[SchemaMismatch] 缺少列: {missing} {hint}")

    if extra:
        print(f"[WARN] 这些列在集合中不存在，将忽略: {extra}")

    data = [batch_rows[name] for name in schema_fields]
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

    # 部分 Mongo 部署在同一实例下已存在名称大小写不同的库时会拒绝再次创建，
    # 例如已有 "lawKB" 而传入 "lawkb" 会抛出 `DatabaseDifferCase`。这里先枚举
    # 已有库，若仅大小写不同则复用已有名称并给出提示，以避免运行时才报错。
    db_names = set(client.list_database_names())
    db_lower_map = {name.lower(): name for name in db_names}
    target_db = db
    existing_db = db_lower_map.get(db.lower())
    if existing_db and existing_db != db:
        print(
            f"[WARN] Database '{existing_db}' already exists (case-insensitive match for '{db}'). "
            "Will reuse existing database name to avoid DatabaseDifferCase error."
        )
        target_db = existing_db

    return client[target_db][col]


def ensure_mongo_chunks(uri: str, db: str, col: str):
    c = ensure_mongo_collection(uri, db, col)
    c.create_index("chunk_id", unique=True)
    c.create_index([("doc_id", 1), ("chunk_index", 1)])
    c.create_index([("doc_type", 1), ("section", 1)])
    c.create_index([("doc_type", 1), ("business_type", 1)])
    return c


def ensure_mongo_docs(uri: str, db: str, col: str):
    c = ensure_mongo_collection(uri, db, col)
    c.create_index("doc_id", unique=True)
    c.create_index([("doc_type", 1), ("business_type", 1)])
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


# ---------- Helpers ----------


def build_doc_record(doc_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": doc_row.get("doc_id", ""),
        "doc_type": doc_row.get("doc_type", "contract_template"),
        "business_type": doc_row.get("business_type", ""),
        "legal_type": doc_row.get("legal_type", ""),
        "title": doc_row.get("title", ""),
        "chunk_count": doc_row.get("chunk_count", 0),
        "length_chars": doc_row.get("length_chars", 0),
        "tags": doc_row.get("tags", []) or [],
        "presence": doc_row.get("presence", {}),
        "text_md5": doc_row.get("text_md5", ""),
    }


def prepare_chunk_doc(row: Dict[str, Any], embedding_dim: int) -> Dict[str, Any]:
    doc = dict(row)
    doc.setdefault("doc_type", "contract_template")
    doc.setdefault("business_type", doc.get("contract_type", ""))
    doc.setdefault("chunk_index", int(doc.get("order", doc.get("chunk_index", 0))))
    doc.setdefault("order", doc.get("chunk_index", 0))
    doc["embedding_dim"] = embedding_dim
    return doc


def build_milvus_rows(sub_rows: List[Dict[str, Any]], sub_emb: List[List[float]]) -> Dict[str, List[Any]]:
    return {
        "chunk_id": [r["chunk_id"] for r in sub_rows],
        "doc_id": [r.get("doc_id", "") for r in sub_rows],
        "doc_type": [r.get("doc_type", "contract_template") for r in sub_rows],
        "business_type": [r.get("business_type") or r.get("contract_type") or "" for r in sub_rows],
        "legal_type": [r.get("legal_type", "") for r in sub_rows],
        "section": [r.get("section") or r.get("clause_no") or "" for r in sub_rows],
        "order": [int(r.get("order", r.get("chunk_index", 0))) for r in sub_rows],
        "chunk_index": [int(r.get("chunk_index", 0)) for r in sub_rows],
        "embedding": sub_emb,
    }


def load_doc_map(docs_files: List[str]) -> Dict[str, Dict[str, Any]]:
    doc_map: Dict[str, Dict[str, Any]] = {}
    for fp in docs_files:
        for row in load_jsonl(fp):
            doc_id = row.get("doc_id")
            if not doc_id:
                continue
            doc_map[doc_id] = build_doc_record(row)
    return doc_map


# ---------- Main ----------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--chunk_pattern", default="*-chunks.jsonl")
    ap.add_argument("--docs_pattern", default="*-docs.jsonl")

    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    ap.add_argument("--mongo_db", default="lawkb")
    ap.add_argument("--mongo_chunk_col", default="contract_kb_chunks")
    ap.add_argument("--mongo_doc_col", default="contract_kb_docs")
    ap.add_argument("--mongo_only", type=int, default=0, help="1=仅写Mongo，不连接Milvus/不计算向量")

    ap.add_argument("--milvus_host", default="127.0.0.1")
    ap.add_argument("--milvus_port", default="19530")
    ap.add_argument("--milvus_token", default="")
    ap.add_argument("--milvus_uri", default="")
    ap.add_argument("--collection", default="contract_kb_chunks")

    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--recreate", type=int, default=0, help="1 = 重建 Milvus 集合")

    args = ap.parse_args()

    if not args.mongo_only:
        embedder = load_embedder(args.model)
        try:
            args.dim = embedder.get_sentence_embedding_dimension()
        except Exception:
            pass

    milvus_col = None
    if not args.mongo_only:
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

    doc_files = list(iter_jsonl_files(args.input_dir, args.docs_pattern))
    doc_map = load_doc_map(doc_files)
    print(f"[INFO] Doc files: {len(doc_files)} (records: {len(doc_map)})")

    chunk_files = list(iter_jsonl_files(args.input_dir, args.chunk_pattern))
    print(f"[INFO] Chunk files: {len(chunk_files)}")

    for fp in chunk_files:
        print(f"\n[FILE] {fp}")
        rows = list(load_jsonl(fp))
        if not rows:
            print("[WARN] empty file, skip")
            continue
        if not args.mongo_only:
            texts = [r.get("text", "") for r in rows]
            embs = encode_batches(embedder, texts, batch_size=args.batch)
        else:
            embs = None

        doc_records: Dict[str, Dict[str, Any]] = {}
        chunk_docs: List[Dict[str, Any]] = []
        for r in rows:
            doc_id = r.get("doc_id", "")
            if doc_id and doc_id not in doc_records:
                if doc_id in doc_map:
                    doc_records[doc_id] = doc_map[doc_id]
                else:
                    doc_records[doc_id] = build_doc_record(r)
            chunk_docs.append(prepare_chunk_doc(r, args.dim))

        BATCH = 2000
        for start in range(0, len(chunk_docs), BATCH):
            sub_rows = chunk_docs[start : start + BATCH]
            upsert_mongo_chunks(mongo_chunks, sub_rows)
            if not args.mongo_only and milvus_col is not None:
                sub_emb = embs[start : start + BATCH]
                batch_rows = build_milvus_rows(sub_rows, sub_emb)
                insert_milvus(milvus_col, batch_rows)

        upsert_mongo_docs(mongo_docs, doc_records.values())

        print(f"[OK] wrote {len(rows)} chunks (doc_ids: {len(doc_records)})")

    print("\n[DONE] All files ingested.")


if __name__ == "__main__":
    main()
