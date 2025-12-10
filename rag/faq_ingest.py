#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
faq_ingest.py

用途：
    将咨询问答（FAQ）数据写入：
        - MongoDB: law_kb_chunks（文档真源）
        - Milvus : faq_kb_chunks（FAQ 专用向量库，不存正文）

FAQ JSONL 示例（每行一条）：
{
  "id": "417f7220-732e-4362-959f-73164244afc0",
  "consult_time": "2025-07-14 09:00",
  "consult_type": "律师咨询 - 劳动人事",
  "question_title": "拖欠2025年4月5月的工资",
  "question_text": "公司2025年4月份/5月份工资到今天还没有发完，而且公司答应是6月底将4月份工资发完，但只发百分之三十。",
  "answer_text": "公司无故拖欠工资，是违法的……",
  "reply_time": "2025-07-14 09:35",
  "consult_category": "labor"
}

Milvus FAQ collection schema（自动创建）：
    - chunk_id         VarChar(256)  PK
    - doc_type         VarChar(32)
    - consult_category VarChar(32)
    - embedding        FloatVector(dim)

使用示例：

    python faq_ingest.py \
      --jsonl /path/to/consults_extracted_labeled.jsonl \
      --mongo-uri "mongodb://adminUser:pwd@host:27019" \
      --mongo-db "lawKB" \
      --mongo-coll "law_kb_chunks" \
      --milvus-host "120.46.59.93" \
      --milvus-port 19530 \
      --milvus-collection "faq_kb_chunks" \
      --milvus-token 'root:xxx' \
      --embed-model "/path/to/bge-large-zh-v1.5"
"""

import argparse
import datetime
import json
from typing import List, Optional

from pymongo import MongoClient
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ====== 1. 向量模型封装 ======

class EmbeddingClient:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"[EMB] Loading embedding model from {model_path} on {device} ...")
        self.model = SentenceTransformer(model_path, device=device)

    def encode(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embs.tolist()


# ====== 2. content 构造 ======

def build_content(q_title: str, q_text: str, a_text: str) -> str:
    """把问答拼成一个用于向量的 content 字段（不入 Milvus，只入 Mongo）"""
    parts = []
    if q_title:
        parts.append(f"【问题】{q_title.strip()}")
    if q_text:
        parts.append(f"【补充说明】{q_text.strip()}")
    if a_text:
        parts.append(f"【解答】{a_text.strip()}")
    return "\n".join(parts)


# ====== 3. Milvus：创建 / 获取 FAQ collection ======

def get_or_create_faq_collection(name: str, dim: int) -> Collection:
    """
    在 Milvus 中获取或创建 FAQ 专用 collection。
    schema:
        - chunk_id         VarChar(256)  PK
        - doc_type         VarChar(32)
        - consult_category VarChar(32)
        - embedding        FloatVector(dim)
    """
    if utility.has_collection(name):
        c = Collection(name)
        c.load()
        print(f"[Milvus] Use existing collection '{name}'")
        return c

    print(f"[Milvus] Creating new collection '{name}' for FAQ ...")

    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=256,
        ),
        FieldSchema(
            name="doc_type",
            dtype=DataType.VARCHAR,
            max_length=32,
        ),
        FieldSchema(
            name="consult_category",
            dtype=DataType.VARCHAR,
            max_length=32,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="FAQ KB chunks (咨询问答向量库，无正文)",
    )

    c = Collection(name=name, schema=schema)

    # 建一个简单的向量索引
    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 8, "efConstruction": 64},
    }
    c.create_index(field_name="embedding", index_params=index_params)
    c.load()

    print(f"[Milvus] Collection '{name}' created and loaded.")
    return c


# ====== 4. 主逻辑 ======

def ingest_faq(
    jsonl_path: str,
    mongo_uri: str,
    mongo_db: str,
    mongo_coll: str,
    milvus_host: str,
    milvus_port: int,
    milvus_collection: str,
    embed_model_path: str,
    embed_device: str = "cuda",
    dry_run: bool = False,
    milvus_token: Optional[str] = None,
    embed_dim: int = 1024,
):
    # --- 读取 JSONL ---
    print(f"[DATA] Loading JSONL from {jsonl_path} ...")
    records_raw = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records_raw.append(obj)
    print(f"[DATA] Loaded {len(records_raw)} FAQ records")

    # --- 构造 Mongo 文档 & content ---
    docs = []
    contents: List[str] = []

    now = datetime.datetime.utcnow()

    for i, obj in enumerate(records_raw):
        consult_id = obj.get("id")
        consult_time = obj.get("consult_time")
        consult_type = obj.get("consult_type")
        consult_category = obj.get("consult_category") or "other"

        q_title = obj.get("question_title", "")
        q_text = obj.get("question_text", "")
        a_text = obj.get("answer_text", "")
        reply_time = obj.get("reply_time")

        content = build_content(q_title, q_text, a_text)

        # 唯一 chunk_id：优先用原始 id
        if consult_id:
            chunk_id = f"faq_{consult_id}"
        else:
            chunk_id = f"faq_auto_{i}"

        doc = {
            "chunk_id": chunk_id,           # 用于 Mongo 唯一索引 & Milvus 映射
            "doc_type": "faq",
            "consult_time": consult_time,
            "consult_type": consult_type,
            "consult_category": consult_category,
            "question_title": q_title,
            "question_text": q_text,
            "answer_text": a_text,
            "reply_time": reply_time,
            "content": content,
            "created_at": now,
        }

        docs.append(doc)
        contents.append(content)

    print(f"[DATA] Prepared {len(docs)} docs")

    if dry_run:
        print(f"[DRY-RUN] Would insert {len(docs)} docs into Mongo "
              f"and {len(contents)} vectors into Milvus collection '{milvus_collection}'.")
        return

    # --- Mongo 连接 & 写入 ---
    print(f"[Mongo] Connecting to {mongo_uri} / db={mongo_db}, coll={mongo_coll} ...")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[mongo_db]
    coll = db[mongo_coll]

    print(f"[Mongo] Inserting {len(docs)} docs into {mongo_coll} ...")
    result = coll.insert_many(docs, ordered=False)
    print(f"[Mongo] Inserted {len(result.inserted_ids)} documents.")

    # --- Milvus 连接 & collection 准备 ---
    print(f"[Milvus] Connecting to {milvus_host}:{milvus_port} ...")
    connections.connect(
        "default",
        host=milvus_host,
        port=str(milvus_port),
        token=milvus_token,
    )
    m_collection = get_or_create_faq_collection(milvus_collection, dim=embed_dim)

    # --- 向量模型 ---
    emb_client = EmbeddingClient(embed_model_path, device=embed_device)

    # --- 计算向量 ---
    print(f"[EMB] Encoding {len(contents)} FAQ contents ...")
    vectors = emb_client.encode(contents)

    if len(vectors) != len(docs):
        raise RuntimeError(f"vectors({len(vectors)}) != docs({len(docs)})")

    # --- 组装行插入 Milvus ---
    print(f"[Milvus] Inserting {len(vectors)} rows into collection '{milvus_collection}' ...")

    rows = []
    for doc, vec in zip(docs, vectors):
        rows.append({
            "chunk_id": doc["chunk_id"],
            "doc_type": "faq",
            "consult_category": doc.get("consult_category") or "other",
            "embedding": vec,
        })

    m_collection.insert(rows)
    m_collection.flush()
    print("[DONE] FAQ ingest finished.")


# ====== 5. CLI ======

def main():
    parser = argparse.ArgumentParser(
        description="Ingest FAQ JSONL into Mongo (law_kb_chunks) + Milvus (faq_kb_chunks)."
    )
    parser.add_argument("--jsonl", required=True, help="Path to FAQ JSONL file.")

    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--mongo-db", default="lawKB")
    parser.add_argument("--mongo-coll", default="law_kb_chunks")

    parser.add_argument("--milvus-host", default="127.0.0.1")
    parser.add_argument("--milvus-port", type=int, default=19530)
    parser.add_argument(
        "--milvus-collection",
        default="faq_kb_chunks",
        help="Milvus collection name for FAQ vectors.",
    )
    parser.add_argument("--milvus-token", default=None, help="Milvus auth token if enabled.")

    parser.add_argument(
        "--embed-model",
        required=True,
        help="Path or name of embedding model (e.g. bge-large-zh-v1.5).",
    )
    parser.add_argument("--embed-device", default="cuda", help="cuda / cpu")
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=1024,
        help="Embedding dimension, default 1024 for bge-large-zh-v1.5/bge-m3.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print stats, do not write to Mongo/Milvus.",
    )

    args = parser.parse_args()

    ingest_faq(
        jsonl_path=args.jsonl,
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        mongo_coll=args.mongo_coll,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        milvus_collection=args.milvus_collection,
        embed_model_path=args.embed_model,
        embed_device=args.embed_device,
        dry_run=args.dry_run,
        milvus_token=args.milvus_token,
        embed_dim=args.embed_dim,
    )


if __name__ == "__main__":
    main()
