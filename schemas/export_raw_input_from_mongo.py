from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from tqdm import tqdm


DEFAULT_KEEP_SECTIONS = [
    # 你这套 chunker 的 section 命名我按你示例写的
    "开头", "标题", "案号行", "当事人信息", "审理经过",
    "起诉书指控", "证据", "判决理由", "判决结果", "尾部"
]


def read_done_doc_ids(out_jsonl: Path) -> Set[str]:
    """断点续跑：已导出的 doc_id 不再重复导出"""
    done: Set[str] = set()
    if not out_jsonl.exists():
        return done
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                did = obj.get("doc_id") or (obj.get("meta") or {}).get("doc_id")
                if did:
                    done.add(did)
            except Exception:
                continue
    return done


def join_chunks(
    chunks: List[Dict[str, Any]],
    keep_sections: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    把一个 doc_id 的所有 chunks 拼回 text。
    - 默认按 chunk_index 升序拼接（最稳）
    - 可选 keep_sections：只拼关键段落以省 token
    """
    # 排序
    chunks = sorted(chunks, key=lambda x: int(x.get("chunk_index", 0)))

    texts: List[str] = []
    chunk_ids: List[str] = []
    sections: List[str] = []

    for ch in chunks:
        sec = ch.get("section") or ""
        if keep_sections is not None and sec and sec not in keep_sections:
            continue

        t = (ch.get("text") or "").strip()
        if not t:
            continue

        # 你可以加更明显的分隔符，便于大模型定位
        texts.append(f"\n{t}")
        chunk_ids.append(ch.get("chunk_id"))
        sections.append(sec)

    full_text = "\n\n".join(texts).strip()
    return {"text": full_text, "chunk_ids": chunk_ids, "sections": sections}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo_uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db", required=True, help="数据库名，例如 lawKB")
    ap.add_argument("--docs_col", default="law_kb_docs")
    ap.add_argument("--chunks_col", default="law_kb_chunks")

    ap.add_argument("--doc_type", choices=["judgment", "contract", "faq", "all"], default="judgment")
    ap.add_argument("--out", default="raw_input.jsonl")
    ap.add_argument("--err", default="raw_input.errors.jsonl")

    ap.add_argument("--limit", type=int, default=0, help="0 表示不限制")
    ap.add_argument("--batch", type=int, default=200)

    ap.add_argument("--use_chunks", action="store_true", help="从 law_kb_chunks 拼 text（推荐）")
    ap.add_argument("--use_spans_if_exists", action="store_true",
                    help="如果 docs.judgment_info.spans 存在，就用 spans 拼 text（否则用 chunks）")

    ap.add_argument("--keep_sections", default=",".join(DEFAULT_KEEP_SECTIONS),
                    help="仅在 use_chunks 时生效，逗号分隔。留空则拼全部 sections。")
    ap.add_argument("--doc_id", default="", help="指定 doc_id 导出该 doc 的 raw_input")
    args = ap.parse_args()
    print("[ARGS]", vars(args))

    out_path = Path(args.out)
    err_path = Path(args.err)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = read_done_doc_ids(out_path)
    print(f"[INFO] resume mode: already exported {len(done)} docs")

    keep_sections = None
    if args.keep_sections.strip():
        keep_sections = set([x.strip() for x in args.keep_sections.split(",") if x.strip()])

    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    docs: Collection = db[args.docs_col]
    chunks: Collection = db[args.chunks_col]

    query = {}
    if args.doc_type != "all":
        query["doc_type"] = args.doc_type
    if args.doc_id.strip():
        query["doc_id"] = args.doc_id.strip()

    projection = {
        "_id": 0,
        "doc_id": 1,
        "doc_type": 1,
        "case_number": 1,
        "case_subtype": 1,
        "case_system": 1,
        "court": 1,
        "judgment_date": 1,
        "trial_level": 1,
        "doctype_detail": 1,
        "updated_at": 1,
        "created_at": 1,
        "judgment_info.spans": 1,  # 可选走 spans
    }

    print("[QUERY]", query)

    cursor = docs.find(query, projection=projection, batch_size=args.batch)
    if args.limit and args.limit > 0:
        cursor = cursor.limit(args.limit)

    total = docs.count_documents(query) if (args.limit == 0) else args.limit
    pbar = tqdm(total=total, desc="export raw_input")

    with out_path.open("a", encoding="utf-8") as f_out, err_path.open("a", encoding="utf-8") as f_err:
        for d in cursor:
            pbar.update(1)

            doc_id = d.get("doc_id")
            if not doc_id or doc_id in done:
                continue

            try:
                doc_type = d.get("doc_type")

                # ---------- 1) 先尝试用 spans（如果你已经在 docs 里存了三段关键摘要） ----------
                text = ""
                chunk_ids = []
                sections = []

                if args.use_spans_if_exists and doc_type == "judgment":
                    spans = (((d.get("judgment_info") or {}).get("spans")) or [])
                    if isinstance(spans, list) and spans:
                        # 保持顺序：按你存进去的顺序
                        parts = []
                        for sp in spans:
                            field = (sp.get("field") or "").strip()
                            t = (sp.get("text") or "").strip()
                            if t:
                                parts.append(f"\n{t}")
                        text = "\n\n".join(parts).strip()
                        sections = [sp.get("field") for sp in spans if sp.get("text")]
                        chunk_ids = []  # spans 模式下可不填 chunk_ids

                # ---------- 2) 否则用 chunks 拼全文 ----------
                if not text:
                    if not args.use_chunks:
                        raise RuntimeError("text is empty and --use_chunks not enabled")
                    ch_list = list(chunks.find(
                        {"doc_id": doc_id, "doc_type": doc_type},
                        {"_id": 0, "chunk_id": 1, "chunk_index": 1, "section": 1, "text": 1}
                    ))
                    joined = join_chunks(ch_list, keep_sections=keep_sections)
                    text = joined["text"]
                    chunk_ids = joined["chunk_ids"]
                    sections = joined["sections"]

                if not text:
                    raise RuntimeError("empty text after joining chunks/spans")

                row = {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "subtype": d.get("case_subtype") if doc_type == "judgment" else None,
                    "source": "crawler",  # 你们如果有来源字段就替换；没有就先写 crawler/template
                    "text": text,
                    "meta": {
                        "case_number": d.get("case_number"),
                        "court": d.get("court"),
                        "case_system": d.get("case_system"),
                        "judgment_date": d.get("judgment_date"),
                        "trial_level": d.get("trial_level"),
                        "doctype_detail": d.get("doctype_detail"),
                        "chunk_ids": chunk_ids,
                        "sections": sections,
                        "export_from": args.docs_col,
                    },
                }

                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                done.add(doc_id)

            except Exception as e:
                f_err.write(json.dumps({
                    "doc_id": doc_id,
                    "doc_type": d.get("doc_type"),
                    "error": f"{type(e).__name__}: {e}"
                }, ensure_ascii=False) + "\n")
    pbar.close()
    print(f"[OK] wrote: {out_path.resolve()}")
    print(f"[OK] errors: {err_path.resolve()}")

if __name__ == "__main__":
    main()
