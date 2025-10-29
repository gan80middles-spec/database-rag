# -*- coding: utf-8 -*-
"""Aggregate law chunks into document-level records."""
import re
from collections import Counter, defaultdict

from pymongo import MongoClient, UpdateOne

MONGO_URI = "mongodb://adminUser:~Q2w3e4r@192.168.110.36:27019"
DB_NAME = "lawKB"
SRC_COL = "law_kb_chunks"
DST_COL = "law_kb_docs"

CN_PREFIX = re.compile(r"^\s*中华人民共和国")
BRACKETS = re.compile(r"[()（）\[\]【】＜＞<>]")
MULTI_WS = re.compile(r"\s+")


def norm_name(s: str) -> str:
    if not s:
        return ""
    s = s.strip().replace("《", "").replace("》", "")
    s = CN_PREFIX.sub("", s)
    s = re.sub(r"（[^）]*修正[^）]*）", "", s)
    s = re.sub(r"\([^)]*修正[^)]*\)", "", s)
    s = BRACKETS.sub("", s)
    s = s.replace("　", " ").replace("·", " ")
    s = MULTI_WS.sub("", s)
    return s


mc = MongoClient(MONGO_URI)
src = mc[DB_NAME][SRC_COL]
dst = mc[DB_NAME][DST_COL]

# 幂等索引
dst.create_index("doc_id", unique=True)
dst.create_index("doc_type")
dst.create_index("law_name_norm")
dst.create_index("law_alias_norm")

q = {"doc_type": {"$in": ["statute", "judicial_interpretation"]}}
fields = {"doc_id": 1, "doc_type": 1, "law_name": 1, "version_date": 1}
cur = src.find(q, fields)

bucket = defaultdict(list)
for d in cur:
    if not d.get("doc_id"):
        continue
    bucket[(d["doc_id"], d.get("doc_type", ""))].append(d)

ops = []
for (doc_id, doc_type), rows in bucket.items():
    names = [r.get("law_name", "") for r in rows if r.get("law_name")]
    vdates = [r.get("version_date", "") for r in rows if r.get("version_date")]
    main_name = Counter(names).most_common(1)[0][0] if names else ""
    name_norm = norm_name(main_name)
    alias_norm = sorted({name_norm})

    payload = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "law_name": main_name,
        "law_name_norm": name_norm,
        "law_alias": [],
        "law_alias_norm": alias_norm,
        "version_date": vdates[0] if vdates else "",
    }
    ops.append(
        UpdateOne(
            {"doc_id": doc_id},
            {
                "$setOnInsert": payload,
                "$set": {
                    "law_name_norm": name_norm,
                    "version_date": payload["version_date"] or "",
                },
            },
            upsert=True,
        )
    )

if ops:
    res = dst.bulk_write(ops, ordered=False)
    print(
        f"[OK] upserted={getattr(res, 'upserted_count', 0)}, "
        f"modified={res.modified_count}, total={len(ops)}"
    )
else:
    print("[OK] nothing to upsert")
