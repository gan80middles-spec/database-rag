# link_builder_docs_from_statutes.py
# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Tuple, Iterable, Optional
from pymongo import MongoClient, UpdateOne

# === 连接配置 ===
MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "lawkb"
COL_DOCS  = "law_kb_docs"
COL_CHUNKS= "law_kb_chunks"
COL_LINKS = "law_kb_links"

# 允许作为被连到的目标文种（整部法律/整份解释）
TARGET_DOC_TYPES = {"statute", "judicial_interpretation"}

# 《……》第……条 解析（回退用）
PAT = re.compile(r"《\s*([^》]+?)\s*》第([〇零一二三四五六七八九十百千两\d]+)条")

CN_MAP = {"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"百":100,"千":1000}
def cn_to_int(s: str) -> int:
    s = str(s or "").strip().replace("第","").replace("条","")
    if s.isdigit(): return int(s)
    total, tmp = 0, 0
    for ch in s:
        if ch in "十百千":
            unit = CN_MAP[ch]
            total = (total or 1) * unit
            tmp = 0
        else:
            v = CN_MAP.get(ch, 0)
            total += v
            tmp = v
    return total or tmp or 0

def norm_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("《","").replace("》","")
    s = re.sub(r"^中华人民共和国", "", s)  # 去“中华人民共和国”前缀
    s = re.sub(r"\s+", "", s)
    return s

def ensure_indexes(E):
    E.create_index([("from_doc",1)])
    E.create_index([("to_doc",1)])
    E.create_index([("edge",1)])
    # 若你希望强约束去重，可在数据清洗后改为 unique 复合索引：
    # E.create_index([( "from_doc",1),("to_doc",1),("edge",1)], unique=True)

def load_target_docs(D, C) -> Dict[str, List[Tuple[str,str]]]:
    """
    预构建：规范化法名 -> [(doc_id, doc_type)]
    优先用 docs 的 law_name/law_alias；若没有别名，退回到 chunks 汇总别名。
    """
    name2docs: Dict[str, List[Tuple[str,str]]] = {}
    # 1) 先扫 docs
    for d in D.find({"doc_type": {"$in": list(TARGET_DOC_TYPES)}},
                    {"doc_id":1,"doc_type":1,"law_name":1,"law_alias":1}):
        doc_id, dt = d["doc_id"], d["doc_type"]
        ln = norm_name(d.get("law_name",""))
        if ln:
            name2docs.setdefault(ln, []).append((doc_id, dt))
        for al in d.get("law_alias") or []:
            name2docs.setdefault(norm_name(al), []).append((doc_id, dt))
    # 2) 用 chunks 做别名补充
    cur = C.find({"doc_type":{"$in": list(TARGET_DOC_TYPES)}},
                 {"doc_id":1,"law_name":1,"law_alias":1})
    for d in cur:
        doc_id = d.get("doc_id")
        ln = norm_name(d.get("law_name",""))
        if ln and (doc_id, None) not in name2docs.get(ln, []):
            name2docs.setdefault(ln, []).append((doc_id, ""))
        for al in d.get("law_alias") or []:
            name2docs.setdefault(norm_name(al), []).append((doc_id, ""))
    return name2docs

def parse_statutes_from_doc(jdoc: dict) -> Iterable[Tuple[str,int]]:
    """
    优先从 judgment_info.statutes 中取结构化 {law, article}；
    若没有，则从顶层 statutes 字符串列表用正则回退解析。
    返回 (law_name_raw, article_no_int)
    """
    # 结构化优先
    st = (((jdoc.get("judgment_info") or {}).get("statutes")) or [])
    for item in st:
        law = item.get("law") or item.get("name") or ""
        art = item.get("article")
        if isinstance(art, str):
            art = cn_to_int(art)
        if isinstance(art, int) and law:
            yield (law, art)

    # 回退：顶层 statutes 是“《××法》第××条”形式
    if not st:
        for s in jdoc.get("statutes") or []:
            m = PAT.search(s or "")
            if not m: 
                continue
            law_raw, art_raw = m.group(1), m.group(2)
            yield (law_raw, cn_to_int(art_raw))

def main():
    mc = MongoClient(MONGO_URI)
    DB = mc[DB_NAME]
    D  = DB[COL_DOCS]
    C  = DB[COL_CHUNKS]
    E  = DB[COL_LINKS]
    ensure_indexes(E)

    # 预加载候选：规范名 -> 多个 doc_id
    name2docs = load_target_docs(D, C)

    # 仅处理“判决书/裁定书/调解书”等案件文书；这里以 judgment 为例
    q = {"doc_type": "judgment"}
    fields = {"doc_id":1,"doc_type":1,"judgment_info":1,"statutes":1}
    cur = list(D.find(q, fields))
    print(f"[INFO] judgment docs: {len(cur)}")

    ops: List[UpdateOne] = []
    seen = set()
    miss: Dict[str,int] = {}

    for j in cur:
        from_doc = j["doc_id"]
        # 聚合同一法律的条文
        law_to_articles: Dict[str, set] = {}
        law_to_rawname: Dict[str, str] = {}

        for law_raw, art in parse_statutes_from_doc(j):
            key = norm_name(law_raw)
            if not key:
                continue
            if key not in law_to_articles:
                law_to_articles[key] = set()
                law_to_rawname[key] = law_raw
            if art:
                law_to_articles[key].add(int(art))

        for lname_norm, arts in law_to_articles.items():
            targets = name2docs.get(lname_norm) or []
            if not targets:
                miss[lname_norm] = miss.get(lname_norm, 0) + 1
                continue
            # 同名可能映射多个（如主法/解释），全连
            for to_doc, _dt in targets:
                key = (from_doc, to_doc, "applies")
                if key in seen: 
                    continue
                seen.add(key)
                ops.append(
                    UpdateOne(
                        {"from_doc": from_doc, "to_doc": to_doc, "edge": "applies"},
                        {
                            "$setOnInsert": {"law_name_raw": law_to_rawname[lname_norm]},
                            "$addToSet": {"articles": {"$each": sorted(arts)}},
                        },
                        upsert=True,
                    )
                )

    if ops:
        res = E.bulk_write(ops, ordered=False)
        print(f"[OK] upsert doc-edges: {res.upserted_count} upserted / {res.modified_count} modified / total_ops={len(ops)}")
    else:
        print("[OK] nothing to link")

    if miss:
        top = sorted(miss.items(), key=lambda x: -x[1])[:10]
        print("[MISSING LAW NAMES] top 10 (norm):")
        for k, v in top:
            print(f"  - {k} ×{v}")

if __name__ == "__main__":
    main()
