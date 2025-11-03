# link_builder_docs_from_statutes.py
# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Tuple, Iterable, Optional, Any, Set
from pymongo import MongoClient, UpdateOne

# —— 处理“关于适用《…法》的解释”内嵌法名丢失的问题
INNER_LAW = re.compile(r"关于适用\s*[《〈<]\s*([^》〉>]+?)\s*[》〉>]\s*的解释")


def fix_interpretation_title(raw: str) -> str:
    s = str(raw or "").strip()
    m = INNER_LAW.search(s)
    if m:
        inner = m.group(1)
        # 去掉内层书名号，仅保留内容
        inner_fixed = (
            inner.replace("《", "")
            .replace("》", "")
            .replace("〈", "")
            .replace("〉", "")
            .replace("<", "")
            .replace(">", "")
            .strip()
        )
        # 回填一个“完整标题”，避免出现“关于适用的解释”
        s = INNER_LAW.sub(f"关于适用{inner_fixed}的解释", s)
    return s

# === 连接配置 ===
MONGO_URI = "mongodb://adminUser:~Q2w3e4r@192.168.110.36:27019"
DB_NAME   = "lawKB"
COL_DOCS  = "law_kb_docs"
COL_CHUNKS= "law_kb_chunks"
COL_LINKS = "law_kb_links"

# 允许作为被连到的目标文种（整部法律/整份解释）
TARGET_DOC_TYPES = {"statute", "judicial_interpretation"}

# 《……》第……条 解析（回退用）
PAT = re.compile(r"《\s*([^》]+?)\s*》第([〇零一二三四五六七八九十百千两\d]+)条")

CN_PREFIX = re.compile(r"^\s*中华人民共和国")
# 去掉“（…修正…）/（…修订…）”这类括注；随后再统一去各种括号符号
RM_FIX = re.compile(r"（[^）]*修[正订][^）]*）|\([^)]*修[正订][^)]*\)")
BRACKETS = re.compile(r"[()（）\[\]【】〈〉<>《》]")     # 仅去括号“符号”，不吃里面的字
MULTI_WS = re.compile(r"\s+")

CN_MAP = {"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,
          "七":7,"八":8,"九":9,"十":10,"百":100,"千":1000,"万":10000}
def cn_to_int(s: str) -> int:
    s = str(s or "").strip().replace("第","").replace("条","")
    if s.isdigit():
        return int(s)
    total, tmp = 0, 0
    for ch in s:
        if ch in "十百千万":
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
    # 统一去掉成对书名号/角括号
    s = s.replace("《","").replace("》","").replace("〈","").replace("〉","").replace("<","").replace(">","")
    s = CN_PREFIX.sub("", s)
    s = RM_FIX.sub("", s)                         # 先去“修正/修订”类括注内容
    s = BRACKETS.sub("", s)                       # 再去残余括号符号（里面的字会保留）
    s = s.replace("　"," ").replace("·"," ")
    s = MULTI_WS.sub("", s)                       # 去所有空白
    return s

LAW_L  = r"[《〈<⟪⟨]"
LAW_R  = r"[》〉>⟫⟩]"
BRK_OPT = r"(?:\s*(?:（[^）]*）|\([^)]*\)|\[[^\]]*\]|【[^】]*】))*"
PAT = re.compile(
    rf"{LAW_L}\s*([^》〉>⟫⟩]+?)\s*{LAW_R}{BRK_OPT}\s*第([〇零一二三四五六七八九十百千万两\d]+)条"
)

def ensure_indexes(E):
    E.create_index([("from_doc",1)])
    E.create_index([("to_doc",1)])
    E.create_index([("edge",1)])
    # 若你希望强约束去重，可在数据清洗后改为 unique 复合索引：
    # E.create_index([( "from_doc",1),("to_doc",1),("edge",1)], unique=True)

def load_target_docs(D, C) -> Dict[str, List[Tuple[str,str]]]:
    name2docs: Dict[str, List[Tuple[str,str]]] = {}

    # 1) 先用 docs 的预计算规范名（最稳）
    for d in D.find({"doc_type": {"$in": list(TARGET_DOC_TYPES)}},
                    {"doc_id":1,"doc_type":1,"law_name":1,"law_alias":1,
                     "law_name_norm":1,"law_alias_norm":1}):
        doc_id, dt = d["doc_id"], d["doc_type"]
        keys = set()
        if d.get("law_name_norm"):
            keys.add(d["law_name_norm"])
        for k in (d.get("law_alias_norm") or []):
            if k: keys.add(k)
        # 兜底：再把原始名也跑一遍本脚本的 norm
        for raw in [d.get("law_name",""), * (d.get("law_alias") or [])]:
            k = norm_name(raw)
            if k: keys.add(k)
        for k in keys:
            name2docs.setdefault(k, []).append((doc_id, dt))

    # 2) 用 chunks 再补充一圈别名（兼容旧数据）
    cur = C.find({"doc_type":{"$in": list(TARGET_DOC_TYPES)}},
                 {"doc_id":1,"law_name":1,"law_alias":1})
    for d in cur:
        doc_id = d.get("doc_id")
        for raw in [d.get("law_name",""), * (d.get("law_alias") or [])]:
            k = norm_name(raw)
            if k:
                name2docs.setdefault(k, []).append((doc_id, ""))

    return name2docs


def _normalize_articles(raw: Any) -> List[int]:
    """将 article 字段的多形态值统一为正整数列表。"""

    numbers: Set[int] = set()

    def collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for v in value:
                collect(v)
            return
        if isinstance(value, dict):
            # 支持 range 结构，如 {"L": 1, "R": 3}
            left = value.get("L") or value.get("left") or value.get("start") or value.get("from")
            right = value.get("R") or value.get("right") or value.get("end") or value.get("to")
            if left is not None or right is not None:
                l = cn_to_int(left)
                r = cn_to_int(right if right is not None else left)
                if l > 0 and r > 0:
                    if r < l:
                        l, r = r, l
                    for n in range(l, r + 1):
                        numbers.add(n)
                    return
                if l > 0:
                    numbers.add(l)
                if right is not None:
                    rn = cn_to_int(right)
                    if rn > 0:
                        numbers.add(rn)
                return
            for v in value.values():
                collect(v)
            return
        n = cn_to_int(value)
        if n > 0:
            numbers.add(n)

    collect(raw)
    return sorted(numbers)


def parse_statutes_from_doc(jdoc: dict) -> Tuple[List[Tuple[str, str, Optional[int]]], List[str]]:
    """
    返回：
      - parsed: [(law_name_norm, law_name_raw, article_no or None)]
      - failures: 无法解析的原始片段（字符串化）
    """

    parsed: List[Tuple[str, str, Optional[int]]] = []
    failures: List[str] = []

    def emit(law_raw: str, article: Optional[int]) -> bool:
        law_norm = norm_name(law_raw)
        if not law_norm:
            return False
        if article is not None and article <= 0:
            return False
        parsed.append((law_norm, law_raw, article))
        return True

    def handle_item(item: Any) -> None:
        if isinstance(item, dict):
            law_raw = item.get("law") or item.get("name") or item.get("title") or ""
            law_raw = fix_interpretation_title(law_raw)
            arts_raw = item.get("article")
            if arts_raw in (None, "", []):
                arts_raw = item.get("articles")
            if arts_raw in (None, "", []):
                arts_raw = item.get("article_no")

            articles = _normalize_articles(arts_raw)
            emitted = False
            for art in articles:
                if emit(law_raw, art):
                    emitted = True
            if not articles and law_raw:
                emitted = emit(law_raw, None) or emitted
            if not emitted and (law_raw or arts_raw):
                failures.append(str(item))
        elif isinstance(item, str):
            text = fix_interpretation_title(item)
            emitted = False
            for m in PAT.finditer(text):
                law_raw, art_raw = m.group(1), m.group(2)
                art = cn_to_int(art_raw)
                if emit(law_raw, art):
                    emitted = True
            if not emitted and text.strip():
                failures.append(text)
        else:
            if item not in (None, "", []):
                failures.append(str(item))

    structured = (((jdoc.get("judgment_info") or {}).get("statutes")) or [])
    for entry in structured:
        handle_item(entry)

    for entry in jdoc.get("statutes") or []:
        handle_item(entry)

    return parsed, failures

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
    parse_failures: Dict[str, int] = {}

    def record_failures(items: Iterable[str]) -> None:
        for it in items:
            if not it:
                continue
            parse_failures[it] = parse_failures.get(it, 0) + 1

    for j in cur:
        from_doc = j["doc_id"]
        # 聚合同一法律的条文
        law_to_articles: Dict[str, set] = {}
        law_to_rawname: Dict[str, str] = {}
        law_no_articles: Set[str] = set()

        parsed_statutes, failed_items = parse_statutes_from_doc(j)
        record_failures(failed_items)

        for lname_norm, law_raw, art in parsed_statutes:
            if not lname_norm:
                continue
            if lname_norm not in law_to_articles:
                law_to_articles[lname_norm] = set()
                law_to_rawname[lname_norm] = law_raw
            if art is None:
                law_no_articles.add(lname_norm)
                continue
            if art > 0:
                law_to_articles[lname_norm].add(int(art))

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
                update_doc = {"$setOnInsert": {"law_name_raw": law_to_rawname[lname_norm]}}
                sorted_arts = sorted(arts)
                if sorted_arts:
                    update_doc["$addToSet"] = {"articles": {"$each": sorted_arts}}
                elif lname_norm in law_no_articles:
                    update_doc["$setOnInsert"]["articles"] = []
                else:
                    continue
                ops.append(
                    UpdateOne(
                        {"from_doc": from_doc, "to_doc": to_doc, "edge": "applies"},
                        update_doc,
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

    if parse_failures:
        top = sorted(parse_failures.items(), key=lambda x: -x[1])[:10]
        print("[PARSE FAILURES] top 10 raw snippets:")
        for k, v in top:
            print(f"  - {k} ×{v}")

if __name__ == "__main__":
    main()