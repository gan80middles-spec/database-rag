# link_builder_final.py
import re
from pymongo import MongoClient, InsertOne

MC = MongoClient("mongodb://adminUser:~Q2w3e4r@192.168.110.36:27019")
DB = MC["lawKB"]
C  = DB["law_kb_chunks"]
E  = DB["law_kb_links"]

# —— 配置：解释引用可连到哪些文种 —— #
TARGET_DOC_TYPES = ["statute", "judicial_interpretation"]  # 只想连法律就改成 ["statute"]

# 《……》第……条  （注意：有些标题内部还会出现 〈…〉，对匹配不影响）
PAT = re.compile(r"《\s*([^》]+?)\s*》第([〇零一二三四五六七八九十百千两\d]+)条")

CN_MAP = {"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"百":100,"千":1000}
def cn_to_int(s: str) -> int:
    s = str(s).strip().replace("第","").replace("条","")
    if s.isdigit(): return int(s)
    # 简易中文数字到千位
    total, last = 0, 0
    for ch in s:
        if ch in "十百千":
            unit = CN_MAP[ch]
            total = (total or 1) * unit
        else:
            n = CN_MAP.get(ch, 0)
            total += n
            last = n
    return total or last or 0

def norm_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("《","").replace("》","")
    s = re.sub(r"^中华人民共和国", "", s)  # 去前缀
    s = re.sub(r"\s+", "", s)
    return s

def article_span_to_int(span: str):
    """'第九十条' 或 '第九十条~第一百条' → (L,R)"""
    if not span: return (None,None)
    parts = span.split("~", 1)
    L = cn_to_int(parts[0])
    R = cn_to_int(parts[1]) if len(parts) > 1 else L
    return (L, R)

def load_targets():
    """预加载候选目标：按 doc_type 聚合，便于快速过滤"""
    fields = {"chunk_id":1,"doc_type":1,"law_name":1,"law_alias":1,"article_no":1}
    cur = C.find({"doc_type":{"$in": TARGET_DOC_TYPES}}, fields)
    buckets = {dt: [] for dt in TARGET_DOC_TYPES}
    for d in cur:
        d["_name_norm"] = norm_name(d.get("law_name",""))
        d["_aliases"]   = [norm_name(a) for a in (d.get("law_alias") or [])]
        d["_L"], d["_R"] = article_span_to_int(d.get("article_no",""))
        buckets[d["doc_type"]].append(d)
    return buckets

def match_targets(buckets, law_raw: str, art_raw: str):
    """返回所有命中的目标 chunk_id 列表"""
    lname = norm_name(law_raw)
    anum  = cn_to_int(art_raw)
    hits = []

    def name_ok(d):
        # 等值 / 以简称结尾 / 在别名里
        if lname == d["_name_norm"]: return True
        if d["_name_norm"].endswith(lname): return True
        if lname in d["_aliases"]: return True
        return False

    for dt in TARGET_DOC_TYPES:
        for d in buckets[dt]:
            if not name_ok(d): continue
            L, R = d["_L"], d["_R"]
            if L is None: continue
            if L <= anum <= R:
                hits.append(d["chunk_id"])
    return hits

def ensure_indexes():
    E.create_index([("from_chunk",1)])
    E.create_index([("to_chunk",1)])
    E.create_index([("edge",1)])

if __name__ == "__main__":
    interps = list(C.find({"doc_type":"judicial_interpretation"}, {"chunk_id":1,"text":1}))
    print(f"[INFO] judicial_interpretation docs: {len(interps)}")
    buckets = load_targets()

    total_cites = 0
    links = 0
    miss_name = {}
    ops = []
    seen = set()  # 去重： (from,to,edge)

    for d in interps:
        text = d.get("text") or ""
        cites = PAT.findall(text)
        if not cites: continue
        for law_raw, art_raw in cites:
            total_cites += 1
            to_ids = match_targets(buckets, law_raw, art_raw)
            if to_ids:
                for to in to_ids:
                    key = (d["chunk_id"], to, "cites")
                    if key in seen: continue
                    seen.add(key)
                    ops.append(InsertOne({"from_chunk": d["chunk_id"], "to_chunk": to, "edge": "cites"}))
                    links += 1
            else:
                k = norm_name(law_raw)
                miss_name[k] = miss_name.get(k, 0) + 1

    if ops:
        # 批量插入
        res = E.bulk_write(ops, ordered=False)
        ensure_indexes()

    # 汇总缺失法名 Top-10
    top_miss = sorted(miss_name.items(), key=lambda x: -x[1])[:10]
    print(f"[SUMMARY] cites_found={total_cites}, links_created={links}, unique_edges={len(seen)}, miss_law_kinds={len(miss_name)}")
    if top_miss:
        print("[MISSING LAW NAMES]（前10，优先考虑入库或加入 law_alias）")
        for name, cnt in top_miss:
            print(f"  - {name} ×{cnt}")
