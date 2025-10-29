# link_builder_judgment.py
# 作用：从 MongoDB 的判决文书 chunk（doc_type="judgment"）里抽取所“适用”的法条/司法解释，
#      在 law_kb_links 中写入 {from_chunk, to_chunk, edge="applies"}。
#
# 约定：
# - 目标可连到 doc_type ∈ {"statute", "judicial_interpretation"} 的 chunk；
# - 命中文本来自各段落（优先“法律依据/依据/法院认为/裁判理由/判决结果”等），但默认对所有 section 扫描；
# - 仅按“条”粒度建边（款/项被记录在匹配信息里，不参与定位），因为库里 article_no 通常是 “第X条(~第Y条)”。
#
# 运行示例：
#   python link_builder_judgment.py \
#     --mongo_uri "mongodb://localhost:27017" --mongo_db lawkb \
#     --chunk_col law_kb_chunks --link_col law_kb_links \
#     --targets statute judicial_interpretation

import re
import argparse
from typing import Dict, List, Tuple, Set
from pymongo import MongoClient, InsertOne

# ---------------- 基础工具 ---------------- #

CN_MAP = {"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"百":100,"千":1000}
def cn_to_int(s: str) -> int:
    """
    中文数字转阿拉伯：支持 “两百三十五”“一千零二十”“十七”“第二百三十三条之一(→233)”
    条文里的“之一/之二”按主条处理（回落到主条号）以便与库中“第X条(~第Y条)”匹配。
    """
    if not s: return 0
    s = s.strip()
    s = s.replace("之二","").replace("之一","").replace("之三","").replace("之四","")  # 统一回落到主条
    s = s.replace("第","").replace("条","").replace("款","").replace("项","")
    if s.isdigit(): return int(s)

    total, unit_val, num = 0, 1, 0
    # 自左向右处理：遇到单位（十百千）就把当前 num 累乘单位；支持“十七”= 10+7、“两百零三”= 2*100+3
    last_unit = 1
    for ch in s:
        v = CN_MAP.get(ch, None)
        if v is None:
            # 跳过非数字单位字符（如空格）
            continue
        if v in (10,100,1000):
            num = 1 if num == 0 else num
            total += num * v
            num = 0
            last_unit = v
        else:
            num = num * 10 + v if last_unit == 1 else num + v
    return total + num

def norm_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("《","").replace("》","")
    s = re.sub(r"^中华人民共和国", "", s)  # 统一去前缀
    s = re.sub(r"\s+", "", s)
    return s

# 常见简称 → 规范名（用于无书名号场景）
ALIAS_CANON = {
    "民法典": "民法典",
    "刑法": "刑法",
    "刑诉法": "刑事诉讼法",
    "民诉法": "民事诉讼法",
    "行诉法": "行政诉讼法",
    "治安管理处罚法": "治安管理处罚法",
    "公司法": "公司法",
    "合同法": "合同法",
    "劳动法": "劳动法",
    "劳动合同法": "劳动合同法",
    "消费者权益保护法": "消费者权益保护法",
    "证券法": "证券法",
    "道路交通安全法": "道路交通安全法",
}

# ---------------- Mongo 访问 ---------------- #

def ensure_indexes(link_col):
    link_col.create_index([("from_chunk",1)])
    link_col.create_index([("to_chunk",1)])
    link_col.create_index([("edge",1)])

def load_targets(chunks, target_doc_types: List[str]):
    """
    预加载可作为被引目标的 chunk：
    - 记录规范化法名 _name_norm
    - 参与 alias 匹配的 _aliases
    - 将 article_no 转为 [L,R] 便于条号落点判断
    """
    fields = {"chunk_id":1,"doc_type":1,"law_name":1,"law_alias":1,"article_no":1}
    buckets: Dict[str, List[dict]] = {dt: [] for dt in target_doc_types}
    cur = chunks.find({"doc_type":{"$in": target_doc_types}}, fields)
    for d in cur:
        d["_name_norm"] = norm_name(d.get("law_name",""))
        d["_aliases"]   = [norm_name(a) for a in (d.get("law_alias") or [])]
        L, R = article_span_to_range(d.get("article_no",""))
        d["_L"], d["_R"] = L, R
        buckets[d["doc_type"]].append(d)
    return buckets

def article_span_to_range(span: str) -> Tuple[int,int]:
    """
    "第二十一条~第二十六条" → (21,26)
    "第十条" → (10,10)
    其它留空：→(0,0)
    """
    if not span:
        return (0,0)
    parts = span.split("~", 1)
    L = cn_to_int(parts[0])
    R = cn_to_int(parts[1]) if len(parts) > 1 else L
    return (L, R)

def name_ok(query_norm: str, d: dict) -> bool:
    # 等值 / 以简称结尾 / 在别名里
    if query_norm == d["_name_norm"]: return True
    if d["_name_norm"].endswith(query_norm): return True
    if query_norm in d["_aliases"]: return True
    return False

# ---------------- 抽取规则（正则） ---------------- #

# 1) 书名号法名 + 条号（同一法律下多条/区间需一并识别）
RE_LAW_BRACKET = re.compile(r"《\s*([^》]+?)\s*》")
RE_RANGE = re.compile(r"第([〇零一二三四五六七八九十百千万两\d]+)条\s*(?:至|到|~|-)\s*第([〇零一二三四五六七八九十百千万两\d]+)条")
RE_SINGLE = re.compile(r"第([〇零一二三四五六七八九十百千万两\d]+)条")

# 2) 无书名号简写：如“刑法第二百三十三条”“民法典第一千零七十九条”
RE_NAKED = re.compile(
    r"(?:(?:中华人民共和国)?"
    r"(民法典|刑法|刑事诉讼法|民事诉讼法|行政诉讼法|治安管理处罚法|公司法|证券法|合同法|劳动合同法|劳动法|消费者权益保护法|道路交通安全法))"
    r"第([〇零一二三四五六七八九十百千万两\d]+)条"
)

# ---------------- 抽取主流程 ---------------- #

def extract_pairs(text: str) -> List[Tuple[str, int, str]]:
    """
    从文本提取 (law_raw, article_int, matched_text)
    - 先找带书名号的法名，随后在该法名后的“局部窗口”里收集条号（直到下一个书名号或句号）
    - 其次解析无书名号的“简称+第X条”
    """
    pairs: List[Tuple[str,int,str]] = []

    # A) 书名号法名
    for m in RE_LAW_BRACKET.finditer(text):
        law_raw = m.group(1)
        # 窗口：到下一个《 或句号
        start = m.end()
        next_law = text.find("《", start)
        stop = next_law if next_law != -1 else len(text)
        window = text[start:stop]

        # 先识别区间
        for r in RE_RANGE.finditer(window):
            a1 = cn_to_int(r.group(1)); a2 = cn_to_int(r.group(2))
            if a1 and a2 and a2 >= a1 and a2 - a1 <= 1000:
                for a in range(a1, a2+1):
                    pairs.append((law_raw, a, "《%s》第%s~第%s条" % (law_raw, r.group(1), r.group(2))))
        # 再识别单条（包括“、”并列，本质是多个单条）
        for s in RE_SINGLE.finditer(window):
            a = cn_to_int(s.group(1))
            if a:
                pairs.append((law_raw, a, "《%s》第%s条" % (law_raw, s.group(1))))

    # B) 无书名号简写
    for n in RE_NAKED.finditer(text):
        short = n.group(1)
        a = cn_to_int(n.group(2))
        if a:
            canon = ALIAS_CANON.get(short, short)
            pairs.append((canon, a, "%s第%s条" % (short, n.group(2))))

    return pairs

def match_targets(buckets, law_raw: str, a: int) -> List[str]:
    """返回所有命中的目标 chunk_id 列表（在 L~R 内）"""
    lname = norm_name(law_raw)
    hits: List[str] = []
    for dt in buckets.keys():
        for d in buckets[dt]:
            if not name_ok(lname, d): 
                continue
            L, R = d["_L"], d["_R"]
            if L and R and L <= a <= R:
                hits.append(d["chunk_id"])
    return hits

# ---------------- 入口 ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    ap.add_argument("--mongo_db",  default="lawkb")
    ap.add_argument("--chunk_col", default="law_kb_chunks")
    ap.add_argument("--link_col",  default="law_kb_links")
    ap.add_argument("--targets", nargs="+", default=["statute","judicial_interpretation"])
    ap.add_argument("--only_sections", nargs="*", default=[], 
                    help="可选：仅处理这些 section（如：依据 法律依据 法院认为 裁判理由 判决结果）")
    args = ap.parse_args()

    mc = MongoClient(args.mongo_uri)
    C = mc[args.mongo_db][args.chunk_col]
    E = mc[args.mongo_db][args.link_col]

    ensure_indexes(E)
    buckets = load_targets(C, args.targets)

    # 仅处理判决文书
    query = {"doc_type":"judgment"}
    if args.only_sections:
        query["section"] = {"$in": args.only_sections}

    cur = C.find(query, {"chunk_id":1,"text":1})
    total_pairs = 0
    links = 0
    seen: Set[Tuple[str,str]] = set()
    miss_name: Dict[str,int] = {}

    ops: List[InsertOne] = []
    docs = list(cur)
    print(f"[INFO] judgment chunks: {len(docs)}")

    for d in docs:
        cid = d["chunk_id"]
        text = d.get("text") or ""
        pairs = extract_pairs(text)
        total_pairs += len(pairs)
        for law_raw, art, mtxt in pairs:
            to_ids = match_targets(buckets, law_raw, art)
            if to_ids:
                for to in to_ids:
                    key = (cid, to)
                    if key in seen: 
                        continue
                    seen.add(key)
                    ops.append(InsertOne({"from_chunk": cid, "to_chunk": to, "edge": "applies"}))
                    links += 1
            else:
                k = norm_name(law_raw)
                miss_name[k] = miss_name.get(k, 0) + 1

    if ops:
        E.bulk_write(ops, ordered=False)

    top_miss = sorted(miss_name.items(), key=lambda x: -x[1])[:10]
    print(f"[SUMMARY] pairs_found={total_pairs}, links_created={links}, unique_pairs={len(seen)}, miss_law_names={len(miss_name)}")
    if top_miss:
        print("[MISSING LAW NAMES]（前10，建议补充入库或加入 law_alias）")
        for name, cnt in top_miss:
            print(f"  - {name} ×{cnt}")

if __name__ == "__main__":
    main()
