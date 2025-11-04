# -*- coding: utf-8 -*-
"""
link_builder_judgment.py
- 从 law_kb_docs 读取“判决/裁定/调解”等文书
- 解析文书中出现的法规/司法解释引用（含条/款/项）
- 以 law_kb_docs + law_kb_chunks 双索引定位目标文档 doc_id
- 建立 law_kb_links:  from_doc=判决书doc_id → to_doc=法规/解释doc_id, edge="applies", articles=[...]
- 若定位成功，顺带把“裁判文书里的原始全称”追加为该 doc_id 全部 chunk 的 law_alias
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict, Counter

from pymongo import MongoClient, UpdateOne

# === 连接配置 ===
MONGO_URI = "mongodb://adminUser:~Q2w3e4r@192.168.110.36:27019"
DB_NAME   = "lawKB"
COL_DOCS  = "law_kb_docs"
COL_CHUNKS= "law_kb_chunks"
COL_LINKS = "law_kb_links"

# 目标“被连接”的文种
TARGET_DOC_TYPES = {
    "statute",
    "judicial_interpretation",
    "administrative_regulation",  # 可按需增删
    "local_regulation",
}

# ---------- 基础正则与工具 ----------

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

CN_PREFIX = re.compile(r"^\s*中华人民共和国")
BRACKETS  = re.compile(r"[()（）\[\]【】＜＞<>]")  # 仅清符号
MULTI_WS  = re.compile(r"\s+")

def norm_name(s: str) -> str:
    if not s: return ""
    s = s.strip().replace("《","").replace("》","")
    s = CN_PREFIX.sub("", s)
    # 去“（××修正）”一类括注
    s = re.sub(r"（[^）]*修正[^）]*）", "", s)
    s = re.sub(r"\([^)]*修正[^)]*\)", "", s)
    s = BRACKETS.sub("", s)
    s = s.replace("　"," ").replace("·"," ")
    s = MULTI_WS.sub("", s)
    return s

# 去掉内层书名号〈…〉 / <…>，保留里面文字
INNER_QUOTES_ANY = re.compile(r"[〈<]\s*([^〉>]+?)\s*[〉>]")
def strip_inner_quotes(s: str) -> str:
    return INNER_QUOTES_ANY.sub(r"\1", s or "")

# “关于适用《……法》的解释” → “关于适用……法的解释”
INNER_LAW = re.compile(r"关于适用\s*[《〈<]\s*([^》〉>]+?)\s*[》〉>]\s*的解释")
def fix_interpretation_title(raw: str) -> str:
    s = str(raw or "").strip()
    m = INNER_LAW.search(s)
    if m:
        inner = m.group(1)
        inner = inner.replace("《","").replace("》","").replace("〈","").replace("〉","").replace("<","").replace(">","").strip()
        s = INNER_LAW.sub(f"关于适用{inner}的解释", s)
    return s

# 黑名单：不建边、不计失败
BLACKLIST_TITLE_REGEX = re.compile(r"(请示的复函|请示的答复|工作会议纪要)")
def is_blacklisted_title(s: str) -> bool:
    return bool(BLACKLIST_TITLE_REGEX.search(str(s or "")))

# 《……》第……条（可带括注），允许“第×至第×条”
LAW_L   = r"[《〈<⟪⟨]"
LAW_R   = r"[》〉>⟫⟩]"
BRK_OPT = r"(?:\s*(?:（[^）]*）|\([^)]*\)|\[[^\]]*\]|【[^】]*】))*"

PAT_RANGE = re.compile(
    rf"{LAW_L}\s*(.+?)\s*{LAW_R}{BRK_OPT}\s*第\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条\s*[-~－—–]\s*第\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条"
)
PAT_MAIN = re.compile(
    rf"{LAW_L}\s*(.+?)\s*{LAW_R}{BRK_OPT}\s*第\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条"
    r"(?:\s*第\s*[（(]?[〇零一二三四五六七八九十百千万两\d]+[）)]?\s*款"
    r"(?:\s*第\s*[（(]?[〇零一二三四五六七八九十百千万两\d]+[）)]?\s*项)?)?"
)

GENERIC_INTERP = {"最高人民法院关于适用的解释", "关于适用的解释"}

NAME_BLACKLIST_NORMS = {
    norm_name("上海市政府信息公开规定"),
    norm_name("最高人民法院关于适用中华人民共和国合同法若干问题的解释（二）"),
    norm_name("****法"),
    norm_name("上海市国有土地上房屋征收与补偿实施细则"),
    norm_name("北京市住宅专项维修资金管理办法"),
    norm_name("协调会纪要"),
    norm_name("浙江省司法鉴定管理条例"),
    norm_name("第八次全国法院民事商事审判工作会议民事部分纪要"),
    norm_name("西安市城市集中供热管理条例"),
    norm_name("西安市集中供热条例"),
    norm_name("公安机关内部执法监督工作规定"),
    norm_name("公安机关办理刑事案件程序规定"),
    norm_name("公安机关办理行政案件程序规定"),
    norm_name("公安机关鉴定规则"),
    norm_name("工商行政管理机关行政处罚程序暂行规定"),
    norm_name("工商行政管理机关行政赔偿实施办法"),
    norm_name("市场监督管理行政处罚听证办法"),
    norm_name("市场监督管理行政处罚程序规定"),
    norm_name("住房和城乡建设部房屋建筑和市政基础设施工程竣工验收备案管理办法"),
    norm_name("因工死亡职工供养亲属范围规定"),
    norm_name("关于收取城市基础设施配套费有关问题的规定"),
    norm_name("道路交通事故处理程序规定"),
    norm_name("矿业权交易规则"),
    norm_name("监督规则"),
    norm_name("人力资源社会保障部最高人民法院关于劳动人事争议仲裁与诉讼衔接有关问题的意见一"),
    norm_name("劳动争议处理条例"),
    norm_name("劳动合同法实施条例"),
    norm_name("国务院关于职工工作时间的规定"),
    norm_name("国有企业富余职工安置规定"),
    norm_name("国有土地上房屋征收评估办法"),
    norm_name("城市房地产开发经营管理条例"),
    norm_name("失业保险条例"),
    norm_name("工亡保险条例"),
    norm_name("建设工程质量保证金管理办法"),
    norm_name("建设工程质量管理办法"),
    norm_name("必须招标的工程项目规定"),
    norm_name("暂行规定"),
    norm_name("机动车交通事故责任强制保险条例"),
    norm_name("生产安全事故报告和调查处理条例"),
    norm_name("矿产资源开采登记管理办法"),
    norm_name("行政区域界线管理条例"),
    norm_name("行政区域界限管理条例"),
    norm_name("证券期货违法行为行政处罚办法"),
    norm_name("道路交通安全法实施条例"),
    norm_name("最高人民法院关于****监督程序若干问题的规定"),
    norm_name("最高人民法院关于人民法院执行工作若干问题的规定试行"),
    norm_name("最高人民法院关于人民法院赔偿委员会审理****案件程序的规定"),
    norm_name("最高人民法院关于审理劳动争议案件适用法律若干问题的解释"),
    norm_name("最高人民法院关于审理涉及夫妻债务纠纷案件适用法律有关问题的解释"),
    norm_name("最高人民法院关于审理道路事故损害赔偿案件适用若干问题的解释"),
    norm_name("最高人民法院关于适用中华人民共和国婚姻法若干问题的解释二"),
    norm_name("最高人民法院关于适用中华人民共和国民法总则诉讼时效制度若干问题的解释"),
    norm_name("最高人民法院关于适用中华人民共和国行政诉讼法若干问题的解释"),
    norm_name("最高人民法院关于溯及力和人民法院赔偿委员会受案范围问题的批复"),
    norm_name("最高人民法院关于审理拒不执行判决、裁定案件具体应用法律若干问题的解释"),
    norm_name("最高人民法院关于适用时间效力的若干规定")
}

# ---------- 解析“民/行/刑”系统、条号展开 ----------
def suggest_system(jdoc: dict) -> str:
    info = (jdoc.get("judgment_info") or {})
    txts = []
    for k in ("case_system", "case_type", "cause", "trial_procedure", "title"):
        v = info.get(k) or jdoc.get(k)
        if isinstance(v, str):
            txts.append(v)
    s = "；".join(txts)
    if "行政" in s: return "admin"
    if "刑事" in s: return "criminal"
    return "civil"

_re_rng = re.compile(r"第(.+?)条\s*[-~－—–]\s*第(.+?)条")
_re_one = re.compile(r"第(.+?)条")
def expand_article_no(no: str) -> List[int]:
    if not no: return []
    m = _re_rng.match(no)
    if m:
        L,R = cn_to_int(m.group(1)), cn_to_int(m.group(2))
        if L and R and R>=L:
            return list(range(L, R+1))
    m = _re_one.match(no)
    n = cn_to_int(m.group(1)) if m else 0
    return [n] if n>0 else []

# ---------- 预索引 ----------
def build_docs_index(D) -> Dict[str, Set[str]]:
    """
    law_kb_docs: 规范名/别名 → {doc_id}
    """
    name2docs: Dict[str, Set[str]] = defaultdict(set)
    cur = D.find({"doc_type": {"$in": list(TARGET_DOC_TYPES)}},
                 {"doc_id":1,"law_name":1,"law_alias":1})
    for d in cur:
        did = d["doc_id"]
        nm = norm_name(d.get("law_name",""))
        if nm: name2docs[nm].add(did)
        for al in d.get("law_alias") or []:
            alnm = norm_name(al)
            if alnm: name2docs[alnm].add(did)
    return name2docs

def build_chunks_index(C):
    from collections import defaultdict
    title2docs = defaultdict(set)
    name2docs  = defaultdict(set)
    doc2arts   = defaultdict(set)
    doc2vdate  = {}

    cur = C.find(
        {"doc_type": {"$in": list(TARGET_DOC_TYPES)}},
        {"doc_id":1,"title":1,"law_name":1,"article_no":1,"version_date":1},
    )
    for d in cur:
        did = d.get("doc_id")
        if not did: 
            continue

        t = fix_interpretation_title(strip_inner_quotes(d.get("title") or ""))
        tn = norm_name(t)
        if tn: 
            title2docs[tn].add(did)

        ln = norm_name(d.get("law_name") or "")
        if ln: 
            name2docs[ln].add(did)

        for n in expand_article_no(d.get("article_no")):
            doc2arts[did].add(n)

        v = d.get("version_date") or ""
        if v and (did not in doc2vdate or v > doc2vdate[did]):
            doc2vdate[did] = v

    return title2docs, name2docs, doc2arts, doc2vdate


def find_susufa_docs(C) -> Dict[str, str]:
    """
    直接用 chunks.title 正则找三部“关于适用…诉讼法的解释”的 doc_id
    """
    patt = {
        "民事诉讼法解释": r"适用.*民事诉讼法.*的解释",
        "行政诉讼法解释": r"适用.*行政诉讼法.*的解释",
        "刑事诉讼法解释": r"适用.*刑事诉讼法.*的解释",
    }
    out: Dict[str, str] = {}
    for short, rx in patt.items():
        hit = C.find_one(
            {"doc_type": "judicial_interpretation", "title": {"$regex": rx}},
            {"doc_id": 1},
        )
        if not hit:
            hit = C.find_one(
                {"doc_type": "judicial_interpretation", "law_name": short},
                {"doc_id": 1},
            )
        if hit and hit.get("doc_id"):
            out[short] = hit["doc_id"]
    return out

# 民法典条号集合（从 chunks 的目录推断）
def collect_civil_contract_articles(C, civil_doc_id: str) -> Set[int]:
    """民法典·第三编 合同 -> 条号集合"""
    arts: Set[int] = set()
    for d in C.find({"doc_id": civil_doc_id}, {"path":1,"article_no":1}):
        p = d.get("path") or {}
        key = " ".join(str(x or "") for x in [p.get("编"), p.get("分编"), p.get("章"), p.get("节")])
        if ("第三编" in key and "合同" in key) or "合同编" in key:
            for n in expand_article_no(d.get("article_no")):
                arts.add(n)
    return arts


def collect_civil_property_articles(C, civil_doc_id: str) -> Set[int]:
    """民法典·第二编 物权 -> 条号集合"""
    arts: Set[int] = set()
    for d in C.find({"doc_id": civil_doc_id}, {"path":1,"article_no":1}):
        p = d.get("path") or {}
        key = " ".join(str(x or "") for x in [p.get("编"), p.get("分编"), p.get("章"), p.get("节")])
        if ("第二编" in key and "物权" in key) or "物权编" in key:
            for n in expand_article_no(d.get("article_no")):
                arts.add(n)
    return arts


# 民法典“担保物权 + 保证合同”的条号集合（从 chunks 的目录推断）
def collect_civil_guarantee_articles(C, civil_doc_id: str) -> Set[int]:
    arts: Set[int] = set()
    q = {"doc_id": civil_doc_id}
    for d in C.find(q, {"path":1,"article_no":1}):
        p = d.get("path") or {}
        bian = (p.get("编") or p.get("篇") or "")  # 兼容不同字段名
        zhang = p.get("章") or ""
        fenbian = p.get("分编") or ""
        jiemu = p.get("节") or ""
        # 物权编·第四分编 担保物权
        key = f"{bian} {fenbian} {zhang} {jiemu}"
        if ("第二编" in key and "物权" in key and ("担保物权" in key or "第十六章" in key or "第十七章" in key or "第十八章" in key or "第十九章" in key)) \
           or ("担保物权" in key):
            for n in expand_article_no(d.get("article_no")):
                arts.add(n)
        # 合同编·第十三章 保证合同
        if ("第三编" in key and "合同" in key and ("第十三章" in key or "保证合同" in key)) \
           or ("保证合同" in key):
            for n in expand_article_no(d.get("article_no")):
                arts.add(n)
    return arts

# ---------- 解析 statutes ----------
def _normalize_articles(raw: Any) -> List[int]:
    numbers: Set[int] = set()
    def collect(value: Any) -> None:
        if value is None: return
        if isinstance(value, (list, tuple, set)):
            for v in value: collect(v); return
        if isinstance(value, dict):
            left = value.get("L") or value.get("left") or value.get("start") or value.get("from")
            right= value.get("R") or value.get("right") or value.get("end") or value.get("to")
            if left is not None or right is not None:
                l = cn_to_int(left)
                r = cn_to_int(right if right is not None else left)
                if l>0 and r>0:
                    if r<l: l,r = r,l
                    for n in range(l, r+1): numbers.add(n)
                    return
                if l>0: numbers.add(l)
                if right is not None:
                    rn = cn_to_int(right)
                    if rn>0: numbers.add(rn)
                return
            for v in value.values(): collect(v)
            return
        n = cn_to_int(value)
        if n>0: numbers.add(n)
    collect(raw)
    return sorted(numbers)

def parse_statutes_from_doc(jdoc: dict) -> Tuple[List[Tuple[str, str, Optional[int]]], List[str]]:
    parsed: List[Tuple[str,str,Optional[int]]] = []
    failures: List[str] = []

    def emit(law_raw: str, article: Optional[int]) -> bool:
        lw = law_raw.strip()
        if not lw: return False
        law_norm = norm_name(lw)
        if not law_norm: return False
        if article is not None and article <= 0: return False
        parsed.append((law_norm, lw, article))
        return True

    def handle_item(item: Any) -> None:
        if isinstance(item, dict):
            law_raw = item.get("law") or item.get("name") or item.get("title") or ""
            law_raw = fix_interpretation_title(strip_inner_quotes(law_raw))
            if is_blacklisted_title(law_raw):
                return
            arts_raw = item.get("article")
            if arts_raw in (None,"",[]): arts_raw = item.get("articles")
            if arts_raw in (None,"",[]): arts_raw = item.get("article_no")
            articles = _normalize_articles(arts_raw)
            ok = False
            for a in articles:
                ok = emit(law_raw, a) or ok
            if not articles and law_raw:
                ok = emit(law_raw, None) or ok
            if not ok and (law_raw or arts_raw):
                failures.append(str(item))
        elif isinstance(item, str):
            text = fix_interpretation_title(strip_inner_quotes(item))
            if is_blacklisted_title(text):
                return
            ok = False
            for m in PAT_RANGE.finditer(text):
                law_raw, l, r = m.group(1), m.group(2), m.group(3)
                for a in range(cn_to_int(l), cn_to_int(r)+1):
                    ok = emit(law_raw, a) or ok
            for m in PAT_MAIN.finditer(text):
                law_raw, art_raw = m.group(1), m.group(2)
                a = cn_to_int(art_raw)
                ok = emit(law_raw, a) or ok
            if not ok and text.strip():
                failures.append(text)
        else:
            if item not in (None,"",[]): failures.append(str(item))

    # 结构化与自由文本两路
    structured = (((jdoc.get("judgment_info") or {}).get("statutes")) or [])
    for entry in structured: handle_item(entry)
    for entry in jdoc.get("statutes") or []: handle_item(entry)
    return parsed, failures

# ---------- 主流程 ----------
def main():
    mc = MongoClient(MONGO_URI)
    DB = mc[DB_NAME]
    D  = DB[COL_DOCS]
    C  = DB[COL_CHUNKS]
    E  = DB[COL_LINKS]

    # 索引（幂等）
    E.create_index([("from_doc",1),("to_doc",1),("edge",1)])

    # 预索引
    docs_index = build_docs_index(D)  # 规范名/别名 → doc_ids
    titles_index, chunk_name_index, doc2arts, doc2vdate = build_chunks_index(C)
    susu_map = find_susufa_docs(C)

    # 民法典 doc_id（用于旧法名称映射至各分编）
    civil_ids = titles_index.get(norm_name("民法典")) or docs_index.get(norm_name("民法典")) or chunk_name_index.get(norm_name("民法典")) or set()
    CIVIL_DOC_ID = list(civil_ids)[0] if civil_ids else None
    CIVIL_GUARANTEE_ARTS = collect_civil_guarantee_articles(C, CIVIL_DOC_ID) if CIVIL_DOC_ID else set()
    CIVIL_CONTRACT_ARTS  = collect_civil_contract_articles(C,  CIVIL_DOC_ID) if CIVIL_DOC_ID else set()
    CIVIL_PROPERTY_ARTS  = collect_civil_property_articles(C,  CIVIL_DOC_ID) if CIVIL_DOC_ID else set()

    # 载入判决类文书
    q = {"doc_type": "judgment"}
    fields = {"doc_id":1,"doc_type":1,"judgment_info":1,"statutes":1}
    judgments = list(D.find(q, fields))
    print(f"[INFO] judgment docs: {len(judgments)}")

    ops: List[UpdateOne] = []
    seen = set()
    miss: Dict[str,int] = {}
    parse_failures: Dict[str, int] = {}

    def record_failures(items: Iterable[str]) -> None:
        for it in items:
            if not it: continue
            parse_failures[it] = parse_failures.get(it, 0) + 1

    def pick_best(cands: Iterable[str], arts: Set[int]) -> Optional[str]:
        best, best_ol, best_v = None, -1, ""
        for did in cands:
            ol = len(arts & (doc2arts.get(did) or set()))
            v = doc2vdate.get(did, "")
            if ol > best_ol or (ol == best_ol and v > best_v):
                best, best_ol, best_v = did, ol, v
        return best

    for j in judgments:
        from_doc = j["doc_id"]
        case_sys  = suggest_system(j)

        # 解析
        parsed_statutes, failed_items = parse_statutes_from_doc(j)
        record_failures(failed_items)

        # 聚合同一法律的条文集合
        law_to_articles: Dict[str, Set[int]] = {}
        rawname_cache: Dict[str, str] = {}
        law_no_articles: Set[str] = set()

        for lname_norm, law_raw, art in parsed_statutes:
            if lname_norm not in law_to_articles:
                law_to_articles[lname_norm] = set()
                rawname_cache[lname_norm]  = law_raw
            if art is None:
                law_no_articles.add(lname_norm)
            else:
                law_to_articles[lname_norm].add(int(art))

        for lname_norm, arts in law_to_articles.items():
            if lname_norm in NAME_BLACKLIST_NORMS:
                continue
            # --- 1) 泛称“关于适用的解释” → 民/行/刑 + 条号重合度 ---
            if lname_norm in GENERIC_INTERP and susu_map:
                prefer = {"civil": "民事诉讼法解释", "admin": "行政诉讼法解释", "criminal": "刑事诉讼法解释"}.get(case_sys)
                cand: List[Tuple[str, int]] = []
                for short, did in susu_map.items():
                    overlap = len(arts & (doc2arts.get(did) or set()))
                    bias = 100 if short == prefer else 0
                    cand.append((did, overlap + bias))
                cand.sort(key=lambda x: -x[1])
                if cand and cand[0][1] > 0:
                    to_doc = cand[0][0]
                    key = (from_doc, to_doc, "applies")
                    if key not in seen:
                        seen.add(key)
                        upd = {"$setOnInsert": {"law_name_raw": rawname_cache[lname_norm]}}
                        if arts:
                            upd["$addToSet"] = {"articles": {"$each": sorted(arts)}}
                        ops.append(UpdateOne({"from_doc": from_doc, "to_doc": to_doc, "edge": "applies"}, upd, upsert=True))
                        C.update_many({"doc_id": to_doc}, {"$addToSet": {"law_alias": rawname_cache[lname_norm]}})
                    continue

            # --- 旧法名 → 民法典分编（合同/担保/物权） ---
            if CIVIL_DOC_ID and lname_norm in {"合同法", "中华人民共和国合同法"}:
                to_doc = CIVIL_DOC_ID
                arts_to_add = CIVIL_CONTRACT_ARTS or arts
                key = (from_doc, to_doc, "applies")
                if key not in seen:
                    seen.add(key)
                    upd = {
                        "$setOnInsert": {"law_name_raw": rawname_cache[lname_norm]},
                        "$addToSet": {"articles": {"$each": sorted(arts_to_add)}},
                    }
                    ops.append(UpdateOne({"from_doc": from_doc, "to_doc": to_doc, "edge":"applies"}, upd, upsert=True))
                    C.update_many({"doc_id": to_doc}, {"$addToSet": {"law_alias": rawname_cache[lname_norm]}})
                continue

            if CIVIL_DOC_ID and lname_norm in {"担保法", "中华人民共和国担保法", "担保法解释", "最高人民法院关于适用中华人民共和国担保法若干问题的解释"}:
                to_doc = CIVIL_DOC_ID
                arts_to_add = CIVIL_GUARANTEE_ARTS or arts
                key = (from_doc, to_doc, "applies")
                if key not in seen:
                    seen.add(key)
                    upd = {
                        "$setOnInsert": {"law_name_raw": rawname_cache[lname_norm]},
                        "$addToSet": {"articles": {"$each": sorted(arts_to_add)}},
                    }
                    ops.append(UpdateOne({"from_doc": from_doc, "to_doc": to_doc, "edge":"applies"}, upd, upsert=True))
                    C.update_many({"doc_id": to_doc}, {"$addToSet": {"law_alias": rawname_cache[lname_norm]}})
                continue

            if CIVIL_DOC_ID and lname_norm in {"物权法", "中华人民共和国物权法"}:
                to_doc = CIVIL_DOC_ID
                arts_to_add = CIVIL_PROPERTY_ARTS or arts
                key = (from_doc, to_doc, "applies")
                if key not in seen:
                    seen.add(key)
                    upd = {
                        "$setOnInsert": {"law_name_raw": rawname_cache[lname_norm]},
                        "$addToSet": {"articles": {"$each": sorted(arts_to_add)}},
                    }
                    ops.append(UpdateOne({"from_doc": from_doc, "to_doc": to_doc, "edge":"applies"}, upd, upsert=True))
                    C.update_many({"doc_id": to_doc}, {"$addToSet": {"law_alias": rawname_cache[lname_norm]}})
                continue

            # --- 3) 常规名匹配：docs 索引 / 标题索引 ---
            candidates: Set[str] = set()
            # 3.1 docs 索引（规范名/别名）
            candidates |= (docs_index.get(lname_norm) or set())
            # 3.2 标题索引（多数解释的标题就是“全称”）
            candidates |= (titles_index.get(lname_norm) or set())
            # 3.3 law_name 索引（chunks 中的 law_name）
            candidates |= (chunk_name_index.get(lname_norm) or set())

            base_ge = "建设工程施工合同纠纷案件适用法律问题的解释"
            if base_ge in rawname_cache[lname_norm]:
                target_norm = norm_name(f"{base_ge}（一）")
                ge_id_set = (titles_index.get(target_norm) or set()) | (docs_index.get(target_norm) or set()) | (chunk_name_index.get(target_norm) or set())
                if ge_id_set:
                    to_doc = next(iter(ge_id_set))
                    key = (from_doc, to_doc, "applies")
                    if key not in seen:
                        seen.add(key)
                        upd = {"$setOnInsert": {"law_name_raw": rawname_cache[lname_norm]}}
                        if arts:
                            upd["$addToSet"] = {"articles": {"$each": sorted(arts)}}
                        ops.append(UpdateOne({"from_doc": from_doc, "to_doc": to_doc, "edge":"applies"}, upd, upsert=True))
                        C.update_many({"doc_id": to_doc}, {"$addToSet": {"law_alias": rawname_cache[lname_norm]}})
                    continue

            if not candidates:
                miss[lname_norm] = miss.get(lname_norm, 0) + 1
                continue

            # 候选不唯一时：按条号重叠度选择；若没有条号就任取一个
            best_id = pick_best(candidates, arts)
            to_doc = best_id or (next(iter(candidates)) if candidates else None)

            key = (from_doc, to_doc, "applies")
            if key in seen: 
                continue
            seen.add(key)

            update_doc = {"$setOnInsert": {"law_name_raw": rawname_cache[lname_norm]}}
            if arts:
                update_doc["$addToSet"] = {"articles": {"$each": sorted(arts)}}
            elif lname_norm in law_no_articles:
                update_doc["$setOnInsert"]["articles"] = []  # 明确“整部/未注明条”
            ops.append(UpdateOne({"from_doc": from_doc, "to_doc": to_doc, "edge":"applies"}, update_doc, upsert=True))

            # --- 成功命中：把“原始全称”追加为该 doc_id 的所有 chunk 的 law_alias ---
            C.update_many({"doc_id": to_doc}, {"$addToSet": {"law_alias": rawname_cache[lname_norm]}})

    # 写入
    if ops:
        res = E.bulk_write(ops, ordered=False)
        print(f"[OK] upsert doc-edges: {res.upserted_count} upserted / {res.modified_count} modified / total_ops={len(ops)}")
    else:
        print("[OK] nothing to link")

    # 报表
    if miss:
        sorted_miss = sorted(miss.items(), key=lambda x: (-x[1], x[0]))
        print("[MISSING LAW NAMES] (norm):")
        for k, v in sorted_miss:
            print(f"  - {k} ×{v}")
    if parse_failures:
        top = sorted(parse_failures.items(), key=lambda x: -x[1])[:10]
        print("[PARSE FAILURES] top 10 raw snippets:")
        for k,v in top: print(f"  - {k} ×{v}")

if __name__ == "__main__":
    main()
