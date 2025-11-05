# -*- coding: utf-8 -*-
"""Utilities for linking judgments to statutes/interpretations.

This script reads judgment documents from ``law_kb_docs`` and attempts to
identify the statutes or judicial interpretations referenced within the
judgment text. For each successful match an ``edge="applies"`` link is upserted
into ``law_kb_links``. When the match relies on a non-canonical form of the law
name, the raw text variant is appended to the target document's chunks as a
``law_alias`` for future runs.

The implementation follows the requirements described in the accompanying
"Dev Brief — Rewrite link_builder_judgment.py" document.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from pymongo import MongoClient

# === Connection configuration ===
MONGO_URI = "mongodb://adminUser:~Q2w3e4r@192.168.110.36:27019"
DB_NAME = "lawKB"
COL_DOCS = "law_kb_docs"
COL_CHUNKS = "law_kb_chunks"
COL_LINKS = "law_kb_links"

TARGET_DOC_TYPES = {"statute", "judicial_interpretation"}
GE_I_DOC = "JI-SPC-CONSTRUCTION-CONTRACT-DISPUTE-I-20201229"
MAX_ARTS_PER_EDGE = 20

# === Normalisation helpers ===
_CN_NUMBER = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
    "千": 1000,
}

_CN_PREFIX = re.compile(r"^\s*中华人民共和国")
_BRACKETS = re.compile(r"[()（）\[\]【】＜＞<>]")
_INNER_QUOTES = re.compile(r"[〈〈⟨<]\s*([^〉》⟩>]+?)\s*[〉》⟩>]")
_WS = re.compile(r"\s+")


def cn_to_int(value: Any) -> int:
    """Convert a Chinese numeral (or mixed string) to ``int``.

    ``value`` may already be numeric. The function supports basic units ``十``,
    ``百`` and ``千``.
    """

    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return 0
    if text.isdigit() or (text[0] == "-" and text[1:].isdigit()):
        try:
            return int(text)
        except ValueError:
            return 0

    text = text.replace("第", "").replace("条", "")
    total = 0
    unit = 1
    temp = 0
    last_was_unit = False
    for char in text:
        if char in ("十", "百", "千"):
            unit_value = _CN_NUMBER[char]
            if temp == 0:
                temp = 1
            total += temp * unit_value
            temp = 0
            unit = unit_value // 10 if unit_value >= 10 else 1
            last_was_unit = True
        else:
            num = _CN_NUMBER.get(char)
            if num is None:
                continue
            total += num
            temp = num
            unit = 1
            last_was_unit = False
    if last_was_unit and temp == 0:
        total += unit
    return total if total > 0 else 0


def norm_title(raw: str) -> str:
    """Normalise a chunk title or alias candidate."""

    if not raw:
        return ""
    text = str(raw)
    text = _INNER_QUOTES.sub(r"\1", text)
    return norm_name(text)


def norm_name(raw: str) -> str:
    if not raw:
        return ""
    text = str(raw).strip()
    text = text.replace("《", "").replace("》", "")
    text = _CN_PREFIX.sub("", text)
    text = re.sub(r"（[^）]*修正[^）]*）", "", text)
    text = re.sub(r"\([^)]*修正[^)]*\)", "", text)
    text = _BRACKETS.sub("", text)
    text = text.replace("·", " ").replace("　", " ")
    text = _WS.sub("", text)
    return text


def canonicalize_raw(raw: str) -> str:
    text = str(raw or "").strip()
    text = text.strip('《》〈〉⟪⟨⟫⟩"“”')
    text = text.replace("·", " ").replace("　", " ")
    text = _WS.sub("", text)
    return text


# === Article extraction ===
_ART_RANGE = re.compile(
    r"第?\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条?\s*(?:至|到|至第|到第|[-~－—–])\s*第?\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条?"
)
_ART_RANGE_SIMPLE = re.compile(
    r"([〇零一二三四五六七八九十百千万两\d]+)\s*[-~－—–]\s*([〇零一二三四五六七八九十百千万两\d]+)"
)
_ART_SINGLE = re.compile(r"第\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条")


def expand_article_no(value: Any) -> Set[int]:
    """Expand different article notations to a set of ``int`` values."""

    results: Set[int] = set()

    def add_range(left: Any, right: Any) -> None:
        l = cn_to_int(left)
        r = cn_to_int(right)
        if l <= 0 and r <= 0:
            return
        if l <= 0:
            l = r
        if r <= 0:
            r = l
        if l > r:
            l, r = r, l
        for num in range(l, r + 1):
            if num > 0:
                results.add(num)

    def handle(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple, set)):
            for sub in item:
                handle(sub)
            return
        if isinstance(item, dict):
            keys = {k.lower(): v for k, v in item.items()}
            if {"l", "r"}.issubset(keys) or {"left", "right"}.issubset(keys) or {"start", "end"}.issubset(keys) or {"from", "to"}.issubset(keys):
                left = keys.get("l") or keys.get("left") or keys.get("start") or keys.get("from")
                right = keys.get("r") or keys.get("right") or keys.get("end") or keys.get("to")
                add_range(left, right)
                return
            for v in item.values():
                handle(v)
            return
        if isinstance(item, (int, float)):
            if int(item) > 0:
                results.add(int(item))
            return
        text = str(item).strip()
        if not text:
            return
        normalized = text.replace("－", "-").replace("—", "-").replace("–", "-").replace("~", "-")
        for match in _ART_RANGE.finditer(normalized):
            add_range(match.group(1), match.group(2))
        normalized = _ART_RANGE.sub(" ", normalized)
        for match in _ART_RANGE_SIMPLE.finditer(normalized):
            add_range(match.group(1), match.group(2))
        normalized = _ART_RANGE_SIMPLE.sub(" ", normalized)
        for match in _ART_SINGLE.finditer(normalized):
            num = cn_to_int(match.group(1))
            if num > 0:
                results.add(num)
        normalized = _ART_SINGLE.sub(" ", normalized)
        for token in re.split(r"[、,，；;\s]", normalized):
            num = cn_to_int(token)
            if num > 0:
                results.add(num)

    handle(value)
    return results


# === Blacklist ===
_BLACKLIST_PATTERNS = [
    re.compile(r"(省|市|自治区|自治州|县|区).*(条例|规定|办法|细则)$"),
    re.compile(r"(部|委|局|总局|办|会).*(规定|规则|办法|程序|细则|标准|指引|指南|制度|通则|实施意见|若干意见|复函|批复)$"),
    re.compile(r"纪要$"),
    re.compile(r"(暂行|试行).*(规定|办法|细则)$"),
]
_BLACKLIST_EXACT = {
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


def is_blacklisted(name_norm: str) -> bool:
    if not name_norm:
        return True
    if name_norm in _BLACKLIST_EXACT:
        return True
    return any(pattern.search(name_norm) for pattern in _BLACKLIST_PATTERNS)


# === Parsing statutes inside judgments ===
LAW_L = r"[《〈<⟪⟨]"
LAW_R = r"[》〉>⟫⟩]"
BRK_OPT = r"(?:\s*(?:（[^）]*）|\([^)]*\)|\[[^\]]*\]|【[^】]*】))*"
PAT_RANGE = re.compile(
    rf"{LAW_L}\s*(.+?)\s*{LAW_R}{BRK_OPT}\s*第\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条\s*(?:至|到|[-~－—–])\s*第\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条"
)
PAT_SINGLE = re.compile(
    rf"{LAW_L}\s*(.+?)\s*{LAW_R}{BRK_OPT}\s*第\s*([〇零一二三四五六七八九十百千万两\d]+)\s*条"
)

@dataclass
class ParsedLaw:
    law_norm: str
    raw_text: str
    articles: Set[int]

def _parse_structured_item(item: Dict[str, Any]) -> Tuple[str, Set[int]]:
    law_raw = (
        item.get("law")
        or item.get("name")
        or item.get("title")
        or item.get("law_name")
        or ""
    )
    law_raw = str(law_raw or "").strip()
    articles: Set[int] = set()
    for key in ("article", "articles", "article_no"):
        if key in item:
            articles |= expand_article_no(item[key])
    return law_raw, articles


def _parse_textual_entry(entry: str) -> Iterator[Tuple[str, Set[int]]]:
    text = str(entry or "")
    matches: List[Tuple[int, int]] = []
    for match in PAT_RANGE.finditer(text):
        law_raw = match.group(1)
        left = cn_to_int(match.group(2))
        right = cn_to_int(match.group(3))
        if left <= 0 and right <= 0:
            continue
        if left <= 0:
            left = right
        if right <= 0:
            right = left
        if left > right:
            left, right = right, left
        articles = {num for num in range(left, right + 1) if num > 0}
        if not articles:
            continue
        matches.append(match.span())
        yield law_raw, articles
    def span_overlaps(span: Tuple[int, int]) -> bool:
        start, end = span
        for s, e in matches:
            if not (end <= s or start >= e):
                return True
        return False
    for match in PAT_SINGLE.finditer(text):
        if span_overlaps(match.span()):
            continue
        law_raw = match.group(1)
        article = cn_to_int(match.group(2))
        if article <= 0:
            continue
        yield law_raw, {article}


def _iter_judgment_statute_sources(jdoc: Dict[str, Any]) -> Iterator[Any]:
    info = jdoc.get("judgment_info") or {}
    meta = info.get("meta") if isinstance(info, dict) else None
    sources: List[Any] = [
        info.get("statutes") if isinstance(info, dict) else None,
        meta.get("statutes") if isinstance(meta, dict) else None,
        jdoc.get("statutes"),
    ]
    seen_ids: Set[int] = set()
    for source in sources:
        if not source:
            continue
        if id(source) in seen_ids:
            continue
        seen_ids.add(id(source))
        if isinstance(source, (list, tuple, set)):
            for item in source:
                if item not in (None, "", []):
                    yield item
        elif isinstance(source, dict):
            yield source
        else:
            yield source


def parse_statutes_from_doc(jdoc: Dict[str, Any]) -> Tuple[List[ParsedLaw], List[str]]:
    aggregated: Dict[str, ParsedLaw] = {}
    failures: Counter[str] = Counter()

    def push(law_raw: str, articles: Set[int]) -> None:
        raw_text = str(law_raw or "").strip()
        if not raw_text:
            return
        law_norm = norm_name(raw_text)
        if not law_norm:
            return
        parsed = aggregated.get(law_norm)
        if not parsed:
            aggregated[law_norm] = ParsedLaw(law_norm, raw_text, set(articles))
        else:
            parsed.articles |= set(articles)
            if len(raw_text) > len(parsed.raw_text):
                parsed.raw_text = raw_text

    def handle(item: Any) -> None:
        if isinstance(item, dict):
            law_raw, articles = _parse_structured_item(item)
            if not law_raw and not articles:
                return
            push(law_raw, articles)
        elif isinstance(item, str):
            ok = False
            for law_raw, arts in _parse_textual_entry(item):
                push(law_raw, arts)
                ok = True
            if not ok and item.strip():
                failures[item.strip()] += 1
        elif item not in (None, "", []):
            failures[str(item)] += 1

    for entry in _iter_judgment_statute_sources(jdoc):
        handle(entry)

    return list(aggregated.values()), list(failures.elements())


# === Index building ===
@dataclass
class TargetIndex:
    name2docs: Dict[str, Set[str]]
    alias2docs: Dict[str, Set[str]]
    doc2arts: Dict[str, Set[int]]
    doc2vdate: Dict[str, str]
    doc_meta: Dict[str, Dict[str, Any]]


def build_target_index(db) -> TargetIndex:
    docs_col = db[COL_DOCS]
    chunks_col = db[COL_CHUNKS]

    name2docs: Dict[str, Set[str]] = defaultdict(set)
    alias2docs: Dict[str, Set[str]] = defaultdict(set)
    doc2arts: Dict[str, Set[int]] = defaultdict(set)
    doc2vdate: Dict[str, str] = {}
    doc_meta: Dict[str, Dict[str, Any]] = {}

    doc_cursor = docs_col.find(
        {"doc_type": {"$in": list(TARGET_DOC_TYPES)}},
        {"doc_id": 1, "law_name": 1, "version_date": 1, "doc_type": 1},
    )
    doc_ids: List[str] = []
    for doc in doc_cursor:
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
        doc_ids.append(doc_id)
        doc_meta[doc_id] = {
            "law_name": doc.get("law_name") or "",
            "version_date": doc.get("version_date") or "",
            "doc_type": doc.get("doc_type"),
        }
        norm = norm_name(doc.get("law_name"))
        if norm:
            name2docs[norm].add(doc_id)

    if not doc_ids:
        return TargetIndex(name2docs, alias2docs, doc2arts, doc2vdate, doc_meta)

    chunk_cursor = chunks_col.find(
        {"doc_id": {"$in": doc_ids}},
        {"doc_id": 1, "title": 1, "law_name": 1, "law_alias": 1, "article_no": 1, "version_date": 1},
    )
    for chunk in chunk_cursor:
        doc_id = chunk.get("doc_id")
        if not doc_id:
            continue
        title_norm = norm_title(chunk.get("title"))
        if title_norm:
            name2docs[title_norm].add(doc_id)
        law_norm = norm_name(chunk.get("law_name"))
        if law_norm:
            name2docs[law_norm].add(doc_id)
        for alias in chunk.get("law_alias") or []:
            alias_norm = norm_title(alias)
            if alias_norm:
                alias2docs[alias_norm].add(doc_id)
        doc2arts[doc_id] |= expand_article_no(chunk.get("article_no"))
        chunk_vdate = chunk.get("version_date") or ""
        base_vdate = doc_meta.get(doc_id, {}).get("version_date") or ""
        if chunk_vdate and chunk_vdate > base_vdate:
            doc2vdate[doc_id] = chunk_vdate
        elif base_vdate and doc_id not in doc2vdate:
            doc2vdate[doc_id] = base_vdate
    for doc_id, meta in doc_meta.items():
        if doc_id not in doc2vdate and meta.get("version_date"):
            doc2vdate[doc_id] = meta["version_date"]

    return TargetIndex(name2docs, alias2docs, doc2arts, doc2vdate, doc_meta)


# === Civil Code sections ===
_SECTION_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = [
    ("总则编", re.compile(r"第一编\s*总则|总则编")),
    ("物权编", re.compile(r"第二编\s*物权|物权编")),
    ("合同编", re.compile(r"第三编\s*合同|合同编")),
    ("担保物权", re.compile(r"第四分编\s*担保物权|担保物权")),
    ("保证合同", re.compile(r"保证合同")),
    ("婚姻家庭编", re.compile(r"第五编\s*婚姻家庭|婚姻家庭编")),
    ("侵权责任编", re.compile(r"第七编\s*侵权责任|侵权责任编")),
]


def build_civil_sections(db, civil_doc_id: Optional[str]) -> Dict[str, Set[int]]:
    if not civil_doc_id:
        return {}
    sections: Dict[str, Set[int]] = {name: set() for name, _ in _SECTION_PATTERNS}
    chunks_col = db[COL_CHUNKS]
    cursor = chunks_col.find(
        {"doc_id": civil_doc_id},
        {"path": 1, "title": 1, "article_no": 1},
    )
    for chunk in cursor:
        parts: List[str] = []
        path = chunk.get("path") or {}
        if isinstance(path, dict):
            parts.extend(str(v or "") for v in path.values())
        title = chunk.get("title") or ""
        if title:
            parts.append(str(title))
        joined = " ".join(parts)
        articles = expand_article_no(chunk.get("article_no"))
        if not articles:
            continue
        for section_name, pattern in _SECTION_PATTERNS:
            if pattern.search(joined):
                sections[section_name] |= articles
    return {name: arts for name, arts in sections.items() if arts}


_CIVIL_ROUTING_RULES: Sequence[Tuple[re.Pattern[str], Tuple[str, ...]]] = [
    (re.compile(r"合同法"), ("合同编",)),
    (re.compile(r"担保法"), ("担保物权", "保证合同")),
    (re.compile(r"物权法"), ("物权编",)),
    (re.compile(r"侵权责任法"), ("侵权责任编",)),
    (re.compile(r"婚姻法"), ("婚姻家庭编",)),
    (re.compile(r"民法总则|民法通则"), ("总则编",)),
]


# === Routing helpers ===
GENERIC_INTERP_NORMS = {
    norm_name("最高人民法院关于适用的解释"),
    norm_name("关于适用的解释"),
}


@dataclass
class RoutingContext:
    civil_doc_id: Optional[str]
    civil_sections: Dict[str, Set[int]]
    litigation_targets: Dict[str, Optional[str]]


def route_specials(
    law_norm: str,
    raw_text: str,
    arts: Set[int],
    case_system: str,
    ctx: RoutingContext,
    doc2arts: Dict[str, Set[int]],
) -> Tuple[Optional[str], Optional[Set[int]], bool]:
    # 建工解释族 → 解释（一）
    if "建设工程施工合同纠纷案件适用法律问题的解释" in law_norm or "建设工程施工合同纠纷案件适用法律问题的解释" in raw_text:
        return GE_I_DOC, set(arts), True

    # 旧法路由到民法典
    if ctx.civil_doc_id:
        for pattern, section_names in _CIVIL_ROUTING_RULES:
            if pattern.search(law_norm):
                section_articles: Set[int] = set()
                for section_name in section_names:
                    section_articles |= ctx.civil_sections.get(section_name, set())
                filtered = set(arts) & section_articles if arts and section_articles else set()
                return ctx.civil_doc_id, filtered if filtered else None, True

    # “关于适用的解释”模糊 → 根据系统偏好
    if law_norm in GENERIC_INTERP_NORMS:
        targets = ctx.litigation_targets
        preferred = targets.get(case_system)
        candidates: List[Tuple[str, int]] = []
        for key in ("civil", "admin", "criminal"):
            doc_id = targets.get(key)
            if not doc_id:
                continue
            overlap = len(arts & doc2arts.get(doc_id, set()))
            bias = 100 if doc_id == preferred else 0
            candidates.append((doc_id, overlap + bias))
        if candidates:
            candidates.sort(key=lambda item: (-item[1], item[0]))
            chosen, score = candidates[0]
            if score > 0:
                return chosen, set(arts), False
    return None, None, False


def pick_best(candidates: Iterable[str], arts: Set[int], doc2arts: Dict[str, Set[int]], doc2vdate: Dict[str, str]) -> Optional[str]:
    best_doc: Optional[str] = None
    best_key: Tuple[int, str] = (-1, "")
    arts_set = set(arts)
    for doc_id in candidates:
        overlap = len(arts_set & doc2arts.get(doc_id, set()))
        vdate = doc2vdate.get(doc_id, "")
        key = (overlap, vdate)
        if key > best_key:
            best_key = key
            best_doc = doc_id
    return best_doc


def should_write_alias(raw_text: str, law_norm: str, alias_hit: bool, forced: bool) -> bool:
    if forced or alias_hit:
        return True
    cleaned = canonicalize_raw(raw_text)
    return cleaned != law_norm


def upsert_edge(
    links_col,
    chunks_col,
    from_doc: str,
    to_doc: str,
    raw_law_text: str,
    arts: Optional[Set[int]],
    write_alias: bool,
) -> None:
    query = {"from_doc": from_doc, "to_doc": to_doc, "edge": "applies"}
    update: Dict[str, Any] = {"$setOnInsert": {"law_name_raw": raw_law_text}}
    if arts:
        filtered = sorted({int(x) for x in arts if int(x) > 0})
        if filtered and len(filtered) <= MAX_ARTS_PER_EDGE:
            update["$addToSet"] = {"articles": {"$each": filtered}}
    links_col.update_one(query, update, upsert=True)
    if write_alias and raw_law_text:
        chunks_col.update_many({"doc_id": to_doc}, {"$addToSet": {"law_alias": raw_law_text}})


# === Case system detection ===
_CASE_HINTS = {
    "admin": ["行政", "行诉"],
    "criminal": ["刑事", "刑诉"],
}


def detect_case_system(jdoc: Dict[str, Any]) -> str:
    info = jdoc.get("judgment_info") or {}
    meta = info.get("meta") if isinstance(info, dict) else None
    text_parts: List[str] = []
    containers: List[Dict[str, Any]] = []
    if isinstance(info, dict):
        containers.append(info)
    if isinstance(meta, dict):
        containers.append(meta)
    if isinstance(jdoc, dict):
        containers.append(jdoc)
    for key in ("case_system", "case_type", "cause", "trial_procedure", "title"):
        for container in containers:
            value = container.get(key)
            if isinstance(value, str):
                text_parts.append(value)
                break
    combined = "；".join(text_parts)
    for system, hints in _CASE_HINTS.items():
        if any(hint in combined for hint in hints):
            return system
    return "civil"


# === Litigation interpretation helper ===
_LITIGATION_NAMES = {
    "civil": norm_name("最高人民法院关于适用中华人民共和国民事诉讼法的解释"),
    "admin": norm_name("最高人民法院关于适用中华人民共和国行政诉讼法的解释"),
    "criminal": norm_name("最高人民法院关于适用中华人民共和国刑事诉讼法的解释"),
}


def build_litigation_targets(index: TargetIndex) -> Dict[str, Optional[str]]:
    targets: Dict[str, Optional[str]] = {"civil": None, "admin": None, "criminal": None}

    for system, name_norm in _LITIGATION_NAMES.items():
        doc_ids = index.name2docs.get(name_norm) or set()
        if not doc_ids:
            continue
        chosen = pick_best(doc_ids, set(), index.doc2arts, index.doc2vdate)
        if not chosen and doc_ids:
            chosen = sorted(doc_ids)[0]
        targets[system] = chosen
    return targets


# === Main ===


def main() -> None:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    docs_col = db[COL_DOCS]
    chunks_col = db[COL_CHUNKS]
    links_col = db[COL_LINKS]

    links_col.create_index([("from_doc", 1), ("to_doc", 1), ("edge", 1)], unique=True)

    index = build_target_index(db)
    litigation_targets = build_litigation_targets(index)

    civil_doc_candidates = []
    for doc_id, meta in index.doc_meta.items():
        name_norm = norm_name(meta.get("law_name"))
        if "民法典" in name_norm:
            civil_doc_candidates.append((meta.get("version_date") or "", doc_id))
    civil_doc_id = max(civil_doc_candidates)[1] if civil_doc_candidates else None
    civil_sections = build_civil_sections(db, civil_doc_id)
    routing_ctx = RoutingContext(civil_doc_id, civil_sections, litigation_targets)

    missing_counter: Counter[str] = Counter()
    parse_failure_counter: Counter[str] = Counter()

    cursor = docs_col.find(
        {"doc_type": "judgment"},
        {"doc_id": 1, "judgment_info": 1, "statutes": 1},
    )

    processed = 0
    for jdoc in cursor:
        from_doc = jdoc.get("doc_id")
        if not from_doc:
            continue
        processed += 1
        case_system = detect_case_system(jdoc)
        parsed_laws, failures = parse_statutes_from_doc(jdoc)
        for failure in failures:
            parse_failure_counter[failure] += 1

        for parsed in parsed_laws:
            law_norm = parsed.law_norm
            if is_blacklisted(law_norm):
                continue
            arts = set(parsed.articles)
            to_doc_override, arts_override, forced_alias = route_specials(
                law_norm,
                parsed.raw_text,
                arts,
                case_system,
                routing_ctx,
                index.doc2arts,
            )
            if to_doc_override:
                final_doc = to_doc_override
                final_arts = arts_override if arts_override is not None else (set(arts) if arts else None)
                write_alias = should_write_alias(parsed.raw_text, law_norm, False, forced_alias)
                upsert_edge(links_col, chunks_col, from_doc, final_doc, parsed.raw_text, final_arts, write_alias)
                continue

            name_candidates = index.name2docs.get(law_norm) or set()
            alias_candidates = index.alias2docs.get(law_norm) or set()
            candidates = set(name_candidates) | set(alias_candidates)
            if not candidates:
                missing_counter[law_norm] += 1
                continue
            chosen = pick_best(candidates, arts, index.doc2arts, index.doc2vdate)
            if not chosen:
                chosen = sorted(candidates)[0]
            alias_hit = bool(alias_candidates - name_candidates and chosen in alias_candidates)
            final_arts = set(arts)
            write_alias = should_write_alias(parsed.raw_text, law_norm, alias_hit, False)
            upsert_edge(links_col, chunks_col, from_doc, chosen, parsed.raw_text, final_arts, write_alias)

    if processed == 0:
        print("[INFO] No judgment documents found.")
    else:
        print(f"[INFO] processed judgments: {processed}")

    if missing_counter:
        print("[MISSING LAW NAMES]")
        for name, count in missing_counter.most_common(20):
            print(f"  - {name} ×{count}")
    if parse_failure_counter:
        print("[PARSE FAILURES]")
        for text, count in parse_failure_counter.most_common(20):
            print(f"  - {text} ×{count}")


if __name__ == "__main__":
    main()
