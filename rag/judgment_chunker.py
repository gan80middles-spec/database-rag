# -*- coding: utf-8 -*-
"""
judgment_chunk.py  (TXT 精简版)
- 仅读取 .txt，自动从目录推断 case_system / case_subtype
- “执行”忽略二级目录，按文件名关键词 → 12 类 subtype
- 分段锚点：诉讼请求/审理经过/查明/本院认为/主文/依据/费用/赔偿等
- 切块：以段为原子，合并到 700–1200 字，重叠 120 字；主文/依据尽量独立
- 轻量要素抽取：案号、法院、日期、审级、文书种类、引用法条
- 输出：每个源文件 → 同路径结构的 .jsonl

用法
python judgment_chunk.py --in_dir 文书 --out_dir chunks \
  --min_chars 700 --max_chars 1200 --overlap 120 --workers 4
"""

import os
import re
import json
import argparse
from datetime import datetime
from hashlib import sha1
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 基础清洗 ----------
def normalize_text(t: str) -> str:
    t = t.replace("\u3000"," ").replace("\xa0"," ").replace("\t"," ")
    t = re.sub(r"[ \r\f]+"," ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"^\s*[-－—–~·•]\s*\d+\s*[-－—–~·•]\s*$","", t, flags=re.M)
    return t.strip()

# ---------- 轻量元数据 ----------
RE_CASE_NO = re.compile(r"[（(]\d{4}[）)]\s*[\u4e00-\u9fa5A-Z0-9]{2,}\d+号")
RE_CASE_NO_STRICT = re.compile(r"[（(]\d{4}[）)][^号\n]{0,40}号")
RE_COURT   = re.compile(r"[\u4e00-\u9fa5]{2,30}人民法院")
RE_DATE_CHINESE = re.compile(r"([〇零○ＯO一二三四五六七八九十百千两]{3,})年([〇零○ＯO一二三四五六七八九十两]{1,3})月([〇零○ＯO一二三四五六七八九十两]{1,3})日")
LAW_NAME_RE = re.compile(r"《([^》\n]{1,60})》")

NUM_TOKEN = "一二三四五六七八九十百千万零〇○ＯO两0-9"
ARTICLE_RE = re.compile(
    rf"第[{NUM_TOKEN}]+(?:之[{NUM_TOKEN}]+)?条"
    rf"(?:第[{NUM_TOKEN}]+款)?"
    rf"(?:第[{NUM_TOKEN}]+[项目段])?"
)

NORMATIVE_SUFFIXES = ("法典", "法律", "法", "条例", "规定", "解释", "决定", "办法", "规则", "纪要")
BAN_KEYWORDS = (
    "合同", "协议", "议定书", "章程", "指南", "细则", "目录", "名录",
    "报告", "汇报", "说明", "情况", "统计表", "表", "清单",
    "证书", "许可证", "执照", "证明", "授权书",
    "通知书", "告知书", "答复", "批复", "申请书",
    "购销合同", "委托加工合同"
)

PAREN_PAIRS = {"（": "）", "(": ")", "【": "】", "[": "]", "〔": "〕"}
STATUTE_CONNECTORS = set("、,，；;和及与或并及至到—-~ 　")

STATUTE_VERSION_PAREN_RE = re.compile(r"(》)\s*(?:（[^）]{0,30}）|\([^)]{0,30}\))*")


def _is_normative_title(name: str) -> bool:
    """判断《……》内是否为规范性文件名，而不是合同/证书/报告等。"""
    n = (name or "").strip()
    if not n:
        return False
    if any(bad in n for bad in BAN_KEYWORDS):
        return False
    if any(n.endswith(suf) for suf in NORMATIVE_SUFFIXES):
        return True
    return False


def _skip_whitespace(text: str, pos: int) -> int:
    length = len(text)
    while pos < length and text[pos].isspace():
        pos += 1
    return pos


def _consume_parenthetical(text: str, pos: int) -> int:
    ch = text[pos]
    end_ch = PAREN_PAIRS.get(ch)
    if not end_ch:
        return pos
    depth = 1
    i = pos + 1
    length = len(text)
    while i < length:
        cur = text[i]
        if cur == ch:
            depth += 1
        elif cur == end_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return pos


def _extract_statutes(text: str, allow_title_without_article: bool = False):
    """
    只返回“规范性文件 + 条/款/项”的引用；
    allow_title_without_article=True 时，才会保留不带条/款/项的纯标题（默认 False）。
    """
    clean = _fix_broken_statute_spans(text)
    results = []
    idx, length = 0, len(clean)

    while idx < length:
        m = LAW_NAME_RE.search(clean, idx)
        if not m:
            break

        raw_name = m.group(1).strip()
        if not _is_normative_title(raw_name):
            idx = m.end()
            continue

        law_text = clean[m.start():m.end()]
        pos = m.end()

        while True:
            pos = _skip_whitespace(clean, pos)
            if pos >= length or clean[pos] not in PAREN_PAIRS:
                break
            end = _consume_parenthetical(clean, pos)
            if end <= pos or end - pos > 24:
                break
            inner = clean[pos + 1:end - 1].strip()
            if inner and not re.search(r"(以下简称|下称|简称)", inner):
                law_text += clean[pos:end]
            pos = end

        lookahead_limit = min(length, pos + 160)
        consumed = False

        while pos < lookahead_limit:
            pos = _skip_whitespace(clean, pos)
            if pos >= lookahead_limit:
                break

            ch = clean[pos]
            if ch in PAREN_PAIRS:
                end = _consume_parenthetical(clean, pos)
                if end > pos and end - pos <= 40:
                    pos = end
                    continue
                break

            if ch in STATUTE_CONNECTORS:
                pos += 1
                continue

            m_article = ARTICLE_RE.match(clean, pos)
            if m_article:
                results.append(law_text + m_article.group(0))
                pos = m_article.end()
                consumed = True
                continue

            break

        if not consumed and allow_title_without_article:
            results.append(law_text)

        idx = max(pos, m.end())

    deduped = _dedupe_statutes(results)
    return deduped[:50]


def _dedupe_statutes(statutes):
    seen = {}
    for statute in statutes:
        key = _normalize_statute_key(statute)
        if key not in seen:
            seen[key] = statute
    return list(seen.values())


def _normalize_statute_key(statute: str) -> str:
    if not statute:
        return ""
    normalized = STATUTE_VERSION_PAREN_RE.sub(r"\1", statute)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized

CASE_CODE_TO_LEVEL = {
    # 再审链路
    "民申": "再审审查", "行申": "再审审查", "刑申": "再审审查", "知民申": "再审审查", "赔申": "再审审查", "破申": "再审审查",
    "民监": "再审审查", "行监": "再审审查", "刑监": "再审审查", "知民监": "再审审查", "赔监": "再审审查", "破监": "再审审查",
    "民再": "再审", "行再": "再审", "刑再": "再审", "知民再": "再审", "赔再": "再审", "破再": "再审",
    # 二审 / 一审
    "民终": "二审", "行终": "二审", "刑终": "二审", "知民终": "二审", "知行终": "二审", "知刑终": "二审", "赔终": "二审", "破终": "二审",
    "民初": "一审", "行初": "一审", "刑初": "一审", "知民初": "一审", "知行初": "一审", "知刑初": "一审", "赔初": "一审", "破初": "一审",
}

# 结构： （2023）川民终123号 → 捕获年份与案由代码
RE_CASE_IN_TEXT = re.compile(r"[（(]\s*(20\d{2})\s*[）)]\s*([^\s\n号]*?)\s*(\d+)号")

DOC_TYPE_TAILS = [
    "司法赔偿监督审查决定书",
    "赔偿监督审查决定书",
    "赔偿复议决定书",
    "赔偿判决书",
    "赔偿决定书",
    "赔偿调解书",
    "通知书",
    "调解书",
    "判决书",
    "裁定书",
    "决定书",
]
RE_DOC_TYPE_LINE = re.compile(
    r"(刑事|民事|行政|执行|国家赔偿|赔偿|司法赔偿)?"
    r"(司法赔偿监督审查决定书|赔偿监督审查决定书|赔偿复议决定书|赔偿判决书|赔偿决定书|赔偿调解书|判决书|裁定书|决定书|调解书|通知书)$"
)

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

def detect_doc_type(text: str, file_path: str = "") -> str:
    head = text[:3000]
    best = ""

    def consider(candidate: str):
        nonlocal best
        if not candidate:
            return
        if not best or len(candidate) > len(best):
            best = candidate

    for raw in head.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        collapsed = _clean_spaces(stripped)
        m = RE_DOC_TYPE_LINE.search(collapsed)
        if m:
            prefix = m.group(1) or ""
            tail = m.group(2)
            if prefix and tail.startswith(prefix):
                consider(tail)
            else:
                consider(prefix + tail)
            continue

        for tail in DOC_TYPE_TAILS:
            if collapsed.endswith(tail):
                consider(tail)

    if file_path:
        base = os.path.splitext(os.path.basename(file_path))[0]
        collapsed = _clean_spaces(base)
        for tail in DOC_TYPE_TAILS:
            if tail in collapsed:
                consider(tail)
        for prefix in ["刑事", "民事", "行政", "执行", "国家赔偿", "赔偿", "司法赔偿"]:
            for tail in DOC_TYPE_TAILS:
                combo = prefix + tail
                if combo in collapsed and prefix not in tail:
                    consider(combo)
    return best

def detect_trial_level(text: str, case_no: str = "") -> str:
    src = case_no
    if not src:
        head = text[:6000]
        m = RE_CASE_IN_TEXT.search(head)
        if m:
            src = m.group(0)

    if src:
        m = RE_CASE_IN_TEXT.search(src)
        if m:
            code = m.group(2)
            # 精确匹配（最长优先）
            for k in sorted(CASE_CODE_TO_LEVEL.keys(), key=len, reverse=True):
                if k in code:
                    return CASE_CODE_TO_LEVEL[k]
        # 死刑复核/执行/赔委等特殊词
        if "刑复" in src:
            return "死刑复核"
        if "委赔监" in src:
            return "赔偿委员会申诉"
        if "委赔" in src and "委赔监" not in src:
            return "赔偿委员会决定"
        if re.search(r"[\u4e00-\u9fa5A-Z]*执[\u4e00-\u9fa5A-Z]*\d+号", src):
            return "执行程序"

    # 正文兜底
    head = text[:8000]
    if re.search(r"本(判决|裁定)为终审", head):
        return "二审"
    if re.search(r"死刑复核", head):
        return "死刑复核"
    if re.search(r"再审(申请)?审查", head):
        return "再审审查"
    if re.search(r"(再审|重审).{0,6}(判决|裁定|决定)", head):
        return "再审"
    if re.search(r"二审(判决|裁定|决定)?", head):
        return "二审"
    if re.search(r"一审(判决|裁定|决定)?", head):
        return "一审"
    if re.search(r"执行(裁定|决定|异议|复议|分配|变卖|拍卖|终结|终止|恢复|追加|变更)", head):
        return "执行程序"
    return ""


def _chinese_digits_to_int(s: str) -> int:
    mapping = {
        "零": 0, "〇": 0, "○": 0, "Ｏ": 0, "O": 0, "o": 0,
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    }
    digits = []
    for ch in s:
        if ch in mapping:
            digits.append(str(mapping[ch]))
        elif ch.isdigit():
            digits.append(ch)
        else:
            return -1
    try:
        return int("".join(digits))
    except ValueError:
        return -1


def _parse_chinese_number(num: str) -> int:
    num = num.strip()
    if not num:
        return -1
    digit_value = _chinese_digits_to_int(num)
    if digit_value >= 0:
        return digit_value

    char_map = {
        "零": 0, "〇": 0, "○": 0, "Ｏ": 0,
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    }
    total = 0
    unit = 0
    for ch in num:
        if ch == "十":
            total += (unit if unit else 1) * 10
            unit = 0
        else:
            val = char_map.get(ch)
            if val is None:
                return -1
            unit = unit + val
    total += unit
    return total


def _clean_court_candidate(candidate: str) -> str:
    if not candidate:
        return ""
    candidate = candidate.strip()
    candidate = re.sub(r"^[，。、；：\s]+", "", candidate)

    bad_prefixes = [
        "不服于", "因不服", "不服",
        "原告向", "被告向", "上诉人向", "被上诉人向", "申请人向", "请求人向", "检察院向",
        "原告", "被告", "上诉人", "被上诉人", "申请人", "请求人", "检察院",
        "经", "由", "对", "因", "就", "针对", "依照", "根据",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in bad_prefixes:
            if candidate.startswith(prefix) and len(candidate) - len(prefix) >= 4:
                candidate = candidate[len(prefix):].lstrip("，。、；： \t")
                changed = True
                break

    best = ""
    for m in RE_COURT.finditer(candidate):
        piece = m.group(0).strip()
        if len(piece) > len(best):
            best = piece
    return best

# 预编译正则
_RX_COURT_NAME = re.compile(r"(?P<name>[\u4e00-\u9fa5]{2,40}人民法院(?:赔偿委员会)?)")
# 文书种类行（含国家赔偿常见样式：决定书/赔偿决定书/国家赔偿决定书/中止审理决定书等）
_RX_DOCTYPE = re.compile(r"(判决书|裁定书|决定书|调解书|赔偿决定书|国家赔偿决定书|中止审理决定书)")
# 精确整行仅为“××人民法院(赔偿委员会)”的情况
_RX_EXACT = re.compile(r"^(?P<name>[\u4e00-\u9fa5]{2,40}人民法院(?:赔偿委员会)?)$")

_BAD_PREFIX = ("请求", "撤销", "维持", "依照", "依据", "判处", "决定", "裁定", "驳回", "公告")

def _detect_court(text: str, head_n: int = 100) -> str:
    """
    从裁判/国家赔偿类文书前若干行中抽取承办机关：
    优先返回“××人民法院赔偿委员会”，否则返回“××人民法院”，找不到返回空串。
    """
    # 预处理：去除空行，仅保留前 head_n 行
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:head_n]

    # 1) 精确整行匹配（标题独占一行）
    for ln in lines:
        m = _RX_EXACT.match(ln)
        if m:
            return m.group("name")

    # 2) 找到文书种类行（判决书/裁定书/决定书/赔偿决定书…），在其上方近邻行里找法院机构
    doc_idx = next((i for i, ln in enumerate(lines) if _RX_DOCTYPE.search(ln)), -1)
    if doc_idx > 0:
        for j in range(max(0, doc_idx - 5), doc_idx):  # 往上看 5 行
            m = _RX_COURT_NAME.search(lines[j])
            if m:
                # 若同一行出现“赔偿委员会”，优先返回带后缀
                name = m.group("name")
                if name.endswith("赔偿委员会"):
                    return name
                # 先暂存，若稍后未发现“赔偿委员会”，再回退使用
                fallback = name
                # 再向更近的上方继续搜“赔偿委员会”
                for k in range(max(0, j - 2), j + 1):
                    mm = _RX_COURT_NAME.search(lines[k])
                    if mm and mm.group("name").endswith("赔偿委员会"):
                        return mm.group("name")
                return fallback  # 就近返回

    # 3) 通用候选 + 打分（位置、是否独占、是否带“赔偿委员会”、是否带可疑前缀等）
    candidates = []
    for i, ln in enumerate(lines):
        m = _RX_COURT_NAME.search(ln)
        if not m:
            continue
        name = m.group("name")
        score = 100

        # 越靠前分越低
        if i <= 5: score -= 20
        elif i <= 10: score -= 15
        elif i <= 20: score -= 8
        else: score -= 3

        # 紧邻文书种类行的加权
        if doc_idx != -1:
            dist = doc_idx - i
            if 0 < dist <= 5:
                score -= (15 - dist)  # 越近分越低

        # 独占一行 / 行尾即为机构名
        if _RX_EXACT.match(ln): score -= 15
        if ln.endswith(name): score -= 5

        # 含“赔偿委员会”优先
        if name.endswith("赔偿委员会"): score -= 10

        # 可疑开头（多为叙述句，而非抬头）
        if ln[:2] in _BAD_PREFIX or any(ln.startswith(p) for p in _BAD_PREFIX):
            score += 10

        # 名称长度略作偏好：中等长度更常见
        score += abs(len(name) - 10) * 0.5

        candidates.append((score, i, len(name), name))

    if candidates:
        candidates.sort()
        # 若榜首不是“赔偿委员会”，但前若干名里存在“赔偿委员会”，择其一
        top_names = [c[3] for c in candidates[:5]]
        for nm in top_names:
            if nm.endswith("赔偿委员会"):
                return nm
        return candidates[0][3]

    return ""


def detect_judgment_date(text: str) -> str:
    matches = []
    length = len(text)

    for m in RE_DATE_CHINESE.finditer(text):
        year_raw, month_raw, day_raw = m.group(1), m.group(2), m.group(3)
        year = _parse_chinese_number(year_raw)
        month = _parse_chinese_number(month_raw)
        day = _parse_chinese_number(day_raw)
        if year < 0 or month <= 0 or day <= 0:
            continue
        if month > 12 or day > 31:
            continue
        ctx = text[max(0, m.start()-20):min(length, m.end()+20)]
        has_keyword = bool(re.search(r"判决|裁定|决定|调解|赔偿|结案|审理终结", ctx))
        dist = length - m.start()
        score = (0 if has_keyword else 1000) + dist
        matches.append((score, year, month, day))

    if not matches:
        return ""

    best = min(matches, key=lambda x: (x[0], -x[1], -x[2], -x[3]))
    _, year, month, day = best
    return f"{year:04d}-{month:02d}-{day:02d}"

def _fix_broken_statute_spans(text: str) -> str:
    # 书名号跨行、条款拆行 → 合并
    t = re.sub(r"《\s*([^》\n]{1,40})\s*》", r"《\1》", text)
    t = re.sub(r"《([^》\n]{1,25})\n+([^》\n]{1,25})》", r"《\1\2》", t)
    t = re.sub(r"第\s*([一二三四五六七八九十百千0-9]+)\s*条", r"第\1条", t)
    t = re.sub(r"第\s*([一二三四五六七八九十百千0-9]+)\s*款", r"第\1款", t)
    t = re.sub(r"第\s*([一二三四五六七八九十百千0-9]+)\s*[项目]", r"第\1项", t)
    return t

def extract_light_meta(text: str, file_path: str = ""):
    head = text[:4000]
    m_case_no = RE_CASE_NO.search(head)
    court = _detect_court(text)
    doc_type  = detect_doc_type(text, file_path)
    trial     = detect_trial_level(text, m_case_no.group(0) if m_case_no else "")
    statutes  = list(dict.fromkeys(_extract_statutes(text)))[:50]
    return {
        "case_number": m_case_no.group(0) if m_case_no else "",
        "court": court.strip() if court else "",
        "judgment_date": detect_judgment_date(text),
        "doc_type": doc_type,
        "trial_level": trial,
        "statutes": statutes
    }

# ---------- 深度信息抽取 ----------

# ===== 深度信息抽取：最优实现（可直接粘贴） =====
# 仅用标准库；不改你现有切块逻辑。主入口：deep_extract_from_chunks(chunks)
# 若需要把 doc 级结果挂到 chunk，可调用 attach_doc_info(chunks, doc_info, mode="first"/"all")

_CN_DIGITS = {"零":0, "〇":0, "○":0, "Ｏ":0, "O":0, "一":1, "二":2, "两":2, "三":3, "四":4, "五":5, "六":6, "七":7, "八":8, "九":9}
_CN_UNITS = {"十":10, "百":100, "千":1000, "万":10000, "亿":100000000}


def _cn_simple_num_to_int(s: str) -> Optional[int]:
    """
    将中文数字（含混合阿拉伯数字）转为 int。
    支持：十/二十/二十五/一万六千/三千二百一十/12/12万/1万6千 等常见写法。
    """

    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        return int(s)

    total, unit, num = 0, 1, 0
    i = 0
    s = ''.join({'０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9'}.get(ch, ch) for ch in s)

    for big_unit, big_val in [("亿", 100000000), ("万", 10000)]:
        if big_unit in s:
            parts = s.split(big_unit)
            left = _cn_simple_num_to_int(parts[0])
            right = _cn_simple_num_to_int(parts[1]) if len(parts) > 1 and parts[1] else 0
            if left is None:
                left = 1
            return left * big_val + (right or 0)

    tmp = 0
    has_unit = False
    for ch in s:
        if ch in _CN_DIGITS:
            tmp = tmp * 10 + _CN_DIGITS[ch]
        elif ch.isdigit():
            tmp = tmp * 10 + int(ch)
        elif ch in _CN_UNITS:
            val = _CN_UNITS[ch]
            has_unit = True
            if tmp == 0:
                tmp = 1
            total += tmp * val
            tmp = 0
        else:
            pass
    total += tmp
    if has_unit and total == 0:
        return None
    return total if total != 0 else None


def _parse_term_to_months(term: str) -> Optional[int]:
    """
    刑期描述 -> 月数。
    识别：X年Y个月 / X年 / Y个月 / 数字&中文混合。无期/死刑 -> None。
    """

    if not term:
        return None
    if "无期" in term or "死刑" in term:
        return None
    years = 0
    months = 0
    m_year = re.search(r"([〇零一二两三四五六七八九十百千\d]+)\s*年", term)
    m_mon = re.search(r"([〇零一二两三四五六七八九十百千\d]+)\s*个?\s*月", term)
    if m_year:
        years = _cn_simple_num_to_int(m_year.group(1)) or 0
    if m_mon:
        months = _cn_simple_num_to_int(m_mon.group(1)) or 0
    if not m_year and not m_mon:
        m2 = re.search(r"([〇零一二两三四五六七八九十百千\d]+)\s*月", term)
        if m2:
            months = _cn_simple_num_to_int(m2.group(1)) or 0
    total = years * 12 + months
    return total if total > 0 else None


def _normalize_date_cn(s: str) -> Optional[str]:
    """
    将“YYYY年MM月DD日 / 二〇二五年九月二十九日”等转为 ISO (YYYY-MM-DD)。
    不做闰月学术判断，仅做数值容错。
    """

    if not s:
        return None
    s = s.strip()
    cn_map = {"〇":"0","零":"0","一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9"}
    s = ''.join(cn_map.get(ch, ch) for ch in s)
    m = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", s)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_section_by_names(chunks: List[Dict[str, Any]], names: List[str]) -> str:
    """按 section 名称集合合并文本（适配你当前的中文分段命名）。"""

    buf = []
    name_set = set(names)
    for c in chunks:
        if str(c.get("section", "")).strip() in name_set:
            t = c.get("text") or ""
            buf.append(t)
    return "\n".join(buf).strip()


def _concat_fulltext(chunks: List[Dict[str, Any]]) -> str:
    """按原顺序拼接全文 text。"""

    return "\n".join([c.get("text", "") for c in sorted(chunks, key=lambda x: x.get("chunk_index", 0))])


def _extract_procuratorate(intro: str) -> Optional[str]:
    m = re.search(r"公诉机关(?:[:：]\s*)?([^\n。；]+?检察院)", intro)
    return m.group(1).strip() if m else None


def _extract_prosecutors(intro: str) -> List[str]:
    m = re.search(r"检察员(?:[:：]?\s*)([\u4e00-\u9fa5·、，和以及 ]+?)(?:出庭|到庭|出席|参加)", intro)
    if not m:
        return []
    raw = m.group(1).strip()
    names = [x for x in re.split(r"[、，和以及\s]+", raw) if x]
    return [n.replace("检察员", "").strip() for n in names if n]


def _extract_lawyers(intro: str) -> List[Dict[str, str]]:
    res = []
    pattern = re.compile(
        r"(?:指定)?(?:辩护人|辩护律师|委托诉讼代理人|委托代理人|诉讼代理人)\s*[:：]\s*([^\n。；]+)"
    )
    for m in pattern.finditer(intro):
        seg = m.group(1).strip()
        if not seg:
            continue
        parts = [p.strip() for p in re.split(r"[，、]", seg) if p.strip()]
        if not parts:
            continue
        name = parts[0].strip()
        if name.endswith("律师"):
            name = name[:-2].strip()
        firm = None
        for p in parts[1:]:
            clean = p.strip()
            if clean.endswith("律师"):
                clean = clean[:-2].strip()
            if "律师事务所" in clean or "法律援助" in clean or clean.endswith("律师"):
                firm = clean
                break
        if name:
            res.append({"name": name, "firm": firm})
    uniq = []
    seen = set()
    for x in res:
        k = (x.get("name"), x.get("firm"))
        if k not in seen:
            uniq.append(x)
            seen.add(k)
    return uniq


_PARTY_ROLE_KEYWORDS = [
    "被告人", "被告", "原告", "被上诉人", "上诉人", "申请人", "被申请人",
    "再审申请人", "申请执行人", "被执行人", "第三人", "申诉人", "被申诉人",
    "赔偿请求人"
]

_PARTY_ROLE_RE = re.compile(
    rf"^(?P<role>{'|'.join(re.escape(x) for x in _PARTY_ROLE_KEYWORDS)})"
    r"(?:（[^）]{0,40}）)?\s*[:：]?\s*(?P<body>.+)$"
)

_NAME_HEAD_RE = re.compile(r"^[\u4e00-\u9fa5·]{1,4}(?:[、和及与][\u4e00-\u9fa5·]{1,4})*(?=[，,；;、。\s（）()男女]|$)")
_NAME_SPLIT_RE = re.compile(r"[、和及与]")
_NAME_TAIL_ALLOWED = set("，,；;、。 \t（）()男女")


def _defendant_has_info(item: Dict[str, Any]) -> bool:
    return any([
        item.get("gender"),
        item.get("dob"),
        item.get("address"),
        item.get("prior_convictions"),
        item.get("detention"),
    ])


def _merge_defendant_info(target: Dict[str, Any], source: Dict[str, Any]) -> bool:
    changed = False
    for field in ["aka", "gender", "dob", "address"]:
        if not target.get(field) and source.get(field):
            target[field] = source[field]
            changed = True

    if source.get("prior_convictions"):
        existing = target.get("prior_convictions") or []
        if not isinstance(existing, list):
            existing = []
        for record in source["prior_convictions"]:
            if record not in existing:
                existing.append(record)
                changed = True
        target["prior_convictions"] = existing or None

    if source.get("detention"):
        existing_det = target.get("detention") or {}
        if not isinstance(existing_det, dict):
            existing_det = {}
        merged = dict(existing_det)
        for k, v in source["detention"].items():
            if v and k not in merged:
                merged[k] = v
                changed = True
        target["detention"] = merged or None

    return changed


def _split_name_field(name_field: str) -> List[str]:
    if not name_field:
        return []
    cleaned = re.sub(r"等人?$", "", name_field.strip())
    if not cleaned:
        return []
    return [n.strip() for n in _NAME_SPLIT_RE.split(cleaned) if n.strip()]


def _extract_defendants(intro: str) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []
    for seg in re.split(r"[。；\n]", intro):
        seg = seg.strip()
        if not seg:
            continue

        m_role = _PARTY_ROLE_RE.match(seg)
        names: List[str] = []
        role: Optional[str] = None
        if m_role:
            role = m_role.group("role")
            body = m_role.group("body").strip()
            m_names = _NAME_HEAD_RE.match(body)
            if m_names:
                names = _split_name_field(m_names.group(0))
            else:
                first = re.split(r"[，,；;]", body, 1)[0].strip()
                alt = _NAME_HEAD_RE.match(first)
                if alt:
                    names = _split_name_field(alt.group(0))
        else:
            stripped = seg.lstrip()
            if stripped.startswith("被告人"):
                role = "被告人"
                found = []
                for m in re.finditer(r"被告人\s*([\u4e00-\u9fa5·]{1,4}(?:[、和及与][\u4e00-\u9fa5·]{1,4})*)", stripped):
                    tail = stripped[m.end():m.end() + 1]
                    if tail and tail not in _NAME_TAIL_ALLOWED:
                        continue
                    found.extend(_split_name_field(m.group(1)))
                names = found

        if not names:
            continue

        if not role:
            role = "被告人"

        for name in names:
            gender = (re.search(r"，\s*(男|女)\s*，", seg) or
                      re.search(r"(男|女)[，、]", seg) or
                      re.search(r"(男|女)$", seg))
            gender_val = gender.group(1) if gender else None
            dob = (re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)\s*出生", seg) or
                   re.search(r"生于\s*(\d{4}年\d{1,2}月\d{1,2}日)", seg))
            dob_val = _normalize_date_cn(dob.group(1)) if dob else None
            addr = None
            madd = re.search(r"(?:住址|户籍所在地|户籍地|住所地|所在地|住所|住)[:：]?([^，。；]+)", seg)
            if madd:
                addr = madd.group(1).strip()
            priors = []
            for pm in re.finditer(r"(因?犯[^\n。；]*?于\d{4}年\d{1,2}月\d{1,2}日[^\n。；]*?判处[^\n。；]*?(?:刑)?[^\n。；]*?)", seg):
                priors.append(pm.group(1).strip())
            detention: Dict[str, Any] = {}
            for dm in re.finditer(r"于(\d{4}年\d{1,2}月\d{1,2}日)[^。；]*?(拘留|逮捕|羁押)", seg):
                iso = _normalize_date_cn(dm.group(1))
                kind = dm.group(2)
                if kind == "拘留":
                    detention["detained"] = iso
                elif kind == "逮捕":
                    detention["arrested"] = iso
                else:
                    detention["custody"] = iso

            candidate = {
                "name_masked": name,
                "aka": None,
                "gender": gender_val,
                "dob": dob_val,
                "address": addr,
                "prior_convictions": priors or None,
                "detention": detention or None,
                "role": role,
                "disposition": {
                    "offense": None,
                    "imprisonment_months": None,
                    "imprisonment_desc": None,
                    "fine_amount": None,
                    "confiscation_amount": None,
                    "detention_offset": False,
                    "term_start": None,
                    "term_end": None,
                    "probation": False
                },
                "factors": {
                    "self_surrender": False,
                    "plead_guilty": False,
                    "accessory": False,
                    "recidivist": False,
                    "confession": False
                }
            }

            existing = None
            for item in res:
                if item.get("name_masked") == name and item.get("role") == role:
                    conflict = False
                    for field in ["gender", "dob", "address"]:
                        if item.get(field) and candidate.get(field) and item.get(field) != candidate.get(field):
                            conflict = True
                            break
                    if not conflict:
                        existing = item
                        break

            if existing:
                merged = _merge_defendant_info(existing, candidate)
                if not merged and not _defendant_has_info(candidate):
                    continue
            else:
                res.append(candidate)

    uniq = []
    seen = set()
    for item in res:
        key = (
            item.get("name_masked"),
            item.get("role"),
            item.get("dob"),
            item.get("address")
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def _extract_verdict_blocks(text: str) -> str:
    m = re.search(r"(判决如下|判决结果)\s*[:：]?", text)
    if not m:
        return ""
    return text[m.end():].strip()


def _parse_disposition(verdict_text: str, defendants: List[Dict[str, Any]]) -> None:
    if not verdict_text:
        return
    segs = []
    lines = [ln.strip() for ln in verdict_text.splitlines() if ln.strip()]
    cur = ""
    for ln in lines:
        if re.match(r"^[一二三四五六七八九十]+、", ln):
            if cur:
                segs.append(cur)
                cur = ln
            else:
                cur = ln
        else:
            cur += (" " + ln)
    if cur:
        segs.append(cur)
    if not segs:
        segs = [" ".join(lines)]
    for seg in segs:
        idxs = []
        for i, d in enumerate(defendants):
            nm = d.get("name_masked") or ""
            aka = d.get("aka") or ""
            if nm and nm in seg or (aka and aka in seg):
                idxs.append(i)
        if not idxs:
            if len(defendants) == 1:
                idxs = [0]
            else:
                continue
        off = None
        moff = re.search(r"犯?([^\s，。；、]*?罪)[，、]", seg)
        if moff:
            cand = moff.group(1).strip()
            if not any(x in cand for x in ["有期徒刑","拘役","管制","罚金","追缴"]):
                off = cand
        impr_desc = None
        impr_months = None
        mterm = re.search(r"(有期徒刑|拘役|管制)([〇零一二两三四五六七八九十百千\d年月日天\s]*)", seg)
        if mterm:
            impr_desc = mterm.group(2).strip().rstrip("，、；。")
            impr_months = _parse_term_to_months(impr_desc)
        else:
            if "无期徒刑" in seg:
                impr_desc = "无期徒刑"
            elif "死刑缓期" in seg:
                impr_desc = "死刑缓期二年执行"
            elif "死刑" in seg:
                impr_desc = "死刑"
        probation = True if re.search(r"(宣告)?缓刑", seg) else False
        fine = None
        mf = re.search(r"罚金(?:人民币)?\s*([〇零一二两三四五六七八九十百千万亿\d]+)\s*元", seg)
        if mf:
            fine = _cn_simple_num_to_int(mf.group(1))
        conf = None
        mc = re.search(r"(?:违法所得)?(?:人民币)?\s*([〇零一二两三四五六七八九十百千万亿\d]+)\s*元[^，。；]*?(?:追缴|没收)", seg)
        if mc:
            conf = _cn_simple_num_to_int(mc.group(1))
        offset = True if "折抵刑期" in seg else False
        mrange = re.search(r"从(\d{4}年\d{1,2}月\d{1,2}日)起至(\d{4}年\d{1,2}月\d{1,2}日)止", seg)
        term_start = _normalize_date_cn(mrange.group(1)) if mrange else None
        term_end = _normalize_date_cn(mrange.group(2)) if mrange else None
        for i in idxs:
            d = defendants[i]
            disp = d["disposition"]
            disp.update({
                "offense": off or disp.get("offense"),
                "imprisonment_desc": impr_desc or disp.get("imprisonment_desc"),
                "imprisonment_months": impr_months if impr_months is not None else disp.get("imprisonment_months"),
                "fine_amount": fine if fine is not None else disp.get("fine_amount"),
                "confiscation_amount": conf if conf is not None else disp.get("confiscation_amount"),
                "detention_offset": offset or disp.get("detention_offset", False),
                "term_start": term_start or disp.get("term_start"),
                "term_end": term_end or disp.get("term_end"),
                "probation": probation or disp.get("probation", False)
            })


def _fill_factors(opinion: str, defendants: List[Dict[str, Any]]) -> None:
    if not opinion:
        return
    sents = [s.strip() for s in re.split(r"[。！？]", opinion) if s.strip()]
    keys = {
        "self_surrender": "自首",
        "plead_guilty": "认罪认罚",
        "accessory": "从犯",
        "recidivist": "累犯",
        "confession": "坦白"
    }
    for sent in sents:
        idxs = []
        for i, d in enumerate(defendants):
            nm = d.get("name_masked") or ""
            aka = d.get("aka") or ""
            if (nm and nm in sent) or (aka and aka in sent):
                idxs.append(i)
        if not idxs:
            if len(defendants) == 1 or re.search(r"(两|各|诸)被告人.*(均|皆)", sent):
                idxs = list(range(len(defendants)))
        for k, word in keys.items():
            if word in sent:
                neg = bool(re.search(r"(不(构成|属于)|未|没有|不予采信).{0,6}" + word, sent))
                for i in idxs:
                    defendants[i]["factors"][k] = (False if neg else True)


def _extract_dispute_focus(opinion: str, defense: str) -> List[str]:
    cand = []
    pool = "。".join([opinion or "", defense or ""])
    for s in [x.strip() for x in pool.split("。") if x.strip()]:
        if any(k in s for k in ["焦点","争议","辩护人", "是否", "不予采信"]):
            if ("自首" in s) or ("辩护" in s) or ("争议" in s) or ("焦点" in s):
                cand.append(s + "。")
    out, seen = [], set()
    for s in cand:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out[:5]


def _opinion_summary(opinion: str, topk: int = 3) -> List[str]:
    if not opinion:
        return []
    sents = [s.strip() for s in re.split(r"[。！？]", opinion) if s.strip()]
    if not sents:
        return []
    kw = {
        "本院认为": 3, "依法": 2, "从重": 1, "从轻": 1, "减轻": 1,
        "构成": 1, "证据": 1, "事实": 1, "情节": 1, "危害": 1,
        "认定": 1, "系从犯": 2, "认罪认罚": 2, "自首": 2, "坦白": 1
    }
    scores = []
    for s in sents:
        sc = sum(v for k, v in kw.items() if k in s)
        scores.append((sc, s))
    scores.sort(key=lambda x: (-x[0], len(x[1])))
    picked = [scores[0][1]]
    for sc, s in scores[1:]:
        if len(picked) >= topk:
            break
        if s not in picked:
            picked.append(s)
    return [p + "。" for p in picked]


def _fill_aka(fulltext: str, defendants: List[Dict[str, Any]]) -> None:
    for d in defendants:
        masked = d.get("name_masked")
        if not masked or len(masked) < 2:
            continue
        surname = masked[0]
        m = re.search(rf"{surname}[\u4e00-\u9fa5]{{1,2}}", fulltext)
        if m and m.group(0) != masked:
            d["aka"] = m.group(0)


def deep_extract_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    intro = _extract_section_by_names(chunks, ["开头","案号行","当事人信息","审理经过"])
    defense = _extract_section_by_names(chunks, ["被告人辩解与辩护意见","辩护意见"])
    opinion = _extract_section_by_names(chunks, ["理由","本院认为"])
    tail = _extract_section_by_names(chunks, ["主文","结果","判决结果","尾部"])
    full = _concat_fulltext(chunks)
    proc = _extract_procuratorate(intro)
    pros = _extract_prosecutors(intro)
    lawyers = _extract_lawyers(intro)
    defendants = _extract_defendants(intro)
    _fill_aka(full, defendants)
    verdict_zone = _extract_verdict_blocks(opinion + "\n" + tail)
    _parse_disposition(verdict_zone, defendants)
    _fill_factors(opinion, defendants)
    focus = _extract_dispute_focus(opinion, defense)
    op_sum = _opinion_summary(opinion, topk=3)
    return {
        "procuratorate": proc,
        "prosecutors": pros,
        "defendants": defendants,
        "lawyers": lawyers,
        "dispute_focus": focus,
        "opinion_summary": op_sum
    }


def attach_doc_info(chunks: List[Dict[str, Any]], doc_info: Dict[str, Any], mode: str = "first") -> List[Dict[str, Any]]:
    if not chunks:
        return chunks
    if mode == "first":
        chunks[0]["judgment_info"] = doc_info
        return chunks
    for c in chunks:
        c["judgment_info"] = doc_info
    return chunks
# ===== 以上为新增/替换部分 =====
# ---------- 分段（锚点+兜底） ----------
GEN_STRONG = [
    ("标题", r"^(?:刑事|民事|行政|执行|国家赔偿)?(?:判决书|裁定书|决定书|调解书|赔偿决定书|赔偿监督审查决定书)$"),
    ("案号行", r"^\s*[（(]\d{4}[）)][^号\n]{0,40}号\s*$"),
    ("主文",   r"^(裁判主文|判决如下|裁定如下|决定如下)\s*$"),
    ("依据",   r"^(?:依照|根据)[^。\n]{0,80}$"),
    ("理由",   r"^(?:本院(?:经审理|经审查)?认为|法院认为|合议庭认为|本院意见|判决理由)\s*$"),
    ("诉辩主张", r"^(?:事实[与和]理由)(?:[:：][^\n]{0,80})?$"),
    ("查明",   r"^(?:经审理查明|本院查明|案件基本事实|查明事实)\s*$"),
    ("尾部",   r"^(?:审判长|审判员|人民陪审员|书记员|本判决为终审判决|本裁定为终审裁定)"),
    ("一审判决", r"^(?:原审|一审)(?:法院)?(?:判决|裁定|决定)[^。\n]{0,10}[:：]?$|^(?:原审|一审)[^。\n]{0,40}(?:判决如下|裁定如下|决定如下)"),
    ("一审查明", r"^(?:一审法院|原审法院)(?:认定事实|查明|认为)[:：]?$"),
    ("一审认为", r"^(?:一审法院认为|原审法院认为)[:：]?$"),
    ("二审争议焦点", r"^(?:本案)?二审[^\n]{0,6}争议焦点[:：]?$"),
    ("争议焦点",     r"^(?:本案)?争议焦点[:：]?$"),
]

GEN_WEAK = [
    ("审理经过", r"(?:本院(?:于.*?受理后|受理后)|依法组成合议庭|公开开庭审理|审理经过|本案现已审理终结)"),
    ("原审情况", r"(?:原审法院|一审法院).{0,12}作出[（(]\d{4}[）)].{0,80}(?:判决|裁定)"),
]

SYS_STRONG = {
    "民事": [
        ("诉讼请求", r"^(?:诉讼请求|上诉请求|反诉请求)[:：]?$"),
        ("答辩意见", r"^(?:答辩|辩称|抗辩|异议)[^。\n]{0,8}[:：]?$"),
        ("证据",     r"^(?:证据(?:清单|目录)?|证据与理由|证据采信意见)[:：]?$"),
    ],
    "刑事": [
        ("起诉书指控", r"^(?:起诉书指控|公诉机关指控|检察机关指控)[:：]?$"),
        ("被告人辩解与辩护意见", r"^(?:被告人(?:供述与)?辩解|辩护人意见|辩护意见)[:：]?$"),
        ("证据", r"^(?:证人证言|书证|鉴定意见|勘验检查笔录|辨认笔录|电子数据|证据(?:综合)?分析)[:：]?$"),
    ],
    "行政": [
        ("被诉行政行为", r"^(?:被诉行政行为|被诉具体行政行为|行政行为概况)[:：]?$"),
        ("复议决定",     r"^(?:行政复议决定|复议情况|复议结论)[:：]?$"),
        ("诉辩主张",     r"^(?:上诉人|原告|被告|被上诉人|第三人).{0,6}(?:诉称|主张|辩称|意见)[:：]?$"),
        ("证据",         r"^(?:证据|证据与采信|证据证明力评判)[:：]?$"),
    ],
    "国家赔偿": [
        ("赔偿请求", r"^(?:赔偿请求|申请赔偿事项)[:：]?$"),
        ("赔偿决定依据", r"^(?:赔偿数额计算|赔偿项目|赔偿决定理由)[:：]?$"),
    ],
    "执行": [
        ("执行当事人", r"^(?:申请执行人|被执行人|案外人)[：:]"),
        ("执行经过",   r"^(?:执行经过|执行查明|执行情况)[:：]?$"),
        ("执行异议",   r"^(?:执行异议|案外人执行异议|复议请求)[:：]?$"),
    ],
}

PARTY_ROLE_PATTERN = (
    "上诉人|被上诉人|原告|被告|第三人|申请人|被申请人|赔偿请求人|被申诉人|申诉人|"
    "被执行人|申请执行人|案外人|法定代表人|代理人|委托(?:诉讼)?代理人|住所地|住址|统一社会信用代码|公民身份号码"
)

PARTY_HEAD_PATTERN = rf"^(?:{PARTY_ROLE_PATTERN})(?:[（(][^：:（）()]{{0,30}}[）)])*\s*[：:]"

PARTY_HEAD = re.compile(PARTY_HEAD_PATTERN)

PARTY_HEAD_PREFIX = re.compile(
    rf"(?:{PARTY_ROLE_PATTERN})(?:[（(][^：:（）()]{{0,30}}[）)])*"
)

PARTY_HEAD_BAD_PREFIXES = (
    "诉讼请求",
    "上诉请求",
    "反诉请求",
    "申请",
    "请求",
    "诉称",
    "主张",
    "答辩",
    "辩称",
    "辩护意见",
    "辩护人意见",
    "意见",
    "陈述",
    "情况",
    "事实与理由",
    "事实和理由",
)


def _is_party_head_line(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False
    if PARTY_HEAD.match(line):
        return True

    prefix = PARTY_HEAD_PREFIX.match(line)
    if not prefix:
        return False

    rest = line[prefix.end():].lstrip("：:，,。；;、 　\t")
    if not rest:
        return True

    for bad in PARTY_HEAD_BAD_PREFIXES:
        if rest.startswith(bad):
            return False

    return True


def _find_party_head_index(text: str) -> int:
    if not text:
        return -1

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and _is_party_head_line(stripped):
            start = line.find(stripped)
            if start >= 0:
                return offset + start
        for match in PARTY_HEAD_PREFIX.finditer(line):
            candidate = line[match.start():].lstrip()
            if not candidate:
                continue
            if _is_party_head_line(candidate):
                delta = len(line[match.start():]) - len(candidate)
                return offset + match.start() + delta
        offset += len(line)
    return -1

PARTY_BLOCK_STOP_TOKENS = [
    "本院于", "受理后", "依法组成合议庭", "公开开庭审理", "本案现已审理终结", "审理经过", "原审法院", "一审法院",
    "诉讼请求", "上诉请求", "反诉请求", "答辩", "辩称", "抗辩", "意见", "诉辩主张", "赔偿请求",
    "被诉行政行为", "行政处罚决定", "行政复议决定", "复议决定", "复议情况",
    "起诉书指控", "公诉机关指控", "检察机关指控", "被告人辩解", "辩护意见",
    "经审理查明", "本院查明", "案件基本事实", "证据", "证人证言", "鉴定意见",
    "本院认为", "法院认为", "合议庭认为", "判决理由",
    "裁判主文", "判决如下", "裁定如下", "决定如下", "依照", "根据",
    "事实和理由", "事实与理由",
]

_PARTY_EMBEDDED_HINTS = [
    ("审理经过", re.compile(r"(?:本院于|受理后|依法组成合议庭|公开开庭审理|本案现已审理终结|审理经过)")),
    ("原审情况", re.compile(r"(?:原审法院|一审法院).{0,30}(?:判决|裁定|决定)")),
    ("诉讼请求", re.compile(r"(?:诉讼请求|上诉请求|反诉请求)[:：]?")),
    ("答辩意见", re.compile(r"(?:答辩|辩称|抗辩|意见)[:：]")),
    ("诉辩主张", re.compile(r"事实[与和]理由")),
    ("赔偿请求", re.compile(r"(?:赔偿请求|申请赔偿事项)")),
    ("赔偿决定依据", re.compile(r"(?:赔偿数额计算|赔偿项目|赔偿决定理由)")),
    ("被诉行政行为", re.compile(r"被诉(?:行政)?行为")),
    ("复议决定", re.compile(r"复议(?:决定|情况|结论)")),
    ("起诉书指控", re.compile(r"(?:起诉书指控|公诉机关指控|检察机关指控)")),
    ("被告人辩解与辩护意见", re.compile(r"(?:被告人(?:供述与)?辩解|辩护意见)")),
    ("证据", re.compile(r"(?:证据|证人证言|鉴定意见|勘验检查笔录|辨认笔录|电子数据)")),
    ("查明", re.compile(r"(?:经审理查明|本院查明|案件基本事实|查明事实)")),
    ("一审查明", re.compile(r"(?:一审法院|原审法院)(?:认定事实|查明)")),
    ("一审认", re.compile(r"(?:一审法院认为|原审法院认为)")),
    ("理由", re.compile(r"(?:本院(?:经审理|经审查)?认为|法院认为|合议庭认为|本院意见|判决理由)")),
    ("主文", re.compile(r"(?:裁判主文|判决如下|裁定如下|决定如下)")),
    ("依据", re.compile(r"^(?:依照|根据)")),
]

PARTY_BLOCK_STOP = set(["审理经过","原审情况","查明","一审查明","一审认为","理由","依据","主文"])

WEAK_SECTION = {"审理经过","原审情况"}

SECTION_PATTERNS = GEN_STRONG + GEN_WEAK

TITLE_TIGHT_RE = re.compile(GEN_STRONG[0][1])


def _select_sys_patterns(system: str):
    pats = []
    for nm, rx in SYS_STRONG.get(system or "", []):
        pats.append((nm, re.compile(rx)))
    return pats


def _match_section_name(raw: str, system: str = ""):
    s = raw.strip()
    if not s:
        return None
    tight = re.sub(r"[\s\u3000]+", "", s)
    if TITLE_TIGHT_RE.match(tight):
        return "标题"

    for nm, pat in _select_sys_patterns(system):
        if pat.match(s):
            return nm

    for name, pat in [(n, re.compile(p)) for n, p in GEN_STRONG if n != "标题"]:
        if pat.match(s):
            return name

    for name, pat in [(n, re.compile(p)) for n, p in GEN_WEAK]:
        m = pat.search(s)
        if m and s.find(m.group(0)) <= 12:
            return name

    return None


def _should_stop_party(line: str) -> bool:
    if _match_section_name(line):
        return True
    plain = re.sub(r"\s+", "", line)
    return any(tok in plain for tok in PARTY_BLOCK_STOP_TOKENS)


def _guess_system_from_text(text: str) -> str:
    head = text[:1000]
    if "执行" in head:
        return "执行"
    if "国家赔偿" in head or "赔偿决定书" in head:
        return "国家赔偿"
    if "行政" in head:
        return "行政"
    if "刑事" in head:
        return "刑事"
    return "民事"


def _split_embedded_sections(section_name: str, text: str):
    text = (text or "").strip()
    if not text:
        return None

    lines = text.splitlines()
    if len(lines) <= 1:
        return None

    sys_hint = _guess_system_from_text(text)
    markers = []

    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue

        candidate = _match_section_name(stripped, system=sys_hint)
        if candidate:
            if idx == 0 and candidate == section_name:
                continue
            if not markers or markers[-1][0] != idx:
                markers.append((idx, candidate))
            continue

        for cand, rx in _PARTY_EMBEDDED_HINTS:
            if rx.search(stripped):
                if idx == 0 and cand == section_name:
                    continue
                if not markers or markers[-1][0] != idx:
                    markers.append((idx, cand))
                break

    if not markers:
        return None

    pieces = []
    block_start = 0
    current_name = section_name

    for ln, new_name in markers:
        if ln < block_start:
            continue
        prev_text = "\n".join(lines[block_start:ln]).strip()
        if prev_text:
            pieces.append({"name": current_name, "text": prev_text})
        current_name = new_name
        block_start = ln

    rest_text = "\n".join(lines[block_start:]).strip()
    if rest_text:
        pieces.append({"name": current_name, "text": rest_text})

    if not pieces:
        return None
    if len(pieces) == 1 and pieces[0]["name"] == section_name and pieces[0]["text"] == text:
        return None
    return pieces


def split_sections(text: str, system_hint: str = ""):
    sys = (system_hint or "").strip()
    if not sys:
        head = text[:1000]
        if "执行" in head: sys = "执行"
        elif "国家赔偿" in head or "赔偿决定书" in head: sys = "国家赔偿"
        elif "行政" in head: sys = "行政"
        elif "刑事" in head: sys = "刑事"
        else: sys = "民事"

    lines = text.splitlines()
    markers = []
    seen_case_no = False

    for i, raw in enumerate(lines):
        name = _match_section_name(raw, system=sys)

        if name == "案号行":
            if seen_case_no or i > 30:
                name = None
            else:
                seen_case_no = True

        if name == "主文" and re.search(r"(原审|一审)[^。\n]{0,12}(判决|裁定|决定)", raw):
            name = "一审判决"

        if name:
            markers.append((i, name))

    existing_party_starts = {ln for ln, nm in markers if nm == "当事人信息"}
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if _is_party_head_line(line):
            start = i
            j = i + 1
            while j < n:
                cur = lines[j].strip()
                if _is_party_head_line(cur):
                    j += 1
                    continue
                if not cur:
                    if j + 1 < n and _is_party_head_line(lines[j+1].strip()):
                        j += 1
                        continue
                    break
                if _should_stop_party(cur):
                    break
                j += 1
            if start not in existing_party_starts:
                markers.append((start, "当事人信息"))
                existing_party_starts.add(start)
            i = j
            continue
        i += 1

    if any(nm == "案号行" for _, nm in markers) and not any(nm == "当事人信息" for _, nm in markers):
        start = next(ln for ln, nm in markers if nm == "案号行")
        lim = min(len(lines), start + 60)
        for j in range(start+1, lim):
            if _is_party_head_line(lines[j].strip()):
                markers.append((j, "当事人信息"))
                break

    if not markers:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [{"name": "正文", "text": p} for p in parts]

    markers = sorted(set(markers))

    sections = []
    for idx, (ln, name) in enumerate(markers):
        if idx == 0 and ln > 0:
            head_text = "\n".join(lines[:ln]).strip()
            if head_text:
                sections.append({"name": "开头", "text": head_text})
        end = markers[idx+1][0] if idx+1 < len(markers) else len(lines)
        seg = "\n".join(lines[ln:end]).strip()
        if seg:
            sections.append({"name": name, "text": seg})
        if idx == len(markers) - 1 and end < len(lines):
            tail = "\n".join(lines[end:]).strip()
            if tail:
                sections.append({"name": "结尾", "text": tail})

    return _post_split_fixup(sections)


def _post_split_fixup(sections):
    fixed = []

    def _extend(name: str, content: str):
        content = (content or "").strip()
        if not content:
            return
        if name in {"案号行", "当事人信息"}:
            parts = _split_embedded_sections(name, content)
            if parts:
                fixed.extend(_post_split_fixup(parts))
                return
        fixed.append({"name": name, "text": content})

    for sec in sections:
        nm = sec.get("name")
        tx = (sec.get("text") or "").strip()
        if not tx:
            continue

        if nm == "案号行":
            idx = _find_party_head_index(tx)
            if idx >= 0:
                head_part = tx[:idx].strip()
                rest = tx[idx:].strip()
                if head_part:
                    _extend("案号行", head_part)
                if rest:
                    m2 = re.search(r"(本院于|受理后|依法组成合议庭|公开开庭审理|本案现已审理终结|原审法院.{0,12}作出[（(]\d{4}[）)])", rest)
                    if m2:
                        party_part = rest[:m2.start()].strip()
                        trial_part = rest[m2.start():].strip()
                        if party_part:
                            _extend("当事人信息", party_part)
                        if trial_part:
                            _extend("审理经过", trial_part)
                        continue
                    _extend("当事人信息", rest)
                    continue

        if nm == "当事人信息":
            parts = _split_embedded_sections(nm, tx)
            if parts:
                fixed.extend(_post_split_fixup(parts))
                continue

        fixed.append({"name": nm, "text": tx})

    return fixed


def gentle_split_long_block(s: str, hard_limit=1600, target=900):
    s = s.strip()
    if len(s) <= hard_limit: return [s]
    parts = [p for p in re.split(
        r"(?=（\d+）|（[一二三四五六七八九十]+）|^\s*\d+、|^\s*[一二三四五六七八九十]+、)",
        s, flags=re.M) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    if not parts: parts = [s]
    merged, buf = [], ""
    def push():
        nonlocal buf
        if buf.strip(): merged.append(buf.strip()); buf=""
    for p in parts:
        if not buf: buf = p
        elif len(buf) + 1 + len(p) < target * 1.35:
            buf += "\n" + p
        else:
            push(); buf = p
    push()
    refined = []
    for seg in merged:
        if len(seg) <= hard_limit: refined.append(seg); continue
        clauses = re.split(r"(。|；)", seg)
        tmp = ""
        for i in range(0, len(clauses), 2):
            chunk = clauses[i] + (clauses[i+1] if i+1 < len(clauses) else "")
            if len(tmp) + len(chunk) < target * 1.3: tmp += chunk
            else:
                if tmp: refined.append(tmp); tmp = chunk
        if tmp: refined.append(tmp)
    return [x.strip() for x in refined if x.strip()]

SENT_SPLIT = re.compile(r"(?<=[。！？；])\s*(?![”』】）])")


def sentence_chunks(s: str, min_chars=700, max_chars=1200, overlap_sentences=2):
    s = s.strip()
    if not s:
        return []
    # 先切句
    sents = [x.strip() for x in SENT_SPLIT.split(s) if x.strip()]
    chunks, buf = [], []
    cur_len = 0
    for sent in sents:
        if cur_len + len(sent) <= max_chars or not buf:
            buf.append(sent)
            cur_len += len(sent)
        else:
            chunks.append("".join(buf).strip())
            # 句粒度重叠：带上最后 N 句
            tail = buf[-overlap_sentences:] if overlap_sentences > 0 else []
            buf = tail + [sent]
            cur_len = sum(len(x) for x in buf)
    if buf:
        chunks.append("".join(buf).strip())

    # 太短的块（非主文/依据）尝试合并相邻
    refined = []
    tmp = ""
    for c in chunks:
        if not tmp:
            tmp = c
            continue
        if len(tmp) < min_chars and len(tmp) + len(c) <= max_chars * 1.2:
            tmp = tmp + c
        else:
            refined.append(tmp)
            tmp = c
    if tmp:
        refined.append(tmp)
    return refined


def _chunk_one_section(sec_name: str, sec_text: str,
                       min_chars=700, max_chars=1200, overlap=120):
    # 主文/依据：严格独立
    if sec_name in ("主文", "依据"):
        return sentence_chunks(sec_text, 1, 10**9, overlap_sentences=0)

    # 其他：句子窗口 + 句粒度重叠
    return sentence_chunks(sec_text, min_chars, max_chars, overlap_sentences=2)


def aggregate_chunks(sections, min_chars=700, max_chars=1200, overlap=120):
    """
    改为严格“分段优先”：逐段切块、逐段产出。
    - 不跨 section 合并（因此不会再出现 `开头~当事人信息` 这类跨段标签）
    - section_span 直接就是该段名
    - 主文/依据天然独立
    """
    all_chunks = []
    for sec in sections:
        sec_name = sec["name"]
        sec_text = sec["text"]
        # 逐段切块
        pieces = _chunk_one_section(
            sec_name, sec_text,
            min_chars=min_chars, max_chars=max_chars, overlap=overlap
        )
        for txt in pieces:
            all_chunks.append({
                "text": txt,
                "section_span": sec_name,
                "section": sec_name
            })
    return all_chunks

# ---------- system/subtype 推断 ----------
SYSTEM_ALIASES = {
    "刑事":"刑事","民事":"民事","行政":"行政","执行":"执行","国家赔偿":"国家赔偿","国赔":"国家赔偿",
}

EXEC_SUBTYPE_RULES = [
    ("首次执行", ["首次执行","首次立案","立案执行裁定"]),
    ("恢复执行", ["恢复执行","继续执行","中止后恢复"]),
    ("终结/终止", ["终结执行","终止执行","结案终结","本次执行终结","本案执行终结","终止"]),
    ("执行异议", ["执行异议","案外人执行异议","异议之诉","异议"]),
    ("执行复议", ["执行复议","复议决定","复议裁定","复议"]),
    ("追加/变更被执行人", ["追加被执行人","变更被执行人","追加","变更被执行人"]),
    ("拍卖/变卖", ["拍卖","变卖","处置拍卖","网络司法拍卖"]),
    ("执行分配", ["分配","执行款分配","分配方案","分配裁定"]),
    ("限高", ["限制高消费","限高","限制消费"]),
    ("罚款/拘留", ["罚款","司法拘留","拘留","罚金"]),
    ("失信名单", ["失信被执行人","失信名单","纳入失信"]),
    ("财产执行", ["查封","冻结","扣划","划拨","搜查","控制财产","轮候查封","解除查封"]),
]

def infer_exec_subtype_from_filename(fname: str) -> str:
    name = os.path.splitext(fname)[0]
    for label, keys in EXEC_SUBTYPE_RULES:
        if any(kw in name for kw in keys):
            return label
    for kw in ["查封","冻结","扣划","搜查","控制","轮候"]:
        if kw in name: return "财产执行"
    return "其他执行"

def infer_case_system_and_subtype(root_dir, file_path):
    rel = os.path.relpath(file_path, root_dir)
    parts = rel.replace("\\","/").split("/")
    system = ""; subtype = ""
    if len(parts) >= 2:
        system = SYSTEM_ALIASES.get(parts[0].strip(), parts[0].strip())
        if system != "执行" and len(parts) >= 3:
            subtype = parts[1].strip()
    if system == "执行":
        subtype = infer_exec_subtype_from_filename(os.path.basename(file_path))
    return system, subtype

# ---------- 读取 TXT ----------
def read_text_from_txt(path: str) -> str:
    # 尝试 UTF-8 / GB18030
    for enc in ("utf-8", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    # 兜底二进制解码
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8","ignore")

# ---------- 编号 ----------
def make_doc_id(root_dir: str, file_path: str) -> str:
    try:
        st = os.stat(file_path)
        sig = f"{os.path.relpath(file_path, root_dir)}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        sig = os.path.relpath(file_path, root_dir)
    return "JUD-" + sha1(sig.encode("utf-8")).hexdigest()[:12]

def make_chunk_id(doc_id: str, span: str, idx: int) -> str:
    base = f"{doc_id}#{span}#{idx:03d}"
    return f"{base}-{sha1(base.encode('utf-8')).hexdigest()[:8]}"

# ---------- 单文件处理 ----------
def process_one_file(root_dir, out_dir, path, min_chars, max_chars, overlap):
    raw = read_text_from_txt(path)
    if not raw or not raw.strip():
        return {"file": path, "ok": False, "reason": "empty_or_unreadable"}

    text = normalize_text(raw)
    meta = extract_light_meta(text, path)
    system, subtype = infer_case_system_and_subtype(root_dir, path)

    if not system:
        if "刑事" in meta["doc_type"]: system = "刑事"
        elif "民事" in meta["doc_type"]: system = "民事"
        elif "行政" in meta["doc_type"]: system = "行政"
        elif "国家赔偿" in meta["doc_type"] or "赔偿" in meta["doc_type"]: system = "国家赔偿"
        elif "执行" in meta["doc_type"]: system = "执行"
        else: system = "民事"

    if system == "执行":
        subtype = infer_exec_subtype_from_filename(os.path.basename(path))
        meta["trial_level"] = "执行程序"

    sections = split_sections(text)
    chunks = aggregate_chunks(sections, min_chars, max_chars, overlap)
    for idx, c in enumerate(chunks):
        c["chunk_index"] = idx
        if "section" not in c:
            c["section"] = c.get("section_span")
    doc_info = deep_extract_from_chunks(chunks)
    chunks = attach_doc_info(chunks, doc_info, mode="all")

    rel_path = os.path.relpath(path, root_dir).replace("\\","/")
    out_path = os.path.join(out_dir, os.path.splitext(rel_path)[0] + ".jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    doc_id = make_doc_id(root_dir, path)
    with open(out_path, "w", encoding="utf-8") as w:
        for i, c in enumerate(chunks):
            rec = {
                "chunk_id": make_chunk_id(doc_id, c["section_span"], i),
                "doc_id": doc_id,
                "case_system": system,
                "case_subtype": subtype,
                "doc_type": meta["doc_type"],
                "trial_level": meta["trial_level"],
                "court": meta["court"],
                "case_number": meta["case_number"],
                "judgment_date": meta["judgment_date"],
                "statutes": meta["statutes"],
                "judgment_info": c.get("judgment_info", doc_info),
                "section": c.get("section", c["section_span"]),
                "chunk_index": i,
                "text": c["text"]
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"file": path, "ok": True, "chunks": len(chunks), "doc_id": doc_id, "out": out_path}

# ---------- 批处理 ----------
def list_txt_files(in_dir):
    files = []
    for root, _, names in os.walk(in_dir):
        for n in names:
            if n.startswith("~$"): continue
            if n.lower().endswith(".txt"):
                files.append(os.path.join(root, n))
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="输入根目录（如：/文书）")
    ap.add_argument("--out_dir", required=True, help="输出根目录（JSONL 将按相对路径生成）")
    ap.add_argument("--min_chars", type=int, default=700)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4, help="并行线程数")
    args = ap.parse_args()

    files = list_txt_files(args.in_dir)
    if not files:
        print("[WARN] 未找到 .txt 文件"); return
    os.makedirs(args.out_dir, exist_ok=True)

    total_ok = total_fail = 0
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one_file, args.in_dir, args.out_dir, f,
                              args.min_chars, args.max_chars, args.overlap) for f in files]
            for fu in as_completed(futs):
                r = fu.result()
                if r["ok"]:
                    total_ok += 1
                    print(f"[OK] {r['file']} → {r['out']} ({r['chunks']} 块)")
                else:
                    total_fail += 1
                    print(f"[FAIL] {r['file']} → {r.get('reason','unknown')}")
    else:
        for f in files:
            r = process_one_file(args.in_dir, args.out_dir, f,
                                 args.min_chars, args.max_chars, args.overlap)
            if r["ok"]:
                total_ok += 1
                print(f"[OK] {r['file']} → {r['out']} ({r['chunks']} 块)")
            else:
                total_fail += 1
                print(f"[FAIL] {r['file']} → {r.get('reason','unknown')}")

    print(f"\n[SUMMARY] 成功 {total_ok}，失败 {total_fail}，总计 {len(files)}")

if __name__ == "__main__":
    main()