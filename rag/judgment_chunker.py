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
from hashlib import sha1
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
RE_COURT   = re.compile(r"[\u4e00-\u9fa5]{2,30}人民法院")
RE_DATE_ARABIC = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
RE_DATE_CHINESE = re.compile(r"([〇零○ＯO一二三四五六七八九十百千两]{3,})年([〇零○ＯO一二三四五六七八九十两]{1,3})月([〇零○ＯO一二三四五六七八九十两]{1,3})日")
RE_STATUTE = re.compile(r"《[^》]{1,30}》第?[一二三四五六七八九十百千0-9]+条")

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

def detect_trial_level(text: str) -> str:
    """
    返回统一规范的审级/程序标签（trial_level）：
      - 赔偿委员会申诉｜赔偿委员会决定
      - 复议决定｜赔偿决定（行政赔偿线）
      - 再审审查｜再审｜二审｜一审｜死刑复核
      - 执行程序
    匹配顺序按“越具体优先”。
    """
    head = text[:8000]

    patterns = [
        # —— 国家赔偿（司法赔偿线：赔偿委员会）——
        # 监督程序/申诉到上一级赔偿委员会
        (r"(委赔监|赔偿监督程序|赔偿监督审查)",                   "赔偿委员会申诉"),
        (r"赔偿委员会.{0,30}(申诉|监督程序|监督审查|申诉审查)",     "赔偿委员会申诉"),
        (r"(驳回|支持).{0,6}申诉",                                  "赔偿委员会申诉"),

        # 同级赔偿委员会作出的决定
        (r"赔偿委员会.{0,10}(决定书|决定)",                         "赔偿委员会决定"),
        (r"（\d{4}）[^，\n]*委赔[^号]*号",                           "赔偿委员会决定"),

        # —— 国家赔偿（行政赔偿线）——
        (r"(复议决定书|复议决定).{0,8}(国家赔偿|赔偿)",               "复议决定"),
        (r"(赔偿决定书|赔偿决定).{0,20}(行政机关|赔偿义务机关|国家赔偿)", "赔偿决定"),

        # —— 特殊程序 ——
        (r"死刑复核",                                               "死刑复核"),

        # —— 再审链路 ——
        (r"再审(申请)?审查",                                        "再审审查"),
        (r"(再审|重审).{0,6}(判决|裁定|决定)",                       "再审"),

        # —— 二审/一审：显式用语优先 ——
        (r"二审(判决|裁定|决定)?",                                  "二审"),
        (r"一审(判决|裁定|决定)?",                                  "一审"),

        # —— 二审/一审：从案号代码判断（行终/民终/刑终 ~ 二审；行初/民初/刑初 ~ 一审）——
        (r"(行终|民终|刑终)",                                       "二审"),
        (r"(行初|民初|刑初)",                                       "一审"),

        # —— 执行程序（不是审级，单列为程序标签）——
        (r"执行(裁定|决定|异议|复议|和解|分配|拍卖|变卖|终结|终止|恢复|追加|变更|限制|罚款|拘留)", "执行程序"),
        (r"\b执行\b",                                               "执行程序"),
    ]

    for pat, label in patterns:
        if re.search(pat, head):
            return label
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


def _detect_court(text: str) -> str:
    head = text[:4000]
    best = ""
    for raw in head.splitlines():
        if "人民法院" not in raw:
            continue
        candidates = RE_COURT.findall(raw)
        if not candidates:
            continue
        # 取最长匹配，通常是全称
        candidate = max(candidates, key=len)
        candidate = candidate.strip()
        if len(candidate) > len(best):
            best = candidate
    return best

def detect_judgment_date(text: str) -> str:
    matches = []
    length = len(text)

    for m in RE_DATE_ARABIC.finditer(text):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ctx = text[max(0, m.start()-20):min(length, m.end()+20)]
        has_keyword = bool(re.search(r"判决|裁定|决定|调解|赔偿|结案|审理终结", ctx))
        dist = length - m.start()
        score = (0 if has_keyword else 1000) + dist
        matches.append((score, year, month, day))

    for m in RE_DATE_CHINESE.finditer(text):
        year_raw, month_raw, day_raw = m.group(1), m.group(2), m.group(3)
        year = _parse_chinese_number(year_raw)
        month = _parse_chinese_number(month_raw)
        day = _parse_chinese_number(day_raw)
        if year < 0 or month <= 0 or day <= 0:
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

def extract_light_meta(text: str, file_path: str = ""):
    head = text[:4000]
    m_case_no = RE_CASE_NO.search(head)
    court = _detect_court(text)
    doc_type  = detect_doc_type(text, file_path)
    trial     = detect_trial_level(text)
    statutes  = list(dict.fromkeys([x.group(0) for x in RE_STATUTE.finditer(text)]))[:50]
    return {
        "case_number": m_case_no.group(0) if m_case_no else "",
        "court": court.strip() if court else "",
        "judgment_date": detect_judgment_date(text),
        "doc_type": doc_type,
        "trial_level": trial,
        "statutes": statutes
    }

# ---------- 分段（锚点+兜底） ----------
SECTION_PATTERNS = [
    ("标题", r"[\u4e00-\u9fa5A-Za-z0-9（）()〔〕【】\-\.]{2,30}(判决书|裁定书|决定书|调解书|赔偿决定书|赔偿监督审查决定书)$"),
    ("案号行", r"[（(]\d{4}[）)]\s*[\u4e00-\u9fa5A-Z0-9]{2,}\d+号$"),
    ("诉讼请求", r"(诉讼请求|上诉请求|抗诉请求)$"),
    ("答辩/抗辩", r"(答辩情况|抗辩意见|辩称|被告答辩|被上诉人答辩)$"),
    ("审理经过", r"(审理经过|案件受理|庭审情况|程序经过)$"),
    ("查明", r"(经审理查明|本院查明|查明事实|案件基本事实)$"),
    ("理由", r"(本院认为|法院认为|合议庭认为|二审法院认为)$"),
    ("依据", r"(依照|根据)$"),
    ("主文", r"(裁判主文|判决如下|裁定如下|决定如下)$"),
    ("费用", r"(案件受理费|诉讼费用|手续费|执行费用)$"),
    ("赔偿专段", r"(赔偿决定|赔偿范围|赔偿项目|赔偿数额)$"),
]
SEC_COMPILED = [(name, re.compile(pat)) for name, pat in SECTION_PATTERNS]

def _match_section_name(raw: str):
    stripped = raw.strip()
    if not stripped:
        return None
    collapsed = _clean_spaces(stripped)
    for name, pat in SEC_COMPILED:
        if pat.search(stripped) or pat.search(collapsed):
            return name
    if collapsed.endswith("经审理查明"):
        return "查明"
    if collapsed.endswith("经审理认为") or collapsed.endswith("合议庭认为"):
        return "理由"
    return None

def split_sections(text: str):
    lines = text.splitlines()
    markers = []
    for i, raw in enumerate(lines):
        name = _match_section_name(raw)
        if name:
            markers.append((i, name))
    if len(markers) < 2 or {name for _, name in markers} == {"案号行"}:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) <= 1:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                return []
            header, body = [], []
            for idx, line in enumerate(lines):
                header.append(line)
                collapsed = _clean_spaces(line)
                if any(token in collapsed for token in ["判决书","裁定书","决定书","调解书","赔偿决定书","赔偿监督审查决定书"]):
                    body = lines[idx+1:]
                    break
                if idx >= 2:
                    body = lines[idx+1:]
                    break
            else:
                body = []
            sections = []
            if header:
                sections.append({"name": "标题", "text": "\n".join(header).strip()})
            if body:
                sections.append({"name": "正文", "text": "\n".join(body).strip()})
            return sections if sections else [{"name": "正文", "text": text.strip()}]
        sections = []
        first = parts[0]
        if any(token in _clean_spaces(first) for token in ["判决书","裁定书","决定书","调解书","赔偿决定书","赔偿监督审查决定书"]):
            sections.append({"name": "标题", "text": first})
            for idx, p in enumerate(parts[1:], start=1):
                sections.append({"name": f"段落{idx}", "text": p})
        else:
            for idx, p in enumerate(parts, start=1):
                sections.append({"name": f"段落{idx}", "text": p})
        return sections
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
    return sections

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

def aggregate_chunks(sections, min_chars=700, max_chars=1200, overlap=120):
    expanded = []
    for sec in sections:
        for seg in gentle_split_long_block(sec["text"]):
            expanded.append({"name": sec["name"], "text": seg})

    chunks, buf, buf_len = [], [], 0
    def flush(allow_overlap=True):
        nonlocal buf, buf_len
        if not buf: return
        text = "\n".join([b["text"] for b in buf]).strip()
        names = [b["name"] for b in buf]
        chunks.append({
            "text": text,
            "section_span": names[0] + (("~" + names[-1]) if names[-1]!=names[0] else "")
        })
        if allow_overlap and overlap > 0 and text:
            tail = text[-overlap:]
            buf = [{"name": buf[-1]["name"], "text": tail}]
            buf_len = len(tail)
        else:
            buf, buf_len = [], 0

    for seg in expanded:
        seglen = len(seg["text"])
        if buf_len == 0:
            buf = [seg]; buf_len = seglen
            if buf_len >= max_chars: flush(True)
            continue
        if buf[-1]["name"] in ("主文","依据") and seg["name"] not in ("主文","依据"):
            flush(False)
        if buf_len + 1 + seglen <= max_chars:
            buf.append(seg); buf_len += 1 + seglen
        else:
            if buf_len >= min_chars:
                flush(True); buf = [seg]; buf_len = seglen
                if buf_len >= max_chars: flush(True)
            else:
                flush(True); buf = [seg]; buf_len = seglen
                if buf_len >= max_chars: flush(True)
    if buf_len > 0: flush(False)
    return chunks

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

    sections = split_sections(text)
    chunks = aggregate_chunks(sections, min_chars, max_chars, overlap)

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
                "section": c["section_span"],
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
