import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from docx import Document
from pypdf import PdfReader


LEGAL_MAP = {
    "buy_sell": "买卖合同",
    "lease": "租赁合同",
    "labor": "劳动合同法体系",
    "outsourcing": "承揽/委托/技术合同（按具体条款）",
    "nda": "保密协议（非典型合同，保密义务）",
    "software": "买卖/技术开发/技术许可（按内容）",
}

VALID_BUSINESS_TYPES = set(LEGAL_MAP.keys())

FULLWIDTH_TABLE = {
    ord("："): ":",
    ord("，"): ",",
    ord("；"): ";",
    ord("（"): "(",
    ord("）"): ")",
    ord("【"): "[",
    ord("】"): "]",
    ord("％"): "%",
    ord("．"): ".",
    ord("。"): "。",
}

for i in range(10):
    FULLWIDTH_TABLE[ord(chr(0xFF10 + i))] = str(i)

FULLWIDTH_MAP = str.maketrans(FULLWIDTH_TABLE)


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if suffix == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n".join(texts)
    raise ValueError(f"Unsupported file type: {path}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(FULLWIDTH_MAP)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def sliding_windows(text: str, window: int, overlap: int) -> List[str]:
    if len(text) <= window:
        return [text]
    segments = []
    start = 0
    step = max(1, window - overlap)
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + window)
        segments.append(text[start:end])
        if end >= text_len:
            break
        start += step
    return segments


class DeepseekClient:
    def __init__(self, api_base: str, model: str, api_key: str, qps: float, retries: int, timeout: int):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.min_interval = 1.0 / qps if qps > 0 else 0
        self.retries = max(1, retries)
        self.timeout = timeout
        self._last_call = 0.0

    def _respect_rate_limit(self):
        if self.min_interval <= 0:
            return
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def chat(self, prompt: str) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是中文合同解析器，只返回严格 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        backoffs = [1, 3, 7]
        last_err = None
        for attempt in range(self.retries):
            self._respect_rate_limit()
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                self._last_call = time.time()
                if resp.status_code >= 500:
                    last_err = RuntimeError(f"Server error {resp.status_code}")
                elif resp.status_code == 429:
                    last_err = RuntimeError("Rate limited")
                elif resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
            except (requests.RequestException, ValueError, RuntimeError) as exc:  # ValueError for resp.json
                last_err = exc
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])
        raise RuntimeError(f"LLM request failed: {last_err}")


JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_json(content: str) -> Optional[Dict]:
    if not content:
        return None
    match = JSON_RE.search(content)
    if not match:
        return None
    candidate = match.group(0).strip()
    candidate = candidate.strip("\ufeff\n ")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        fixed = candidate.rstrip(", ")
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


CLAUSE_HEADING_RE = re.compile(
    r"^\s*(第[一二三四五六七八九十百千]+条)(?:[：:.．\s]*(.*))?$"
)
ALT_HEADING_RE = re.compile(r"^\s*([一二三四五六七八九十百千]+)[、\.．]\s*(.*)$")


def fallback_clauses(text: str) -> List[Dict[str, str]]:
    if not text.strip():
        return [{"clause_no": "", "section": "", "text": text.strip()}]
    lines = text.splitlines()
    clauses = []
    current = {"clause_no": "", "section": "", "lines": []}

    def flush():
        if current["lines"]:
            clause_text = "\n".join(current["lines"]).strip()
            if clause_text:
                clauses.append(
                    {
                        "clause_no": current["clause_no"],
                        "section": current["section"],
                        "text": clause_text,
                    }
                )
        current["lines"] = []

    for line in lines:
        if not line.strip():
            if current["lines"]:
                current["lines"].append("")
            continue
        m = CLAUSE_HEADING_RE.match(line)
        if m:
            flush()
            current["clause_no"] = m.group(1) or ""
            section = (m.group(2) or "").strip()
            current["section"] = section
            remainder = line[m.end():].strip()
            current["lines"] = [remainder] if remainder else []
            continue
        m2 = ALT_HEADING_RE.match(line)
        if m2:
            flush()
            current["clause_no"] = m2.group(1)
            current["section"] = m2.group(2).strip()
            current["lines"] = []
            continue
        current["lines"].append(line)
    flush()

    if clauses:
        return clauses

    # heading not found: use sliding chunks
    return chunk_text_by_window(text)


def chunk_text_by_window(text: str, min_chars: int = 400, max_chars: int = 900, overlap: int = 100) -> List[Dict[str, str]]:
    text = text.strip()
    if not text:
        return [{"clause_no": "", "section": "", "text": ""}]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        segment = text[start:end]
        chunks.append({"clause_no": "", "section": "", "text": segment.strip()})
        if end >= length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks or [{"clause_no": "", "section": "", "text": text}]


def llm_split_text(
    title: str,
    text: str,
    client: DeepseekClient,
    max_chars: int,
    window: int,
    overlap: int,
) -> Optional[List[Dict[str, str]]]:
    prompt_template = (
        "任务：将下列《{TITLE}》合同文本分割为“条款级”结构化 JSON。\n"
        "要求：\n"
        "1) 按“第……条/第一条/第二条”等条款标题切分；若标题不明确，按语义连续自然段合并成约400–900字片段；\n"
        "2) 仅返回 JSON，键为：clauses: [{clause_no, section, text}]；\n"
        "   - clause_no：如“第一条”，可为空；\n"
        "   - section：条标题（如“违约责任”），可为空；\n"
        "   - text：该条完整正文（去多余空行，不改写）；\n"
        "3) 条款顺序必须与原文一致；不要任何解释、前后缀。\n"
        "合同全文如下：\n{TEXT}\n"
        "输出示例：\n"
        "{\"clauses\":[\n  {\"clause_no\":\"第一条\",\"section\":\"车辆基本情况\",\"text\":\"……\"},\n"
        "  {\"clause_no\":\"第二条\",\"section\":\"价款与支付\",\"text\":\"……\"}\n]}"
    )
    segments = sliding_windows(text, window, overlap) if len(text) > max_chars else [text]
    all_clauses: List[Dict[str, str]] = []
    for seg in segments:
        prompt = prompt_template.replace("{TITLE}", title).replace("{TEXT}", seg)
        try:
            content = client.chat(prompt)
            parsed = extract_json(content)
            if not parsed or "clauses" not in parsed:
                return None
            seg_clauses = []
            for clause in parsed.get("clauses", []):
                text_value = (clause.get("text") or "").strip()
                if not text_value:
                    continue
                seg_clauses.append(
                    {
                        "clause_no": clause.get("clause_no", ""),
                        "section": clause.get("section", ""),
                        "text": text_value,
                    }
                )
            all_clauses.extend(seg_clauses)
        except Exception:
            return None
    return all_clauses or None


def estimate_tokens(text: str) -> int:
    text = text or ""
    return max(1, len(text) // 2)


def business_type_from_path(path: Path) -> str:
    parent = path.parent.name.lower()
    return parent if parent in VALID_BUSINESS_TYPES else "buy_sell"


def compute_doc_id(path: Path, business_type: str) -> str:
    abs_path = str(path.resolve())
    size = path.stat().st_size if path.exists() else 0
    digest = hashlib.md5(f"{abs_path}|{size}".encode("utf-8")).hexdigest()[:8]
    return f"TPL-{business_type.upper()}-{digest}"


def process_file(path: Path, args, client: Optional[DeepseekClient], stats: Dict[str, int], docs_writer, chunks_writer):
    business_type = business_type_from_path(path)
    legal_type = LEGAL_MAP.get(business_type, LEGAL_MAP["buy_sell"])
    title = path.stem

    try:
        raw_text = load_text(path)
    except Exception as exc:
        print(f"[跳过] {path}: 读取失败 {exc}")
        return

    cleaned = clean_text(raw_text)
    length_chars = len(cleaned)

    clauses = None
    used_fallback = False
    attempted_llm = bool(cleaned and client)

    if attempted_llm:
        clauses = llm_split_text(title, cleaned, client, args.max_chars, args.window, args.overlap)

    if not clauses:
        clauses = fallback_clauses(cleaned)
        used_fallback = True
        if attempted_llm:
            stats["fallback"] += 1

    chunk_entries = []
    for clause in clauses:
        text_value = clause.get("text", "").strip()
        if not text_value and len(clauses) > 1:
            continue
        clause_no = clause.get("clause_no", "").strip()
        section = clause.get("section", "").strip()
        if clause_no and section:
            section_label = f"{clause_no} {section}"
        elif clause_no:
            section_label = clause_no
        else:
            section_label = section
        chunk_entries.append(
            {
                "clause_no": clause_no,
                "section": section_label,
                "text": text_value,
            }
        )

    if not chunk_entries:
        chunk_entries = [{"clause_no": "", "section": "", "text": cleaned}]

    doc_id = compute_doc_id(path, business_type)
    doc_record = {
        "doc_id": doc_id,
        "doc_type": "contract_template",
        "business_type": business_type,
        "legal_type": legal_type,
        "title": title,
        "language": "zh",
        "jurisdiction": "CN",
        "chunk_count": len(chunk_entries),
        "length_chars": length_chars,
        "tags": [],
    }
    docs_writer.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

    for idx, clause in enumerate(chunk_entries, start=1):
        chunk_record = {
            "chunk_id": f"{doc_id}#clause-{idx}",
            "doc_id": doc_id,
            "doc_type": "contract_template",
            "business_type": business_type,
            "legal_type": legal_type,
            "order": idx,
            "clause_no": clause.get("clause_no", ""),
            "section": clause.get("section", ""),
            "text": clause.get("text", ""),
            "token_count": estimate_tokens(clause.get("text", "")),
        }
        chunks_writer.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

    stats["files"] += 1
    stats.setdefault(business_type, 0)
    stats[business_type] += 1
    if used_fallback:
        print(f"[回退] {path}")
    else:
        print(f"[完成] {path}")


def main():
    parser = argparse.ArgumentParser(description="合同范本条款切分脚本")
    parser.add_argument("--input_dir", required=True, help="合同范本根目录")
    parser.add_argument("--output_dir", default="out_contracts_chunks", help="输出目录")
    parser.add_argument("--api_base", default="https://api.deepseek.com", help="DeepSeek API base URL")
    parser.add_argument("--model", default="deepseek-chat", choices=["deepseek-chat", "deepseek-reasoner"])
    parser.add_argument("--max_chars", type=int, default=200000)
    parser.add_argument("--window", type=int, default=20000)
    parser.add_argument("--overlap", type=int, default=1000)
    parser.add_argument("--rate_limit_qps", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("环境变量 DEEPSEEK_API_KEY 未设置", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = output_dir / "docs.jsonl"
    chunks_path = output_dir / "chunks.jsonl"

    client = DeepseekClient(
        api_base=args.api_base,
        model=args.model,
        api_key=api_key,
        qps=args.rate_limit_qps,
        retries=args.retries,
        timeout=args.timeout,
    )

    all_files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".docx", ".pdf"}]
    stats = {"files": 0, "fallback": 0}
    with docs_path.open("w", encoding="utf-8") as docs_writer, chunks_path.open(
        "w", encoding="utf-8"
    ) as chunks_writer:
        for file_path in all_files:
            process_file(file_path, args, client, stats, docs_writer, chunks_writer)

    print("处理完成：")
    print(f"  总文件数：{stats['files']}")
    print(f"  回退次数：{stats['fallback']}")
    for btype in VALID_BUSINESS_TYPES:
        if btype in stats:
            print(f"  {btype}: {stats[btype]}")


if __name__ == "__main__":
    main()
