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
from tqdm import tqdm


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
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}
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


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ``` 之类的 Markdown 代码块外壳，只保留内部内容（支持同一行内有 JSON 的情况）。"""
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()

    # 处理第一行：可能是 "```" / "```json" / "```json { ... }"
    if lines:
        first = lines[0]
        m = re.match(r"^```[a-zA-Z0-9_-]*\s*(.*)$", first)
        if m:
            # m.group(1) 是去掉 ```json 之后的内容
            rest = m.group(1).strip()
            if rest:
                # 保留这一行的 JSON 内容
                lines[0] = rest
            else:
                # 这一行只有 ``` 或 ```json，整行丢弃
                lines = lines[1:]

    # 处理最后一行：可能是 "..." 或 "... ```"
    if lines:
        last = lines[-1]
        if "```" in last:
            idx = last.find("```")
            before = last[:idx].rstrip()
            if before:
                lines[-1] = before
            else:
                # 行里只有 ```，整体删掉
                lines = lines[:-1]

    return "\n".join(lines).strip()



def _iter_json_blocks(text: str):
    """
    从一段文本里，按括号配对依次抽出形如 {...} 或 [...] 的候选 JSON 片段。
    避免 {.*} 贪婪匹配导致的 {...}{...} 问题。
    """
    n = len(text)
    i = 0
    while i < n:
        # 找到下一个起始括号
        while i < n and text[i] not in "{[":
            i += 1
        if i >= n:
            break

        start = i
        stack = [text[i]]
        i += 1

        while i < n and stack:
            ch = text[i]
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                stack.pop()
            i += 1

        # stack 为空说明找到了一段完整的 JSON 块
        if not stack:
            block = text[start:i].strip()
            if block:
                yield block
        else:
            # 括号没闭合，直接结束
            break


def extract_json(content: str) -> Optional[Dict]:
    """
    尽量鲁棒地从 LLM 返回内容中抽出一个 JSON：
    1. 去掉 ```json 代码块外壳；
    2. 先整体尝试 json.loads；
    3. 再按括号配对迭代每一段 {...} / [...]；
    4. 根为 list 时包一层 {"clauses": [...]}。
    """
    if not content:
        return None

    text = _strip_code_fence(content)

    # 1) 直接整体尝试
    for cand in (text, text.rstrip(", ")):
        if not cand:
            continue
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                return {"clauses": data}
        except json.JSONDecodeError:
            pass

    # 2) 按括号配对依次尝试各个 JSON 块
    for block in _iter_json_blocks(text):
        for cand in (block, block.rstrip(", ")):
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    return data
                if isinstance(data, list):
                    return {"clauses": data}
            except json.JSONDecodeError:
                continue

    return None



# ========= 规则切分：按“第×条”拆合同条款 =========

# 匹配“第×条 标题”的条款抬头：
# - no:  第三十八条
# - title: 保守商业秘密和竞业限制
CLAUSE_HEADING_RE = re.compile(
    r"(?P<full>(?P<no>第[一二三四五六七八九十百千万零两\d]+条)\s*(?P<title>[^\n\r，。,：:]{0,40}))"
)


def split_clauses_by_regex(full_text: str) -> List[Dict[str, str]]:
    """
    使用正则按“第×条”切分合同条款。

    返回的每个元素包含：
    - clause_no: 例如 “第三十八条”，找不到则为 ""；
    - section: 条款标题，例如 “保守商业秘密和竞业限制”，如果没有标题则为 ""；
    - text: 条款完整正文（包含抬头行）。
    """
    text = full_text or ""
    text = text.strip()
    if not text:
        return []

    matches = list(CLAUSE_HEADING_RE.finditer(text))
    clauses: List[Dict[str, str]] = []

    # 没有任何“第×条”时，整个文档视为一个条款
    if not matches:
        clauses.append({"clause_no": "", "section": "", "text": text})
        return clauses

    # 前言：第一个“第×条”之前的部分
    first_start = matches[0].start()
    if first_start > 0:
        preface = text[:first_start].strip()
        if preface:
            clauses.append(
                {
                    "clause_no": "",
                    "section": "",
                    "text": preface,
                }
            )

    # 正式条款
    for idx, m in enumerate(matches):
        clause_no = (m.group("no") or "").strip()
        raw_title = (m.group("title") or "").strip()

        # 清理标题前面的标点
        section_title = raw_title.lstrip("：:、.．，,")

        start = m.start()
        if idx + 1 < len(matches):
            end = matches[idx + 1].start()
        else:
            end = len(text)
        body = text[start:end].strip()
        if not body:
            continue

        clauses.append(
            {
                "clause_no": clause_no,
                "section": section_title,
                "text": body,
            }
        )

    return clauses


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


def sanitize_filename(name: str) -> str:
    name = name.strip().replace("\u3000", " ")
    safe = re.sub(r"[\\/\:*?\"<>|]+", "_", name)
    safe = re.sub(r"\s+", "_", safe)
    return safe or "document"


def llm_generate_tags(title: str, text: str, client: Optional[DeepseekClient], max_chars: int = 2000) -> List[str]:
    if not text or not client:
        return []

    snippet = text.strip()[:max_chars]
    if not snippet:
        return []

    prompt = (
        "任务：根据下列合同标题和正文摘要，生成 2-4 个简短中文标签，总结合同类型、标的或场景。\n"
        "要求：\n"
        "1) 标签不超过 8 个汉字；\n"
        "2) 避免编号、标点或重复含义；\n"
        "3) 仅返回 JSON，格式为 {\"tags\": [\"标签1\", \"标签2\"]}；\n"
        "4) 不要附加解释。\n"
        f"合同标题：《{title}》\n"
        f"正文摘要：\n{snippet}\n"
    )

    try:
        content = client.chat(prompt)
    except Exception as exc:
        print(f"[LLM TAGS ERROR][{title}] {exc}")
        return []

    parsed = extract_json(content)
    tags = parsed.get("tags") if isinstance(parsed, dict) else None  # type: ignore[arg-type]
    if not isinstance(tags, list):
        print(f"[LLM TAGS FAIL][{title}] content={(content or '').strip()[:200]}")
        return []

    cleaned_tags: List[str] = []
    for tag in tags:
        if isinstance(tag, str):
            tag = tag.strip()
            if tag:
                cleaned_tags.append(tag)

    return cleaned_tags[:4]


def process_file(
    path: Path,
    args,
    client: Optional[DeepseekClient],
    stats: Dict[str, int],
):
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

    # 1) 规则优先：按“第×条”切分条款
    clauses = split_clauses_by_regex(cleaned)

    # 理论上 split_clauses_by_regex 至少会返回 1 条；这里再兜一层底
    if not clauses:
        clauses = [{"clause_no": "", "section": "", "text": cleaned}]

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
    tags = llm_generate_tags(title, cleaned, client)

    doc_record = {
        "doc_id": doc_id,
        "doc_type": "contract_template",
        "business_type": business_type,
        "legal_type": legal_type,
        "title": title,
        "chunk_count": len(chunk_entries),
        "length_chars": length_chars,
        "tags": tags,
    }
    chunk_records = []
    for idx, clause in enumerate(chunk_entries, start=1):
        chunk_records.append(
            {
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
        )

    stats["files"] += 1
    stats.setdefault(business_type, 0)
    stats[business_type] += 1
    print(f"[完成] {path}")

    return doc_record, chunk_records


def main():
    parser = argparse.ArgumentParser(description="合同范本条款切分脚本")
    parser.add_argument("--input_dir", required=True, help="合同范本根目录")
    parser.add_argument("--output_dir", default="out_contracts_chunks", help="输出目录")
    parser.add_argument("--api_base", default="https://api.deepseek.com", help="DeepSeek API base URL")
    parser.add_argument("--model", default="deepseek-chat", choices=["deepseek-chat", "deepseek-reasoner"])
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

    client = DeepseekClient(
        api_base=args.api_base,
        model=args.model,
        api_key=api_key,
        qps=args.rate_limit_qps,
        retries=args.retries,
        timeout=args.timeout,
    )

    all_files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".docx", ".pdf"}]
    stats = {"files": 0}
    for file_path in tqdm(all_files, desc="Processing files"):
        result = process_file(file_path, args, client, stats)
        if not result:
            continue
        doc_record, chunk_records = result
        base_name = sanitize_filename(file_path.stem)
        doc_file = output_dir / f"{base_name}-docs.jsonl"
        chunk_file = output_dir / f"{base_name}-chunks.jsonl"
        with doc_file.open("w", encoding="utf-8") as dw:
            dw.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
        with chunk_file.open("w", encoding="utf-8") as cw:
            for chunk in chunk_records:
                cw.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("处理完成：")
    print(f"  总文件数：{stats['files']}")
    for btype in VALID_BUSINESS_TYPES:
        if btype in stats:
            print(f"  {btype}: {stats[btype]}")


if __name__ == "__main__":
    main()