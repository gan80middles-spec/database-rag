import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _ensure_compact_cache(full_text: str, cache: Dict[str, Any]) -> None:
    if cache.get("compact_text") is not None:
        return

    non_ws_indices: List[int] = []
    compact_chars: List[str] = []
    prefix_counts = [0] * (len(full_text) + 1)
    count = 0
    for idx, ch in enumerate(full_text):
        prefix_counts[idx] = count
        if ch.isspace():
            continue
        non_ws_indices.append(idx)
        compact_chars.append(ch)
        count += 1
    prefix_counts[len(full_text)] = count
    cache["compact_text"] = "".join(compact_chars)
    cache["non_ws_indices"] = non_ws_indices
    cache["prefix_counts"] = prefix_counts


def _find_anchor_loose(
    full_text: str,
    anchor: str,
    start_hint: int = 0,
    cache: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    if not anchor:
        return -1, -1

    cache = cache if cache is not None else {}

    exact = full_text.find(anchor, start_hint)
    if exact != -1:
        return exact, exact + len(anchor)

    exact = full_text.find(anchor)
    if exact != -1:
        return exact, exact + len(anchor)

    compact_anchor = _compact(anchor)
    if not compact_anchor:
        return -1, -1

    _ensure_compact_cache(full_text, cache)
    compact_full = cache.get("compact_text", "")
    non_ws_indices: List[int] = cache.get("non_ws_indices", [])
    prefix_counts: List[int] = cache.get("prefix_counts", [])

    bounded_hint = max(0, min(start_hint, len(prefix_counts) - 1)) if prefix_counts else 0
    hint_compact_idx = prefix_counts[bounded_hint] if prefix_counts else 0
    pos = compact_full.find(compact_anchor, hint_compact_idx)
    if pos == -1:
        return -1, -1

    start_idx = non_ws_indices[pos]
    last_pos = pos + len(compact_anchor) - 1
    if last_pos >= len(non_ws_indices):
        return start_idx, start_idx + len(anchor)
    end_idx = non_ws_indices[last_pos] + 1
    return start_idx, end_idx


def materialize_clauses_from_anchors(full_text: str, raw_clauses: List[Dict]) -> List[Dict]:
    """根据锚点把条款正文从原文中切片出来。"""

    if not full_text or not raw_clauses:
        return []

    result: List[Dict[str, str]] = []
    cursor = 0
    text_len = len(full_text)
    compact_cache: Dict[str, Any] = {}

    for clause in raw_clauses:
        start_anchor = (clause.get("start_anchor") or "").strip()
        end_anchor = (clause.get("end_anchor") or "").strip()

        if not start_anchor:
            continue

        start, start_match_end = _find_anchor_loose(
            full_text, start_anchor, cursor, compact_cache
        )
        if start == -1:
            continue

        if end_anchor:
            end_start, end_match_end = _find_anchor_loose(
                full_text, end_anchor, start, compact_cache
            )
            if end_start == -1:
                end = start_match_end
            else:
                end = end_match_end
        else:
            end = start_match_end

        end = min(max(end, start_match_end), text_len)
        text = full_text[start:end].strip()
        if not text:
            continue

        result.append(
            {
                "clause_no": clause.get("clause_no", ""),
                "section": clause.get("section", ""),
                "text": text,
            }
        )
        cursor = end

    return result


def fill_gaps(
    full_text: str,
    clauses: List[Dict[str, str]],
    min_gap_chars: int = 50,
) -> List[Dict[str, str]]:
    """检测条款之间的缺口，并把缺失正文补成匿名条款。"""

    if not full_text or not clauses:
        return clauses or []

    matched: List[Tuple[int, int, Dict[str, str]]] = []
    matched_ids = set()
    cursor = 0
    text_len = len(full_text)

    for clause in clauses:
        text_value = (clause.get("text") or "").strip()
        if not text_value:
            continue
        start = full_text.find(text_value, cursor)
        if start == -1:
            start = full_text.find(text_value)
        if start == -1:
            continue
        end = start + len(text_value)
        matched.append((start, end, clause))
        matched_ids.add(id(clause))
        cursor = end

    if not matched:
        return clauses

    matched.sort(key=lambda item: item[0])
    augmented: List[Tuple[int, int, Dict[str, str]]] = []
    prev_end = 0

    for start, end, clause in matched:
        gap = start - prev_end
        if gap >= min_gap_chars:
            gap_text = full_text[prev_end:start].strip()
            if gap_text:
                augmented.append(
                    (
                        prev_end,
                        start,
                        {"clause_no": "", "section": "", "text": gap_text},
                    )
                )
        augmented.append((start, end, clause))
        prev_end = end

    tail_gap = text_len - prev_end
    if tail_gap >= min_gap_chars:
        gap_text = full_text[prev_end:].strip()
        if gap_text:
            augmented.append(
                (
                    prev_end,
                    text_len,
                    {"clause_no": "", "section": "", "text": gap_text},
                )
            )

    ordered = [item[2] for item in augmented]

    for clause in clauses:
        if id(clause) not in matched_ids:
            ordered.append(clause)

    return ordered


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
        "\n"
        "严格要求（非常重要）：\n"
        "1) 对每一条款，你必须先在【原文】中找到这一条的正文范围，然后：\n"
        "   - start_anchor = 该条款正文【第一个字符开始的连续 30 个字符】；\n"
        "   - end_anchor   = 该条款正文【最后的连续 30 个字符】；\n"
        "2) start_anchor / end_anchor 必须满足：\n"
        "   - 完全逐字从【原文】复制，不得擅自加入、省略或修改任何一个字、标点或空格；\n"
        "   - 不得加入“...”等省略号；不得把不相邻的两段文本拼接在一起；\n"
        "   - 不得跨越页码、章节标题等无关内容；\n"
        "   - 每个锚点长度上限约 60 个字符（通常 30 个字符即可）；\n"
        "   - 必须能在【原文】中用 Ctrl+F 精确搜索到完全一致的子串（忽略大小写）。\n"
        "3) 如果某条款正文中间有换行，你也必须原样保留换行，不要合并成一行；\n"
        "4) 绝对禁止的错误示例（不要这样做）：\n"
        "   - 把“第一章 合作内容”与“第二章 合同金额”连成一个很长的 start_anchor；\n"
        "   - 在锚点中加入“...”表示省略；\n"
        "   - 把页码“1”或“2”与条款标题一起写进锚点。\n"
        "5) 合法锚点示例：\n"
        "   原文：\n"
        "     第一章 服务项目\\n"
        "     1.项目定义:根据甲乙双方友好协商,甲方委托乙方设计制作“四海易购”商城产品详情页面设计。\\n"
        "   正确的 start_anchor 示例：\n"
        "     \"1.项目定义:根据甲乙双方友好协商,甲方委托\"\n"
        "   错误的 start_anchor 示例（禁止）：\n"
        "     \"第一章服务项目 1.项目定义:根据甲乙双方友好协商,甲方委托乙方设计制作“四海易购”商城产品详情页面设计。第二章合同金额与付款方式\"（跨了下一章）。\n"
        "\n"
        "分割规则：\n"
        "1) 优先按“第……条”“第一条”“第二条”等条款编号切分；\n"
        "2) 有些合同使用“第一章/第二章”等章节标题，章节下可继续按条款编号细分；\n"
        "3) 如果某些部分没有明显条款标题，则按语义连续自然段合并成约 400–900 字的片段，每个片段也视为一条记录；\n"
        "4) 条款顺序必须与原文一致，不要跳过条款，不要合并本应独立的条款。\n"
        "\n"
        "输出格式要求：\n"
        "1) 仅返回 JSON，顶层键为：clauses。\n"
        "2) clauses 是一个数组，元素为：{clause_no, section, start_anchor, end_anchor}。\n"
        "   - clause_no：条款编号，如“第一条”“第二十二条”，没有则填空字符串 \"\"；\n"
        "   - section：条款标题，如“违约责任”“价款与支付”，没有标题则填空字符串 \"\"；\n"
        "   - start_anchor：条款正文开头的锚点（见上面的严格要求）；\n"
        "   - end_anchor：条款正文结尾的锚点（见上面的严格要求）。\n"
        "\n"
        "示例输出（仅为格式示例）：\n"
        "{\"clauses\": [\n"
        "  {\"clause_no\": \"第一条\", \"section\": \"车辆基本情况\", \"start_anchor\": \"本合同所指车辆为……\", \"end_anchor\": \"……并保证车辆不存在抵押、查封。\"},\n"
        "  {\"clause_no\": \"第二条\", \"section\": \"价款与支付\", \"start_anchor\": \"本合同项下车辆总价款为人民币\", \"end_anchor\": \"……乙方应于收到车辆之日起三日内付清。\"}\n"
        "]}\n"
        "\n"
        "合同全文如下：\n"
        "{TEXT}\n"
    )

    segments = sliding_windows(text, window, overlap) if len(text) > max_chars else [text]
    all_clauses: List[Dict[str, str]] = []

    for idx, seg in enumerate(segments):
        prompt = prompt_template.replace("{TITLE}", title).replace("{TEXT}", seg)
        try:
            content = client.chat(prompt)
        except Exception as e:
            # 1) 直接打印出是哪一段 HTTP / 网络错误
            print(f"[LLM ERROR][{title}][seg={idx}] {e}")
            return None

        parsed = extract_json(content)
        if not parsed or "clauses" not in parsed:
            # 2) 把模型原始输出截断打印 & 落到文件里
            preview = (content or "").strip().replace("\n", " ")
            if len(preview) > 300:
                preview = preview[:300] + "..."

            print(f"[LLM JSON FAIL][{title}][seg={idx}] preview={preview}")

            with open("deepseek_bad_responses.log", "a", encoding="utf-8") as f:
                f.write(f"\n===== {title} seg={idx} =====\n")
                f.write(content or "")
                f.write("\n")
            return None

        raw_clauses = parsed.get("clauses", []) or []
        seg_clauses = materialize_clauses_from_anchors(seg, raw_clauses)

        print(
            f"[DEBUG][{title}][seg={idx}] "
            f"raw_clauses={len(raw_clauses)}, matched={len(seg_clauses)}"
        )

        
        debug_path = f"debug_anchors_{sanitize_filename(title)}_seg{idx}.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(raw_clauses, f, ensure_ascii=False, indent=2)
        abs_path = os.path.abspath(debug_path)
        print(
            f"[DEBUG][{title}][seg={idx}] 原始锚点已保存到 {abs_path}，"
            "可直接打开查看 DeepSeek 输出"
        )

        if not seg_clauses and raw_clauses:
            print(f"[LLM ANCHOR FAIL][{title}][seg={idx}] 无法匹配锚点")
        all_clauses.extend(seg_clauses)

    if not all_clauses:
        print(f"[LLM EMPTY CLAUSES][{title}]")
        return None

    return all_clauses


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

    clauses = None
    attempted_llm = bool(cleaned and client)

    if attempted_llm:
        clauses = llm_split_text(title, cleaned, client, args.max_chars, args.window, args.overlap)
        if clauses:
            clauses = fill_gaps(cleaned, clauses, min_gap_chars=50)

    if not clauses:
        raise RuntimeError(f"LLM 切分失败：{path}")

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