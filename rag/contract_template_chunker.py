import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

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

PRESENCE_SCHEMAS = {
    "buy_sell": [
        {"id": "party_info", "label": "当事人信息（出卖人/买受人）", "required": True},
        {"id": "subject", "label": "标的物名称/型号", "required": True},
        {"id": "quantity", "label": "数量与计量方式", "required": True},
        {"id": "quality", "label": "质量标准/技术规格", "required": True},
        {"id": "price_payment", "label": "价款与结算方式", "required": True},
        {"id": "delivery", "label": "交付期限/地点/方式", "required": True},
        {"id": "inspection", "label": "检验标准与方法/验收", "required": True},
        {"id": "packaging", "label": "包装与随附资料", "required": False},
        {"id": "risk_title_transfer", "label": "风险转移/所有权转移", "required": False},
        {"id": "after_sales", "label": "质保/售后", "required": False},
        {"id": "ip_warranty", "label": "知识产权不侵权保证（如适用）", "required": False},
        {"id": "breach_liability", "label": "违约责任/损害赔偿", "required": True},
        {"id": "dispute_resolution", "label": "争议解决（适用法/仲裁/法院）", "required": True},
    ],
    "lease": [
        {"id": "party_info", "label": "出租人/承租人信息", "required": True},
        {"id": "leased_property", "label": "租赁物名称/数量/状况", "required": True},
        {"id": "purpose", "label": "用途与使用限制", "required": True},
        {"id": "term", "label": "租赁期限", "required": True},
        {"id": "rent_payment", "label": "租金及支付期限/方式", "required": True},
        {"id": "delivery_return", "label": "交付/期满返还与原状恢复", "required": True},
        {"id": "maintenance", "label": "维修责任与费用分担", "required": True},
        {"id": "deposit", "label": "押金/担保（如有）", "required": False},
        {"id": "sublease_assignment", "label": "转租/转借/转让限制", "required": False},
        {"id": "risk_liability", "label": "风险承担/保险（如适用）", "required": False},
        {"id": "breach_termination", "label": "违约与解除/提前终止", "required": True},
        {"id": "dispute_resolution", "label": "争议解决", "required": True},
    ],
    "labor": [
        {"id": "party_info", "label": "用人单位/劳动者信息", "required": True},
        {"id": "term", "label": "合同期限/试用期（如有）", "required": True},
        {"id": "job_and_location", "label": "工作内容/岗位与地点", "required": True},
        {"id": "hours_leave", "label": "工作时间/休息休假", "required": True},
        {"id": "compensation", "label": "劳动报酬（工资标准/发放周期）", "required": True},
        {"id": "social_insurance", "label": "社会保险/公积金约定", "required": True},
        {"id": "safety_conditions", "label": "劳动保护/条件/职业危害防护", "required": True},
        {"id": "training", "label": "培训条款（如约定）", "required": False},
        {"id": "confidentiality_noncompete", "label": "保密/竞业限制（如约定）", "required": False},
        {"id": "discipline", "label": "劳动纪律/规章制度遵守", "required": False},
        {"id": "termination", "label": "解除/终止与经济补偿", "required": True},
        {"id": "dispute_resolution", "label": "争议解决（仲裁/诉讼）", "required": True},
    ],
    "nda": [
        {"id": "party_info", "label": "当事人信息", "required": True},
        {"id": "confidential_definition", "label": "保密信息的范围/定义", "required": True},
        {"id": "use_scope", "label": "使用范围/目的限制", "required": True},
        {"id": "exceptions", "label": "保密义务例外（公开/已知/独立获得等）", "required": True},
        {"id": "security_measures", "label": "安全措施/访问控制", "required": False},
        {"id": "term_duration", "label": "保密期限/存续期", "required": True},
        {"id": "return_destroy", "label": "资料返还/销毁", "required": True},
        {"id": "third_party", "label": "第三方披露与传递限制", "required": False},
        {"id": "ip_ownership", "label": "知识产权归属与不授予条款", "required": False},
        {"id": "breach_remedy", "label": "违约责任/禁令救济/损害赔偿", "required": True},
        {"id": "dispute_resolution", "label": "争议解决", "required": True},
    ],
    "outsourcing": [
        {"id": "party_info", "label": "委托方/服务方信息", "required": True},
        {"id": "scope_deliverables", "label": "工作范围/交付成果与规格", "required": True},
        {"id": "timeline_milestones", "label": "进度/里程碑/服务期限", "required": True},
        {"id": "acceptance", "label": "验收标准与流程", "required": True},
        {"id": "fees_settlement", "label": "费用/结算/发票", "required": True},
        {"id": "change_control", "label": "变更管理（需求/范围/价格）", "required": False},
        {"id": "materials_ip", "label": "资料/工具/知识产权归属与许可", "required": True},
        {"id": "confidentiality", "label": "保密义务/数据合规", "required": True},
        {"id": "warranty_support", "label": "质量保证/维护支持（如有）", "required": False},
        {"id": "subcontracting", "label": "分包/人员更替限制", "required": False},
        {"id": "breach_liability", "label": "违约责任/赔偿/限责", "required": True},
        {"id": "termination", "label": "解除/终止与费用清算", "required": True},
        {"id": "dispute_resolution", "label": "争议解决", "required": True},
    ],
    "software": [
        {"id": "party_info", "label": "许可方/被许可方信息", "required": True},
        {"id": "license_grant", "label": "许可范围（地域/期限/方式/是否独占）", "required": True},
        {"id": "usage_restrictions", "label": "使用限制（并发/终端/不得反向工程等）", "required": True},
        {"id": "delivery_support", "label": "交付方式/安装部署/技术支持", "required": False},
        {"id": "updates_maintenance", "label": "升级/维护/服务级别（SLA）", "required": False},
        {"id": "fees_audit", "label": "费用/计费方式/审计权", "required": True},
        {"id": "ip_ownership", "label": "知识产权归属与不授予条款", "required": True},
        {"id": "data_protection", "label": "数据安全/个人信息/接口调用合规", "required": False},
        {"id": "oss_thirdparty", "label": "开源与第三方组件约束（如适用）", "required": False},
        {"id": "infringement_indemnity", "label": "侵权担保与赔偿", "required": True},
        {"id": "restricted_clauses_compliance", "label": "技术合同限制性条款合规（民法典864）", "required": True},
        {"id": "termination", "label": "终止/到期后的处置（停用/反回/销毁）", "required": True},
        {"id": "dispute_resolution", "label": "争议解决", "required": True},
    ],
}

PRESENCE_ID_SET: Dict[str, Set[str]] = {
    btype: {item["id"] for item in schema}
    for btype, schema in PRESENCE_SCHEMAS.items()
}

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

CHAPTER_HEADING_RE = re.compile(
    r"(?m)^\s*(?P<full>(?P<no>第[一二三四五六七八九十百千万零两\d]+章)\s*(?P<title>[^\n\r，。,：:]{0,40}))"
)

GENERIC_HEADING_RE = re.compile(
    r"(?m)^\s*(?P<full>(?P<marker>(?:\d{1,3}|[一二三四五六七八九十]+|[（(][一二三四五六七八九十0-9]{1,3}[)）]))[\.．、)]\s*(?P<title>[^\n\r]{0,60}))"
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


def split_sections_by_heading(
    full_text: str,
    heading_regex: re.Pattern,
    clause_group: Optional[str] = None,
) -> List[Dict[str, str]]:
    """通用的章节/编号切分：按给定正则把文档拆成多个 section。"""

    text = full_text or ""
    text = text.strip()
    if not text:
        return []

    matches = list(heading_regex.finditer(text))
    if not matches:
        return []

    clauses: List[Dict[str, str]] = []

    first_start = matches[0].start()
    if first_start > 0:
        preface = text[:first_start].strip()
        if preface:
            clauses.append({"clause_no": "", "section": "", "text": preface})

    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        clause_no = ""
        if clause_group:
            clause_no = (m.groupdict().get(clause_group) or "").strip()

        title = (m.groupdict().get("title") or "").strip()
        full_heading = (m.groupdict().get("full") or "").strip()
        section_label = title or full_heading or clause_no

        clauses.append({"clause_no": clause_no, "section": section_label, "text": body})

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


def compute_text_md5(text: str) -> str:
    text = text or ""
    return f"md5:{hashlib.md5(text.encode('utf-8')).hexdigest()}"


def compute_chunk_md5(chunk_id: str, text: str) -> str:
    text = text or ""
    return f"md5:{hashlib.md5(f'{chunk_id}|{text}'.encode('utf-8')).hexdigest()}"


def sanitize_filename(name: str) -> str:
    name = name.strip().replace("\u3000", " ")
    safe = re.sub(r"[\\/\:*?\"<>|]+", "_", name)
    safe = re.sub(r"\s+", "_", safe)
    return safe or "document"


def build_presence_template(business_type: str) -> Dict[str, bool]:
    ids = PRESENCE_ID_SET.get(business_type, set())
    return {pid: False for pid in ids}


def llm_infer_legal_type(title: str, text: str, client: Optional[DeepseekClient], max_chars: int = 2000) -> Optional[str]:
    if not client:
        return None

    snippet = text.strip()[:max_chars]
    prompt = (
        "请判断下列合同的法律体系或合同类别标签。\n"
        "必须从候选列表中选择最贴近的一项，并以 JSON 返回：{\"legal_type\": \"...\"}。\n"
        "候选列表：\n"
        "- 买卖合同\n"
        "- 租赁合同\n"
        "- 劳动合同法体系\n"
        "- 承揽/委托/技术合同（按具体条款）\n"
        "- 保密协议（非典型合同，保密义务）\n"
        "- 买卖/技术开发/技术许可（按内容）\n"
        "合同标题：\n"
        f"{title}\n"
        "正文摘要：\n"
        f"{snippet}\n"
    )

    try:
        content = client.chat(prompt)
    except Exception as exc:
        print(f"[LLM LEGAL TYPE ERROR][{title}] {exc}")
        return None

    parsed = extract_json(content)
    if not isinstance(parsed, dict):
        return None
    legal_type = parsed.get("legal_type")
    if isinstance(legal_type, str):
        return legal_type.strip() or None
    return None


def llm_classify_clause(
    business_type: str,
    section_label: str,
    text: str,
    client: Optional[DeepseekClient],
    max_chars: int = 1200,
) -> List[str]:
    if not client:
        return []

    schema = PRESENCE_SCHEMAS.get(business_type, [])
    if not schema:
        return []

    snippet = text.strip()[:max_chars]
    choices = "\n".join(
        [f"- {item['id']}: {item['label']} (必填:{'是' if item['required'] else '否'})" for item in schema]
    )
    prompt = (
        "请根据下列合同条款内容，判断其涵盖的条款类型 ID。\n"
        "仅能从给定列表中选择，允许多选；若均不匹配则返回空数组。\n"
        "返回 JSON 形如 {\"clause_type\": [\"id1\", \"id2\"]}。\n"
        f"合同业务类型：{business_type}\n"
        f"候选条款：\n{choices}\n"
        f"条款标题：{section_label}\n"
        "条款内容：\n"
        f"{snippet}\n"
    )

    try:
        content = client.chat(prompt)
    except Exception as exc:
        print(f"[LLM CLAUSE TYPE ERROR][{section_label}] {exc}")
        return []

    parsed = extract_json(content)
    clause_type = parsed.get("clause_type") if isinstance(parsed, dict) else None  # type: ignore[arg-type]
    if not isinstance(clause_type, list):
        print(f"[LLM CLAUSE TYPE FAIL][{section_label}] content={(content or '').strip()[:200]}")
        return []

    allowed = PRESENCE_ID_SET.get(business_type, set())
    cleaned: List[str] = []
    for item in clause_type:
        if isinstance(item, str):
            item_clean = item.strip()
            if item_clean and item_clean in allowed and item_clean not in cleaned:
                cleaned.append(item_clean)

    return cleaned


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

    if len(clauses) <= 1:
        chapter_sections = split_sections_by_heading(
            cleaned,
            CHAPTER_HEADING_RE,
            clause_group="no",
        )
        if len(chapter_sections) > 1:
            clauses = chapter_sections

    if len(clauses) <= 1:
        numbered_sections = split_sections_by_heading(
            cleaned,
            GENERIC_HEADING_RE,
            clause_group="marker",
        )
        if len(numbered_sections) > 1:
            clauses = numbered_sections

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
    presence = build_presence_template(business_type)
    legal_type = llm_infer_legal_type(title, cleaned, client) or LEGAL_MAP.get(
        business_type, LEGAL_MAP["buy_sell"]
    )
    doc_text_md5 = compute_text_md5(cleaned)

    chunk_records = []
    for idx, clause in enumerate(chunk_entries, start=1):
        clause_text = clause.get("text", "")
        chunk_id = f"{doc_id}#clause-{idx}"
        clause_types = llm_classify_clause(
            business_type,
            clause.get("section", ""),
            clause_text,
            client,
        )
        for ctype in clause_types:
            if ctype in presence:
                presence[ctype] = True
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_type": "contract_template",
                "business_type": business_type,
                "legal_type": legal_type,
                "order": idx,
                "clause_no": clause.get("clause_no", ""),
                "section": clause.get("section", ""),
                "text": clause_text,
                "token_count": estimate_tokens(clause_text),
                "clause_type": clause_types,
                "chunk_md5": compute_chunk_md5(chunk_id, clause_text),
                "text_md5": compute_text_md5(clause_text),
            }
        )

    doc_record = {
        "doc_id": doc_id,
        "doc_type": "contract_template",
        "business_type": business_type,
        "legal_type": legal_type,
        "title": title,
        "chunk_count": len(chunk_entries),
        "length_chars": length_chars,
        "tags": tags,
        "presence": presence,
        "text_md5": doc_text_md5,
    }

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
