# deepseek_casegraph.py
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# Pydantic v2
from pydantic import TypeAdapter, ValidationError


# --------------------------- logging ---------------------------

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
    )
    return logging.getLogger("deepseek_casegraph")


log = setup_logger()


# --------------------------- io utils ---------------------------

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_doc_ids(out_path: Path, err_path: Optional[Path]) -> set[str]:
    done = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            doc_id = r.get("doc_id") or r.get("source_doc_id") or r.get("id")
            if doc_id:
                done.add(str(doc_id))
    if err_path and err_path.exists():
        for r in read_jsonl(err_path):
            doc_id = r.get("doc_id")
            if doc_id:
                done.add(str(doc_id))
    return done


# --------------------------- tool loading ---------------------------

def load_tool(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    # allow formats:
    # 1) tool dict: {"type":"function","function":{...}}
    # 2) list of tools: [ {...}, {...} ]
    # 3) wrapper: {"tools":[...]}
    if isinstance(obj, list):
        if not obj:
            raise ValueError("tool file is an empty list")
        tool = obj[0]
    elif isinstance(obj, dict) and "tools" in obj and isinstance(obj["tools"], list):
        if not obj["tools"]:
            raise ValueError("tool file has empty 'tools'")
        tool = obj["tools"][0]
    elif isinstance(obj, dict) and obj.get("type") == "function" and "function" in obj:
        tool = obj
    else:
        raise ValueError("Unrecognized tool JSON shape. Expect tool / tools[] / {tools:[...]}")

    fn = tool.get("function", {})
    name = fn.get("name")
    if not name:
        raise ValueError("tool.function.name missing")
    if fn.get("parameters", {}).get("type") != "object":
        log.warning("tool.function.parameters.type is not 'object' (may reduce tool-call reliability)")
    return tool


def set_tool_strict(tool: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    tool2 = json.loads(json.dumps(tool, ensure_ascii=False))
    tool2["function"]["strict"] = bool(strict)
    return tool2


# --------------------------- pydantic model loading ---------------------------

def load_casegraph_type(import_path: str):
    """
    import_path example: "schemas.graphs:CaseGraph"
    """
    if ":" not in import_path:
        raise ValueError(f"--casegraph-import must be like 'schemas.graphs:CaseGraph', got: {import_path}")
    mod_name, attr = import_path.split(":", 1)
    mod = importlib.import_module(mod_name)
    cg = getattr(mod, attr, None)
    if cg is None:
        raise ImportError(f"Cannot find {attr} in module {mod_name}")
    return cg


def get_model_fields(model_type: Any) -> Optional[set[str]]:
    # Pydantic v2 BaseModel has model_fields
    fields = getattr(model_type, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    return None


# --------------------------- extraction ---------------------------

@dataclass
class ExtractResult:
    ok: bool
    args: Optional[Dict[str, Any]] = None
    err: Optional[str] = None
    finish_reason: Optional[str] = None
    has_tool_calls: bool = False
    tool_name: Optional[str] = None
    content_preview: str = ""
    raw_response: Optional[Dict[str, Any]] = None


def _safe_preview(s: Any, n: int = 600) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", "\\n")
    return s[:n]


def _resp_to_dict(resp: Any) -> Optional[Dict[str, Any]]:
    if resp is None:
        return None
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:
            pass
    if hasattr(resp, "dict"):
        try:
            return resp.dict()
        except Exception:
            pass
    return None


def _parse_tool_args(arguments: Any) -> Dict[str, Any]:
    # DeepSeek/OpenAI usually returns JSON string
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise ValueError(f"tool arguments type={type(arguments)} not str/dict")

    s = arguments.strip()

    # strip code fences
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).strip()

    # try direct
    try:
        return json.loads(s)
    except Exception:
        # try best-effort substring
        l = s.find("{")
        r = s.rfind("}")
        if 0 <= l < r:
            return json.loads(s[l : r + 1])
        raise


def _deepseek_extra_body(thinking: bool, thinking_budget: int) -> Optional[Dict[str, Any]]:
    """
    DeepSeek thinking mode is implemented via extra JSON body rather than a top-level param,
    so it won't trigger "unexpected keyword argument thinking" in OpenAI SDK.
    """
    if not thinking:
        return None
    budget = max(0, int(thinking_budget))
    # DeepSeek docs: reasoning_content is returned when thinking mode enabled.
    # The exact request shape is handled as extra_body to avoid SDK signature issues.
    if budget > 0:
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    return {"thinking": {"type": "enabled"}}


def call_deepseek_once(
    *,
    client: OpenAI,
    model: str,
    tool: Dict[str, Any],
    tool_choice: Any,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    thinking_budget: int,
) -> ExtractResult:
    extra_body = _deepseek_extra_body(thinking, thinking_budget)

    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=[tool],
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)

    choice = resp.choices[0]
    msg = choice.message
    finish_reason = getattr(choice, "finish_reason", None)
    tool_calls = getattr(msg, "tool_calls", None)

    raw = _resp_to_dict(resp)

    if tool_calls:
        tc0 = tool_calls[0]
        fn = tc0.function
        tool_name = getattr(fn, "name", None)
        try:
            args = _parse_tool_args(fn.arguments)
            return ExtractResult(
                ok=True,
                args=args,
                finish_reason=finish_reason,
                has_tool_calls=True,
                tool_name=tool_name,
                content_preview=_safe_preview(msg.content),
                raw_response=raw,
            )
        except Exception as e:
            return ExtractResult(
                ok=False,
                err=f"tool_calls present but arguments parse failed: {e}",
                finish_reason=finish_reason,
                has_tool_calls=True,
                tool_name=tool_name,
                content_preview=_safe_preview(msg.content),
                raw_response=raw,
            )

    return ExtractResult(
        ok=False,
        err="no tool_calls",
        finish_reason=finish_reason,
        has_tool_calls=False,
        tool_name=None,
        content_preview=_safe_preview(msg.content),
        raw_response=raw,
    )


# --------------------------- local validation (Pydantic) ---------------------------

def _contains_any(text: str, pats: List[str]) -> bool:
    return any(p in text for p in pats)


def semantic_soft_checks(payload: Dict[str, Any], raw_text: str) -> List[Dict[str, Any]]:
    """
    软规则：只要原文明显出现了某块信息，但 payload 对应字段为空，就产生“软错误”，触发 repair。
    注意：只在字段存在时检查，避免和你的 schema/CaseGraph 不一致。
    """
    errs: List[Dict[str, Any]] = []

    # parties
    if "parties" in payload and isinstance(payload.get("parties"), list):
        if _contains_any(raw_text, ["被告人", "原告", "上诉人", "被上诉人", "申请人", "被申请人", "公诉机关"]):
            if len(payload["parties"]) == 0:
                errs.append({"loc": ["parties"], "msg": "raw_text mentions parties roles but parties is empty", "type": "soft_missing"})

    # evidence
    if "evidence" in payload and isinstance(payload.get("evidence"), list):
        if _contains_any(raw_text, ["证据", "证言", "鉴定", "勘查", "书证", "物证", "笔录"]):
            if len(payload["evidence"]) == 0:
                errs.append({"loc": ["evidence"], "msg": "raw_text mentions evidence but evidence is empty", "type": "soft_missing"})

    # claims / prosecution allegation
    if "claims" in payload and isinstance(payload.get("claims"), list):
        if _contains_any(raw_text, ["公诉机关指控", "起诉书指控", "诉称"]):
            if len(payload["claims"]) == 0:
                errs.append({"loc": ["claims"], "msg": "raw_text mentions claims/allegations but claims is empty", "type": "soft_missing"})

    # legal issues / applied law candidates
    if "law_candidates" in payload and isinstance(payload.get("law_candidates"), list):
        if "《" in raw_text and "条" in raw_text:
            if len(payload["law_candidates"]) == 0:
                errs.append({"loc": ["law_candidates"], "msg": "raw_text contains law citations but law_candidates is empty", "type": "soft_missing"})

    return errs


def _has_extra_forbidden(pyd_errs: List[Dict[str, Any]]) -> bool:
    for e in pyd_errs:
        t = e.get("type", "")
        if t == "extra_forbidden" or "extra_forbidden" in t:
            return True
    return False


def normalize_to_casegraph(tool_args: dict, doc: dict, raw_text: str) -> dict:
    """
    将 DeepSeek tool 输出对齐 graphs.py 的 CaseGraph：
    - 补齐 CaseGraph 必填字段（raw_text/case_id/source_type/meta）
    - 兜底 meta.procedure_stage（从 trial_level 推）
    - 所有 list 字段兜底为 []
    """
    out = dict(tool_args)

    doc_id = str(doc.get("doc_id", "")).strip() or "UNKNOWN_DOC"
    meta_in = doc.get("meta") or {}

    # 1) case_id / schema_version / source_type
    out.setdefault("case_id", doc_id)
    out.setdefault("schema_version", "0.1.0")
    # judgment 输入就强制 judgement
    out["source_type"] = "judgment"

    # 2) raw_text 必填：强制本地填（不让模型输出）
    out["raw_text"] = raw_text

    # 3) meta：补齐 & 映射
    m = dict(out.get("meta") or {})
    m.setdefault("jurisdiction", "CN")
    m.setdefault("language", "zh-CN")

    # court/case_type/cause
    m.setdefault("court", meta_in.get("court"))
    m.setdefault("case_type", doc.get("subtype") or meta_in.get("cause") or meta_in.get("case_system"))
    m.setdefault("cause", meta_in.get("cause"))

    # procedure_stage: trial_level → enum
    tl = (meta_in.get("trial_level") or "").strip()
    if tl in ("一审", "1", "First", "first_instance"):
        ps = "first_instance"
    elif tl in ("二审", "2", "Second", "second_instance"):
        ps = "second_instance"
    elif "再审" in tl or tl == "retrial":
        ps = "retrial"
    else:
        # 文书默认一审更合理；但如果你想严格，用 consultation 也行
        ps = "first_instance"
    m.setdefault("procedure_stage", ps)

    out["meta"] = m

    # 4) list 字段兜底
    for k in [
        "parties", "relationships", "claims", "facts",
        "evidence", "legal_issues", "law_candidates",
        "similar_judgments", "timeline"
    ]:
        if k not in out or out[k] is None:
            out[k] = []
        if not isinstance(out[k], list):
            out[k] = []

    # 5) summary 可空
    out.setdefault("summary", None)

    return out


def pydantic_validate_casegraph(
    *,
    adapter: Optional[TypeAdapter],
    model_fields: Optional[set[str]],
    payload: Dict[str, Any],   # tool_args
    raw_text: str,
    doc: Dict[str, Any],
    keep_raw_text: bool,
) -> Tuple[bool, Optional[Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: ok, model_obj, errors, validated_input_used
    """
    if adapter is None:
        # 不做校验就直接认为 ok
        return True, None, [], payload

    # ✅ 关键：先把 tool 输出对齐 CaseGraph
    candidate = normalize_to_casegraph(payload, doc, raw_text)

    try:
        obj = adapter.validate_python(
            candidate,
            context={"raw_text": raw_text, "doc": doc},  # v2 支持 context，validators 里可用 info.context
        )
        errs: List[Dict[str, Any]] = []
    except ValidationError as e:
        obj = None
        errs = e.errors()  # 结构化错误列表，最适合喂回去做 repair
        return False, None, errs, candidate

    # ✅ 输出时默认别带全文
    if not keep_raw_text and isinstance(obj, object):
        # 这里不改 obj，本来输出阶段就会 strip；如果你 output_mode=validated，再 strip 一次即可
        pass

    return True, obj, [], candidate


def sanitize_output_payload(payload: Dict[str, Any], doc_id: str, keep_raw_text: bool) -> Dict[str, Any]:
    out = dict(payload)
    if "doc_id" not in out:
        out["doc_id"] = doc_id
    if not keep_raw_text and "raw_text" in out:
        out["raw_text"] = ""
    return out


# --------------------------- repair prompting ---------------------------

def build_system_prompt() -> str:
    return (
        "你是一个【结构化信息抽取引擎】。你的唯一任务：从给定裁判文书中抽取结构化 JSON，"
        "并且【只能】以工具调用（tool_calls）形式输出。\n"
        "\n"
        "=== 绝对输出规则（违反任意一条都算失败）===\n"
        "0) 你必须且只能输出【一次】工具调用（one tool_call），不要输出任何普通文本、解释、前后缀、换行提示。\n"
        "1) 工具参数必须是【严格 JSON】（function arguments）：\n"
        "   - 只能使用双引号；不要使用单引号\n"
        "   - 不要使用 Markdown code fence\n"
        "   - 不要输出 NaN/Infinity\n"
        "   - 不要输出注释\n"
        "2) 【严禁泄露原文】：不得包含原文全文或大段复述；summary 允许短摘要（<=120字）或为 null。\n"
        "3) 【严禁输出 schema 未定义字段】：不要输出 raw_text / start_anchor / end_anchor / text_digest / head_200 等任何定位或摘要字段。\n"
        "4) 必须输出 schema 要求的所有【顶层字段】，且类型严格正确：\n"
        "   - parties / relationships / claims / facts / evidence / legal_issues / law_candidates / similar_judgments / timeline 必须是数组（没有就 []）。\n"
        "   - meta 必须是对象，且包含 schema 要求的必填键。\n"
        "   - 任何字段如果原文没有或无法确定：按字段类型用 null / \"unknown\" / \"\" / []（不要瞎猜）。\n"
        "\n"
        "=== 质量与一致性规则（强制执行）===\n"
        "A) 不编造：只抽取原文明确支持的信息。禁止推理补全未出现的事实（例如具体金额、具体速度、是否酒驾等）。\n"
        "B) 枚举值：如果 schema 对某字段有限定枚举（enum），必须使用【枚举允许的精确值】。\n"
        "   - 不确定枚举时：优先选 \"other\"（若 enum 里存在），否则用 null。\n"
        "C) ID 一致性：\n"
        "   - parties[].id 必须唯一（建议 p1/p2/...）；relationships/claims/facts/timeline 中引用的 *_party_id 必须指向已存在的 parties.id。\n"
        "   - evidence[].id 必须唯一（e1/e2/...）；facts[].evidence_ids 只能引用已存在的 evidence.id。\n"
        "   - legal_issues[].related_fact_ids/related_claim_ids 只能引用已存在的 facts/claims 的 id。\n"
        "D) 去冗余：避免把同一句话拆成大量重复 facts；能合并就合并。\n"
        "E) 规范化：\n"
        "   - 日期尽量用 YYYY-MM-DD；时间尽量用 YYYY-MM-DD HH:MM；仅有“某年某月某日”就转成 YYYY-MM-DD。\n"
        "   - 法条写法保持原文条号（如“第一百三十三条”“第六十七条第一款”），不要自作主张改写。\n"
        "   - 人名/机构名按原文；不要额外脱敏替换（除非原文本来就匿名）。\n"
        "\n"
        "=== 内容选择优先级（建议你按这个抽取）===\n"
        "1) meta：法院、案号、案由/罪名、审级、文书类型、裁判日期、语种、法域。\n"
        "2) parties：公诉机关/原告/被告/被害人/第三人/法院等核心主体（优先<=10个）。\n"
        "3) facts：构成要件相关事实、关键时间地点、损害后果、责任认定、到案/自首线索、赔偿谅解等量刑情节。\n"
        "4) evidence：原文列明的证据类型与名称（不要杜撰证据内容）；强度与争议可保守填。\n"
        "5) legal_issues：法院认定的核心争点/要件判断/量刑情节（尽量<=5条）。\n"
        "6) law_candidates：判决引用法条为主；没有明确引用时可留空 []。\n"
        "7) timeline：只放“关键节点”（事故/到案/立案或取保/起诉/审理/判决等），不要为了凑条数硬拆。\n"
        "\n"
        "=== 数量上限（硬限制，超过也算失败）===\n"
        "parties<=10, relationships<=20, claims<=15, facts<=25, evidence<=25, "
        "legal_issues<=10, law_candidates<=25, timeline<=30。\n"
        "\n"
        "=== 校验自检（在脑中完成，不要输出自检过程）===\n"
        "在输出前，确保：\n"
        "1) 顶层字段齐全；2) 所有数组字段都是数组；3) 不存在 schema 外字段；\n"
        "4) 所有引用 ID 都能解析到对象；5) JSON 可被严格解析；6) 未泄露原文大段文本。\n"
        "\n"
        "=== 纠错回合规则（如果你收到校验错误/repair 指令）===\n"
        "你必须：只输出新的 tool_call JSON；不要解释；不要复述错误；只做最小修改让其通过校验。\n"
    )


def build_user_prompt(doc: Dict[str, Any], text: str) -> str:
    doc_id = str(doc.get("doc_id", ""))
    meta = doc.get("meta") or {}
    chunk_ids = meta.get("chunk_ids") or doc.get("chunk_ids") or []
    if not isinstance(chunk_ids, list):
        chunk_ids = []

    # 把你已有的 meta 明确喂给模型（减少它猜测）
    meta_hint_keys = [
        "case_number", "court", "case_system", "judgment_date",
        "trial_level", "doctype_detail", "cause"
    ]
    meta_hint = {k: meta.get(k) for k in meta_hint_keys if k in meta and meta.get(k) is not None}

    return (
        f"doc_id={doc_id}\n"
        f"doc_type={doc.get('doc_type','')}\n"
        f"known_meta={json.dumps(meta_hint, ensure_ascii=False)}\n"
        f"chunk_ids={chunk_ids}\n"
        "请根据下面原文抽取结构化信息（只输出工具调用）。\n"
        "-----BEGIN TEXT-----\n"
        f"{text}\n"
        "-----END TEXT-----\n"
    )


def build_repair_user_prompt(
    doc: Dict[str, Any],
    raw_text: str,
    prev_json: Dict[str, Any],
    errors: List[Dict[str, Any]],
) -> str:
    """
    repair：把“错误列表 + 上次 JSON”喂回去，让模型只修 JSON
    """
    doc_id = str(doc.get("doc_id", ""))
    err_txt = json.dumps(errors[:80], ensure_ascii=False, indent=2)
    prev_txt = json.dumps(prev_json, ensure_ascii=False, indent=2)

    # 原文提示片段：保持短，避免模型复述全文
    head = raw_text[:900]
    tail = raw_text[-600:] if len(raw_text) > 600 else ""
    hint = head + ("\n...\n" + tail if tail else "")

    return (
        f"doc_id={doc_id}\n"
        "你的上一次工具输出未通过本地 Pydantic 校验，请修复 JSON 以通过校验。\n"
        "要求：\n"
        "1) 只能输出工具调用（tool arguments），不要输出解释性文本。\n"
        "2) 不要输出原文全文，不要输出 raw_text。\n"
        "3) 不要输出 schema 中不存在的字段（例如 start_anchor/end_anchor/text_digest 等）。\n"
        "4) 按错误提示修正字段名/类型/枚举值；原文明确提到的信息不要留空。\n\n"
        f"【本地校验错误】\n{err_txt}\n\n"
        f"【你上次输出的 JSON】\n{prev_txt}\n\n"
        f"【原文提示片段（非全文）】\n{hint}\n"
    )


# --------------------------- main extraction per doc ---------------------------

def extract_one_doc(
    *,
    client: OpenAI,
    model: str,
    base_tool: Dict[str, Any],
    doc: Dict[str, Any],
    max_tokens: int,
    max_calls: int,
    max_retry_sleep: float,
    temperature: float,
    strict: bool,
    print_attempt_errors: bool,
    print_raw_on_fail: bool,
    raw_preview_chars: int,
    # pydantic validation
    adapter: Optional[TypeAdapter],
    model_fields: Optional[set[str]],
    output_mode: str,   # "original" | "validated"
    keep_raw_text: bool,
    # thinking
    thinking: bool,
    thinking_budget: int
) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    总调用次数受 max_calls 控制（含 repair）。
    Returns: (ok, output_payload, debug)
    """
    doc_id = str(doc.get("doc_id", ""))
    text = doc.get("raw_text") or doc.get("text") or doc.get("text_head") or ""
    if not isinstance(text, str):
        text = str(text)

    # tool config
    tool = set_tool_strict(base_tool, strict=strict)
    fn_name = tool["function"]["name"]
    tool_choice_named = {"type": "function", "function": {"name": fn_name}}
    tool_choice_required = "required"

    system = build_system_prompt()
    user = build_user_prompt(doc=doc, text=text)

    debug_last: Dict[str, Any] = {}
    calls_used = 0
    repairs_used = 0
    validation_ok: Optional[bool] = None
    validation_errors: List[Dict[str,Any]] = []

    def backoff_sleep(k: int):
        # k = calls_used index (1-based)
        sleep_s = min(max_retry_sleep, 0.8 * (1.6 ** max(0, k - 1))) + random.random() * 0.4
        time.sleep(sleep_s)
        return sleep_s

    def print_fail(tag: str, r: ExtractResult, reason: str):
        if not print_attempt_errors:
            return
        print(f"\n[ATTEMPT FAIL] doc_id={doc_id} tag={tag} finish_reason={r.finish_reason} "
              f"has_tool_calls={r.has_tool_calls} tool_name={r.tool_name}\nReason:\n{reason}")
        if print_raw_on_fail and r.raw_response is not None:
            raw_preview = json.dumps(r.raw_response, ensure_ascii=False, indent=2)
            raw_preview = raw_preview[: max(200, int(raw_preview_chars))]
            print(f"Raw response preview:\n{raw_preview}")

    # 先做 1 次 named（更稳定触发指定工具），不行再 required
    extract_steps = [
        ("nonstrict+named", tool_choice_named),
        ("nonstrict+required", tool_choice_required),
    ]

    for tag, tool_choice in extract_steps:
        if calls_used >= max_calls:
            break

        calls_used += 1
        try:
            r = call_deepseek_once(
                client=client,
                model=model,
                tool=tool,
                tool_choice=tool_choice,
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking,
                thinking_budget=thinking_budget,
            )
        except Exception as e:
            debug_last = {"attempt": calls_used, "stage": tag, "ok": False, "err": f"exception: {e}"}
            sleep_s = backoff_sleep(calls_used)
            log.warning(f"doc_id={doc_id} call={calls_used} stage={tag} exception; sleep={sleep_s:.2f}s")
            continue

        debug_last = {
            "attempt": calls_used,
            "stage": tag,
            **r.__dict__,
        }

        if not r.ok or not isinstance(r.args, dict):
            print_fail(tag, r, r.err or "unknown error")
            sleep_s = backoff_sleep(calls_used)
            log.warning(f"doc_id={doc_id} call={calls_used} stage={tag} failed; sleep={sleep_s:.2f}s reason={r.err}")
            continue

        # hard guard: never allow the model to put full raw_text into output
        tool_args = dict(r.args)
        if "raw_text" in tool_args:
            tool_args["raw_text"] = ""

        if "doc_id" not in tool_args:
            tool_args["doc_id"] = doc_id

        # --- pydantic validate; if fails -> repair loop ---
        ok, model_obj, errs, validated_input = pydantic_validate_casegraph(
            adapter=adapter,
            model_fields=model_fields,
            payload=tool_args,
            raw_text=text,
            doc=doc,
            keep_raw_text=keep_raw_text,
        )

        validation_ok = bool(ok)
        validation_errors = errs[:80] if isinstance(errs, list) else []
        debug_last["calls_used"] = calls_used
        debug_last["repairs_used"] = repairs_used
        debug_last["validation_ok"] = validation_ok
        if not validation_ok:
            debug_last["validation_errors"] = validation_errors
        else:
            debug_last.pop("validation_errors", None)

        if adapter is None:
            debug_last["validation_skipped"] = True

        if ok:
            if output_mode == "validated" and model_obj is not None:
                out = model_obj.model_dump(mode="json", exclude_none=True)
                if not keep_raw_text and "raw_text" in out:
                    out["raw_text"] = ""
                if "doc_id" not in out:
                    out["doc_id"] = doc_id
                return True, out, debug_last
            else:
                out = sanitize_output_payload(tool_args, doc_id=doc_id, keep_raw_text=keep_raw_text)
                return True, out, debug_last

        # repair loop within remaining call budget
        prev = tool_args
        last_errs = errs
        while calls_used < max_calls:
            calls_used += 1
            repairs_used += 1

            repair_user = build_repair_user_prompt(
                doc=doc,
                raw_text=text,
                prev_json=prev,
                errors=last_errs,
            )

            try:
                rr = call_deepseek_once(
                    client=client,
                    model=model,
                    tool=tool,
                    tool_choice=tool_choice_required,
                    system=system,   # 仍然使用同一 system 约束
                    user=repair_user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                )
            except Exception as e:
                debug_last = {"attempt": calls_used, "stage": "repair", "ok": False, "err": f"exception: {e}"}
                sleep_s = backoff_sleep(calls_used)
                log.warning(f"doc_id={doc_id} call={calls_used} stage=repair exception; sleep={sleep_s:.2f}s")
                continue

            debug_last = {"attempt": calls_used, "stage": "repair", **rr.__dict__, "prev_errors": last_errs[:80]}

            if not rr.ok or not isinstance(rr.args, dict):
                print_fail("repair", rr, rr.err or "unknown repair error")
                sleep_s = backoff_sleep(calls_used)
                log.warning(f"doc_id={doc_id} call={calls_used} stage=repair failed; sleep={sleep_s:.2f}s reason={rr.err}")
                prev = prev  # keep old
                last_errs = [{"loc": ["repair"], "msg": "repair produced no tool_calls", "type": "repair_failed"}]
                continue

            cand = dict(rr.args)
            if "raw_text" in cand:
                cand["raw_text"] = ""
            if "doc_id" not in cand:
                cand["doc_id"] = doc_id

            ok2, model_obj2, errs2, _ = pydantic_validate_casegraph(
                adapter=adapter,
                model_fields=model_fields,
                payload=cand,
                raw_text=text,
                doc=doc,
                keep_raw_text=keep_raw_text,
            )
            
            validation_ok = bool(ok2)
            validation_errors = errs2[:80] if isinstance(errs2, list) else []
            debug_last["calls_used"] = calls_used
            debug_last["repairs_used"] = repairs_used
            debug_last["validation_ok"] = validation_ok
            if not validation_ok:
                debug_last["validation_errors"] = validation_errors
            else:
                debug_last.pop("validation_errors", None)
            if adapter is None:
                debug_last["validation_skipped"] = True

            if ok2:
                if output_mode == "validated" and model_obj2 is not None:
                    out = model_obj2.model_dump(mode="json", exclude_none=True)
                    if not keep_raw_text and "raw_text" in out:
                        out["raw_text"] = ""
                    if "doc_id" not in out:
                        out["doc_id"] = doc_id
                    return True, out, debug_last
                else:
                    out = sanitize_output_payload(cand, doc_id=doc_id, keep_raw_text=keep_raw_text)
                    return True, out, debug_last

            prev = cand
            last_errs = errs2

        # 如果 repair 用完预算，继续尝试下一种 extract 步（required），否则直接失败返回
        #（这里不 return，for-loop 会继续）
    if isinstance(debug_last, dict):
        debug_last.setdefault("calls_used", calls_used)
        debug_last.setdefault("repairs_used", repairs_used)
        debug_last.setdefault("validation_ok", validation_ok)
        if validation_errors:
            debug_last.setdefault("validation_errors", validation_errors)
        if adapter is None:
            debug_last.setdefault("validation_skipped", True)

    return False, None, (debug_last or {"err": "unknown failure"})


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in", dest="in_path", required=True, help="input jsonl")
    ap.add_argument("--out", dest="out_path", required=True, help="output jsonl")
    ap.add_argument("--tool", dest="tool_path", required=True, help="tool json path")

    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com/beta")

    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-calls", type=int, default=3, help="total model calls per doc (extract + repair)")
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--resume", action="store_true")

    # output / errors
    ap.add_argument("--err", dest="err_path", default="", help="errors jsonl path; default: <out>.errors.jsonl; set '-' to disable file")
    ap.add_argument("--dump-fail-response", action="store_true")

    # debugging prints
    ap.add_argument("--print-attempt-errors", action="store_true")
    ap.add_argument("--print-raw-on-fail", action="store_true")
    ap.add_argument("--raw-preview-chars", type=int, default=2500)

    # strict toggle (default off per your request)
    ap.add_argument("--strict", action="store_true", help="enable strict function calling (default off)")

    # pydantic validation
    ap.add_argument("--casegraph-import", default="schemas.graphs:CaseGraph", help="Pydantic model import path")
    ap.add_argument("--no-pydantic", action="store_true", help="disable pydantic validation/repair")
    ap.add_argument("--output-mode", choices=["original", "validated"], default="original",
                    help="write original tool args or pydantic validated dump")
    ap.add_argument("--keep-raw-text", action="store_true", help="keep raw_text in output (default: strip)")

    # thinking toggle (optional)
    ap.add_argument("--thinking", action="store_true", help="enable DeepSeek thinking mode via extra_body")
    ap.add_argument("--thinking-budget", type=int, default=0, help="optional budget_tokens for thinking")

    ap.add_argument("--max-retry-sleep", type=float, default=6.0, help="max backoff sleep seconds")
    ap.add_argument("--print-ok", action="store_true",
                    help="print per-doc success line to terminal (uses tqdm.write)")

    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if args.err_path.strip() == "-":
        err_path = None
    elif args.err_path.strip():
        err_path = Path(args.err_path.strip())
    else:
        err_path = Path(str(out_path) + ".errors.jsonl")

    tool = load_tool(Path(args.tool_path))
    log.info(f"tool loaded: name={tool['function']['name']} strict={tool['function'].get('strict', None)}")

    done = set()
    if args.resume:
        done = load_done_doc_ids(out_path, err_path)
        log.info(f"already done: {len(done)}")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env DEEPSEEK_API_KEY")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    adapter: Optional[TypeAdapter] = None
    model_fields: Optional[set[str]] = None
    if not args.no_pydantic:
        cg_type = load_casegraph_type(args.casegraph_import)
        adapter = TypeAdapter(cg_type)
        model_fields = get_model_fields(cg_type)
        log.info(f"pydantic validation enabled: {args.casegraph_import} fields={len(model_fields) if model_fields else 'unknown'}")

    ok = 0
    fail = 0

    rows = list(read_jsonl(in_path))
    for doc in tqdm(rows, desc="extract"):
        doc_id = str(doc.get("doc_id", ""))
        if args.resume and doc_id in done:
            continue

        success, out_obj, debug = extract_one_doc(
            client=client,
            model=args.model,
            base_tool=tool,
            doc=doc,
            max_tokens=args.max_tokens,
            max_calls=args.max_calls,
            max_retry_sleep=args.max_retry_sleep,
            temperature=float(args.temperature),
            strict=bool(args.strict),
            print_attempt_errors=bool(args.print_attempt_errors),
            print_raw_on_fail=bool(args.print_raw_on_fail),
            raw_preview_chars=int(args.raw_preview_chars),
            adapter=adapter,
            model_fields=model_fields,
            output_mode=args.output_mode,
            keep_raw_text=bool(args.keep_raw_text),
            thinking=bool(args.thinking),
            thinking_budget=int(args.thinking_budget)
        )

        if success and out_obj is not None:
            append_jsonl(out_path, out_obj)
            ok += 1
            if args.print_ok: 
                calls_used = debug.get("calls_used", "?") if isinstance(debug, dict) else "?"
                repairs_used = debug.get("repairs_used", "?") if isinstance(debug, dict) else "?"
                val_ok = debug.get("validation_ok", "?") if isinstance(debug, dict) else "?"
                tqdm.write(f"[OK] doc_id={doc_id} validation={val_ok} calls={calls_used} repairs={repairs_used}")
        else:
            fail += 1
            rec = {
                "doc_id": doc_id,
                "doc_type": doc.get("doc_type"),
                "error": debug.get("err") if isinstance(debug, dict) else str(debug),
                "finish_reason": debug.get("finish_reason") if isinstance(debug, dict) else None,
                "has_tool_calls": debug.get("has_tool_calls") if isinstance(debug, dict) else None,
                "tool_name": debug.get("tool_name") if isinstance(debug, dict) else None,
                "content_preview": debug.get("content_preview") if isinstance(debug, dict) else "",
                "debug": debug,
            }
            if args.dump_fail_response and isinstance(debug, dict) and debug.get("raw_response"):
                rec["raw_response"] = debug["raw_response"]

            if err_path is not None:
                append_jsonl(err_path, rec)
            else:
                # 如果禁用了 err 文件，就直接打印到终端
                print(json.dumps(rec, ensure_ascii=False))

    log.info(f"[DONE] ok={ok} fail={fail} out={out_path} err={err_path if err_path else '-'}")


if __name__ == "__main__":
    main()
