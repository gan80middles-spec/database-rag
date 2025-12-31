# -*- coding: utf-8 -*-
"""
deepseek_contractgraph.py

- 输入：JSONL（每行一个合同文档，至少包含 contract_id/doc_id + raw_text + meta.contract_type）
- 调用：DeepSeek(OpenAI-compatible) tool_calls 抽取合同结构
- 本地：补齐 raw_text（Pydantic required），按 PRESENCE_SCHEMAS 生成 presence_summary（可选）
- 校验：Pydantic v2 TypeAdapter.validate_python
- 失败：自动 repair（最多 --max-calls 次）
- 输出：JSONL（默认不落 raw_text，避免体积暴涨；可用 --keep-raw-text 打开）

依赖：
  pip install openai pydantic tqdm
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter
from tqdm import tqdm


# -------------------------
# ToolSpec
# -------------------------

@dataclasses.dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]


# -------------------------
# IO utils
# -------------------------

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_ids(path: str, id_key: str) -> set[str]:
    done = set()
    if not os.path.exists(path):
        return done
    for row in read_jsonl(path):
        if isinstance(row, dict) and id_key in row and isinstance(row[id_key], str):
            done.add(row[id_key])
    return done


# -------------------------
# import helper
# -------------------------

def import_symbol(path: str) -> Any:
    """
    支持两种写法：
      - pkg.module:Symbol
      - pkg.module.Symbol
    """
    if ":" in path:
        mod, sym = path.split(":", 1)
    else:
        mod, sym = path.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, sym)


# -------------------------
# tool loader
# -------------------------

def load_tool(path: str) -> ToolSpec:
    obj = json.load(open(path, "r", encoding="utf-8"))

    # 兼容两种格式：
    # A) {name, description, schema}
    # B) OpenAI tools wrapper: {"type":"function","function":{"name","description","parameters":...}}
    if obj.get("type") == "function" and "function" in obj:
        fn = obj["function"]
        name = fn["name"]
        desc = fn.get("description", "")
        schema = fn.get("parameters") or fn.get("schema") or {}
        return ToolSpec(name=name, description=desc, schema=schema)

    name = obj["name"]
    desc = obj.get("description", "")
    schema = obj.get("schema") or obj.get("parameters") or obj.get("json_schema") or {}
    return ToolSpec(name=name, description=desc, schema=schema)


def to_openai_tool(tool: ToolSpec) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema,
        },
    }


# -------------------------
# prompts
# -------------------------

def build_system_prompt() -> str:
    # 风格沿用你 casegraph：强约束 + tool_calls-only + 不输出 raw_text
    return (
        "你是一个结构化信息抽取引擎。\n"
        "你必须且只能通过工具调用（tool_calls）输出结果，不要输出普通文本。\n"
        "工具参数必须是严格 JSON（function arguments），不要使用 Markdown code fence。\n"
        "\n"
        "重要约束：\n"
        "1) 严禁在输出中包含合同原文全文或大段复述；不要输出 raw_text 字段（raw_text 会在本地由程序补齐）。\n"
        "2) presence_summary 将由本地 PRESENCE_SCHEMAS 计算生成；你可以不输出 presence_summary。\n"
        "3) 如果某信息原文没有或无法确定：允许为 null/unknown/空字符串/空数组（遵循字段类型）。\n"
        "4) parties/definitions/clauses/risks/attachments 必须是数组（没有就 []）。\n"
        "5) clauses.id 必须唯一，且 risks.source_clause_ids 必须引用 clauses.id。\n"
        "6) clauses[].text 不得粘贴原文；如 schema 必填，则只给 <=200 字“摘要/改写”，不要包含引号/表格/附件。\n"
        "7) clauses 数量上限（比如 200）；risks 数量上限（比如 80）。超出就只保留高风险/关键条款。\n"
        "8) 不要输出任何大段原文（包括附件目录/签字页/表格），否则视为失败。"
    )


def build_user_prompt(doc: Dict[str, Any]) -> str:
    contract_id = doc.get("doc_id") or doc.get("contract_id") or doc.get("_id") or "UNKNOWN"
    meta = doc.get("meta") or {}
    contract_type = (
        meta.get("contract_type")
        or doc.get("contract_type")
        or "other"
    )

    raw_text = doc.get("raw_text") or doc.get("text") or ""
    return (
        f"请抽取合同图谱（ContractGraph）。\n"
        f"- contract_id: {contract_id}\n"
        f"- contract_type: {contract_type}\n"
        f"\n"
        f"合同原文如下（仅用于理解，不要在输出中复述全文）：\n"
        f"{raw_text}"
    )


def build_repair_prompt(validation_errors: List[Dict[str, Any]], last_args: Dict[str, Any]) -> str:
    # 把错误列表压成紧凑文本（对齐你 casegraph 的“errors->repair prompt”思路）
    err_lines = []
    for e in validation_errors[:50]:
        loc = ".".join(str(x) for x in e.get("loc", []))
        msg = e.get("msg", "")
        typ = e.get("type", "")
        err_lines.append(f"- loc={loc} type={typ} msg={msg}")
    err_text = "\n".join(err_lines)

    compact = prune_for_repair(last_args, max_chars=12000)

    return (
        "上一次工具输出未通过本地 schema 校验。请基于以下错误信息修复工具输出。\n"
        "要求：仍然必须仅通过 tool_calls 输出修复后的 JSON 参数；不要输出 raw_text；不要输出合同全文。\n"
        "\n"
        "[VALIDATION_ERRORS]\n"
        f"{err_text}\n"
        "\n"
        "[LAST_ARGS_COMPACT]\n"
        f"{json.dumps(compact, ensure_ascii=False)}\n"
    )


def build_json_repair_prompt(finish_reason: Optional[str], bad_args_path: Optional[str]) -> str:
    info_lines = []
    if finish_reason:
        info_lines.append(f"- finish_reason: {finish_reason}")
    if bad_args_path:
        info_lines.append(f"- bad_args_path: {bad_args_path}")
    info_text = "\n".join(info_lines) if info_lines else "(no extra info)"

    return (
        "上一次工具输出的 function.arguments 不是合法 JSON（可能被截断或格式错误）。\n"
        "请重新输出完整且严格合法的 JSON（仅 tool_calls），不要输出普通文本。\n"
        "强制要求：不要输出合同原文；clauses[].text 只给 <=200 字摘要/改写；"
        "clauses<=200、risks<=80；字段类型必须符合 schema。\n"
        "\n"
        "[DEBUG]\n"
        f"{info_text}\n"
    )


def prune_for_repair(obj: Dict[str, Any], max_chars: int = 12000) -> Dict[str, Any]:
    """
    避免把 clause.text 之类塞太长给 repair。保留结构与 id 引用即可。
    """
    x = copy.deepcopy(obj)

    # clauses.text 截断
    clauses = x.get("clauses")
    if isinstance(clauses, list):
        for c in clauses:
            if isinstance(c, dict) and isinstance(c.get("text"), str) and len(c["text"]) > 400:
                c["text"] = c["text"][:400] + "…"

    s = json.dumps(x, ensure_ascii=False)
    if len(s) <= max_chars:
        return x

    # 再更激进：只保留 clauses.id/title
    if isinstance(clauses, list):
        x["clauses"] = [
            {"id": c.get("id"), "title": c.get("title")}
            for c in clauses
            if isinstance(c, dict)
        ]
    return x


# -------------------------
# Presence summary (local)
# -------------------------

def normalize_contract_type(v: Any) -> str:
    # 支持 Enum / str
    if v is None:
        return "other"
    if isinstance(v, str):
        return v
    # Enum-like
    name = getattr(v, "value", None)
    if isinstance(name, str):
        return name
    return str(v)


def load_presence_schemas(spec: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    spec 例子：schemas.presence_schemas:PRESENCE_SCHEMAS
    期望返回 dict[str, list[dict]] or dict[str, list[Any]]
    """
    if not spec:
        return None
    try:
        obj = import_symbol(spec)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def detect_presence_for_item(item: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    一个“通用、可扩展”的要素检测器：
    - 支持 item.match.keywords_any / keywords_all / regex_any / tags_any
    - 返回 (present, source_clause_ids)
    """
    match = item.get("match") or {}
    kw_any = match.get("keywords_any") or []
    kw_all = match.get("keywords_all") or []
    rgx_any = match.get("regex_any") or []
    tags_any = match.get("tags_any") or []

    # 预编译正则
    rgx_list = []
    for p in rgx_any:
        try:
            rgx_list.append(re.compile(p))
        except re.error:
            continue

    hit_clause_ids: List[str] = []
    for c in clauses:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        text = (c.get("text") or "") if isinstance(c.get("text"), str) else ""
        title = (c.get("title") or "") if isinstance(c.get("title"), str) else ""
        blob = f"{title}\n{text}"

        # tags
        tags = c.get("tags") if isinstance(c.get("tags"), list) else []
        if tags_any and any(t in tags for t in tags_any):
            if isinstance(cid, str):
                hit_clause_ids.append(cid)
            continue

        # keywords_all
        if kw_all and all(k in blob for k in kw_all):
            if isinstance(cid, str):
                hit_clause_ids.append(cid)
            continue

        # keywords_any
        if kw_any and any(k in blob for k in kw_any):
            if isinstance(cid, str):
                hit_clause_ids.append(cid)
            continue

        # regex_any
        if rgx_list and any(r.search(blob) for r in rgx_list):
            if isinstance(cid, str):
                hit_clause_ids.append(cid)
            continue

    hit_clause_ids = list(dict.fromkeys(hit_clause_ids))  # 去重保序
    present = len(hit_clause_ids) > 0
    return present, hit_clause_ids


def build_presence_summary_local(
    contract_type: str,
    clauses: List[Dict[str, Any]],
    presence_schemas: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    输出结构对齐你的 Pydantic：
      ContractPresenceSummary(contract_type, items=[PresenceItem...])
    PresenceItem: id/label/required/present/source_clause_ids
    :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
    """
    if not presence_schemas:
        return None

    ct = normalize_contract_type(contract_type)
    schema_items = presence_schemas.get(ct) or presence_schemas.get("other") or []
    if not isinstance(schema_items, list):
        return None

    items_out: List[Dict[str, Any]] = []
    for it in schema_items:
        # 允许 it 是 dict 或 Pydantic/BaseModel（有 model_dump）
        if hasattr(it, "model_dump"):
            it = it.model_dump()
        if not isinstance(it, dict):
            continue

        pid = it.get("id")
        label = it.get("label") or pid or ""
        required = bool(it.get("required", False))

        present, src = detect_presence_for_item(it, clauses)

        items_out.append({
            "id": pid or "",
            "label": str(label),
            "required": required,
            "present": present,
            "source_clause_ids": src,
        })

    return {
        "contract_type": ct,
        "items": items_out,
    }


# -------------------------
# Normalize -> validate
# -------------------------

def normalize_to_contractgraph(
    doc: Dict[str, Any],
    args_obj: Dict[str, Any],
    presence_schemas: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    把 LLM 工具输出补齐到能过你 graphs.py 的 ContractGraph 校验：
      - contract_id: 必填
      - schema_version: 必填（默认）
      - raw_text: 必填（由 doc 补齐）:contentReference[oaicite:12]{index=12}
      - meta: 必填
      - parties/definitions/clauses/risks/attachments: 必须是数组（无则[]）
      - presence_summary: 可选（本地算）:contentReference[oaicite:13]{index=13}
    """
    out = copy.deepcopy(args_obj) if isinstance(args_obj, dict) else {}

    # ---- contract_id
    doc_id = doc.get("doc_id")
    if doc_id:
        out["contract_id"] = str(doc_id)
    elif not isinstance(out.get("contract_id"), str) or not out["contract_id"]:
        cid = doc.get("contract_id") or doc.get("_id")
        out["contract_id"] = str(cid) if cid else "UNKNOWN"

    # ---- schema_version
    if not isinstance(out.get("schema_version"), str) or not out["schema_version"]:
        out["schema_version"] = "0.1.0"

    # ---- raw_text (required in your Pydantic)
    raw_text = doc.get("raw_text") or doc.get("text") or ""
    out["raw_text"] = raw_text if isinstance(raw_text, str) else str(raw_text)

    # ---- meta
    meta = out.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    doc_meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    if "contract_type" not in meta:
        meta["contract_type"] = doc_meta.get("contract_type") or doc.get("contract_type") or "other"
    if isinstance(out.get("contract_id"), str):
        base_contract_id = out["contract_id"].split("#", 1)[0]
        if base_contract_id:
            meta.setdefault("base_contract_id", base_contract_id)
        if "#" in out["contract_id"]:
            part = out["contract_id"].split("#", 1)[1]
            if part:
                meta.setdefault("part", part)
    out["meta"] = meta

    # ---- arrays
    for k in ["parties", "definitions", "clauses", "risks", "attachments"]:
        v = out.get(k)
        if not isinstance(v, list):
            out[k] = []

    party_type_map = {
        "organization": "company",
    }
    for party in out.get("parties", []):
        if not isinstance(party, dict):
            continue
        raw_type = party.get("type")
        if isinstance(raw_type, str) and raw_type:
            raw_normalized = raw_type.strip().lower()
            normalized_type = party_type_map.get(raw_normalized, raw_normalized)
        else:
            normalized_type = "other"
        if normalized_type not in {"natural_person", "company", "government", "other"}:
            normalized_type = "other"
        party["type"] = normalized_type

    seed = doc.get("seed_clauses") or []
    seed_map = {
        c.get("id"): c
        for c in seed
        if isinstance(c, dict) and isinstance(c.get("id"), str)
    }

    if seed and not out.get("clauses"):
        truncated_seed: List[Dict[str, Any]] = []
        for clause in seed[:200]:
            if not isinstance(clause, dict):
                continue
            clause_copy = copy.deepcopy(clause)
            text = clause_copy.get("text")
            if isinstance(text, str) and len(text) > 300:
                clause_copy["text"] = text[:300] + "…"
            truncated_seed.append(clause_copy)
        out["clauses"] = truncated_seed

    if seed_map:
        for c in out.get("clauses", []):
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            if cid in seed_map and not c.get("text"):
                seed_text = seed_map[cid].get("text")
                if isinstance(seed_text, str):
                    c["text"] = seed_text

        if not out.get("clauses"):
            ref_ids: set[str] = set()
            for r in out.get("risks", []):
                if not isinstance(r, dict):
                    continue
                for cid in r.get("source_clause_ids") or []:
                    if isinstance(cid, str):
                        ref_ids.add(cid)
            if ref_ids:
                out["clauses"] = [seed_map[cid] for cid in ref_ids if cid in seed_map]

    # ---- presence_summary (optional)
    if out.get("presence_summary") is None and presence_schemas:
        ct = out.get("meta", {}).get("contract_type", "other")
        clauses_for_presence = doc.get("seed_clauses")
        if not isinstance(clauses_for_presence, list) or not clauses_for_presence:
            clauses_for_presence = out.get("clauses", [])

        out["presence_summary"] = build_presence_summary_local(
            contract_type=str(ct),
            clauses=clauses_for_presence,
            presence_schemas=presence_schemas,
        )

    return out


def strip_for_output(obj: Dict[str, Any], keep_raw_text: bool) -> Dict[str, Any]:
    x = copy.deepcopy(obj)
    if not keep_raw_text:
        x.pop("raw_text", None)
    return x


def write_bad_args(raw_args: Any, contract_id: str, attempt: int) -> Optional[str]:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", contract_id or "UNKNOWN")
    bad_dir = Path("bad_args")
    bad_dir.mkdir(parents=True, exist_ok=True)
    path = bad_dir / f"{safe_id}.{attempt}.txt"
    try:
        with path.open("w", encoding="utf-8") as f:
            if isinstance(raw_args, str):
                f.write(raw_args)
            else:
                f.write(str(raw_args))
        return str(path)
    except OSError:
        return None


# -------------------------
# DeepSeek call
# -------------------------

def call_once(
    client: OpenAI,
    model: str,
    tool: ToolSpec,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    contract_id: str,
    attempt: int,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    tools = [to_openai_tool(tool)]
    tool_choice = {"type": "function", "function": {"name": tool.name}}

    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # 注意：不要传 thinking（你遇到过 Completions.create 不支持 thinking）
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    finish_reason = resp.choices[0].finish_reason
    msg = resp.choices[0].message

    if not getattr(msg, "tool_calls", None):
        raise RuntimeError("No tool_calls in response")

    tc = msg.tool_calls[0]
    fn = tc.function
    if fn.name != tool.name:
        raise RuntimeError(f"Tool name mismatch: got {fn.name}, expected {tool.name}")

    args = fn.arguments
    if isinstance(args, str):
        try:
            return json.loads(args), finish_reason, None
        except json.JSONDecodeError as je:
            bad_path = write_bad_args(args, contract_id, attempt)
            setattr(je, "finish_reason", finish_reason)
            setattr(je, "bad_args_path", bad_path)
            raise
    if isinstance(args, dict):
        return args, finish_reason, None
    # 某些兼容层可能返回 None/其他类型
    try:
        return json.loads(str(args)), finish_reason, None
    except json.JSONDecodeError as je:
        bad_path = write_bad_args(args, contract_id, attempt)
        setattr(je, "finish_reason", finish_reason)
        setattr(je, "bad_args_path", bad_path)
        raise


# -------------------------
# extract loop
# -------------------------

def extract_one(
    client: OpenAI,
    tool: ToolSpec,
    contractgraph_adapter: TypeAdapter,
    doc: Dict[str, Any],
    model: str,
    max_tokens: int,
    temperature: float,
    max_calls: int,
    presence_schemas: Optional[Dict[str, Any]],
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    debug_last: Dict[str, Any] = {
        "calls_used": 0,
        "repairs_used": 0,
        "validation_ok": False,
        "validation_errors": None,
        "finish_reason": None,
        "raw_args_path": None,
    }

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(doc)},
    ]

    last_args: Optional[Dict[str, Any]] = None
    cid = doc.get("doc_id") or doc.get("contract_id") or doc.get("_id") or "UNKNOWN"
    cid = str(cid)

    for attempt in range(max_calls):
        debug_last["calls_used"] += 1

        try:
            args_obj, finish_reason, raw_args_path = call_once(
                client=client,
                model=model,
                tool=tool,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                contract_id=cid,
                attempt=attempt + 1,
                extra_body=extra_body,
            )
            debug_last["finish_reason"] = finish_reason
            debug_last["raw_args_path"] = raw_args_path
        except json.JSONDecodeError as je:
            debug_last["finish_reason"] = getattr(je, "finish_reason", None)
            debug_last["raw_args_path"] = getattr(je, "bad_args_path", None)
            debug_last["validation_errors"] = [{"msg": str(je), "type": "json_decode_error"}]

            repair = build_json_repair_prompt(
                finish_reason=debug_last["finish_reason"],
                bad_args_path=debug_last["raw_args_path"],
            )
            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": repair},
            ]
            debug_last["repairs_used"] += 1
            continue
        last_args = args_obj

        normalized = normalize_to_contractgraph(
            doc=doc,
            args_obj=args_obj,
            presence_schemas=presence_schemas,
        )

        try:
            validated = contractgraph_adapter.validate_python(normalized)
            debug_last["validation_ok"] = True
            return validated, debug_last
        except ValidationError as ve:
            errs = ve.errors()
            debug_last["validation_errors"] = errs

            # 构造 repair prompt 并继续下一轮
            repair = build_repair_prompt(errs, normalized)
            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": repair},
            ]
            debug_last["repairs_used"] += 1
            continue
        except Exception as e:
            debug_last["validation_errors"] = [{"msg": str(e), "type": "unknown"}]
            break

    return None, debug_last


# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input jsonl")
    ap.add_argument("--out", dest="out_path", required=True, help="output jsonl")
    ap.add_argument("--tool", dest="tool_path", required=True, help="tool json path")

    ap.add_argument("--contractgraph-import", default="schemas.graphs:ContractGraph",
                    help="import path, e.g. schemas.graphs:ContractGraph")
    ap.add_argument("--presence-schemas", default="schemas.presence_schemas:PRESENCE_SCHEMAS",
                    help="import path, e.g. schemas.presence_schemas:PRESENCE_SCHEMAS; set empty to disable")

    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""),
                    help="OpenAI-compatible base_url (e.g. https://api.deepseek.com)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="API key (or set OPENAI_API_KEY)")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-chat"),
                    help="model name")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-calls", type=int, default=3)

    ap.add_argument("--keep-raw-text", action="store_true",
                    help="write raw_text into output jsonl (default: off)")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between docs")
    ap.add_argument("--print-attempt-errors", action="store_true")
    ap.add_argument("--print-ok", action="store_true")
    ap.add_argument("--debug-last", default=None, help="write per-doc debug jsonl")

    ap.add_argument("--extra-body-json", default=None,
                    help="optional extra_body json string to pass through (advanced)")

    args = ap.parse_args()

    if not args.base_url:
        raise SystemExit("Missing --base-url (or OPENAI_BASE_URL)")
    if not args.api_key:
        raise SystemExit("Missing --api-key (or OPENAI_API_KEY)")

    tool = load_tool(args.tool_path)

    ContractGraph = import_symbol(args.contractgraph_import)
    adapter = TypeAdapter(ContractGraph)

    presence_spec = args.presence_schemas.strip()
    presence_schemas = load_presence_schemas(presence_spec) if presence_spec else None

    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(str(out_path), id_key="contract_id")

    debug_path = Path(args.debug_last) if args.debug_last else None
    if debug_path:
        debug_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(args.in_path))
    pbar = tqdm(rows, desc="extract_contract_graph", total=len(rows))

    for doc in pbar:
        cid = doc.get("doc_id") or doc.get("contract_id") or doc.get("_id")
        cid = str(cid) if cid else "UNKNOWN"

        if cid in done:
            continue

        try:
            validated, dbg = extract_one(
                client=client,
                tool=tool,
                contractgraph_adapter=adapter,
                doc=doc,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_calls=args.max_calls,
                presence_schemas=presence_schemas,
                extra_body=extra_body,
            )

            if validated is None:
                if args.print_attempt_errors:
                    tqdm.write(f"[FAIL] contract_id={cid} errors={dbg.get('validation_errors')}")
                if debug_path:
                    append_jsonl(str(debug_path), {"contract_id": cid, "ok": False, **dbg})
                continue

            out_obj = validated.model_dump()
            out_obj = strip_for_output(out_obj, keep_raw_text=args.keep_raw_text)

            append_jsonl(str(out_path), out_obj)
            done.add(cid)

            if args.print_ok:
                tqdm.write(f"[OK] contract_id={cid} clauses={len(out_obj.get('clauses', []))} risks={len(out_obj.get('risks', []))}")

            if debug_path:
                append_jsonl(str(debug_path), {"contract_id": cid, "ok": True, **dbg})

        except Exception as e:
            if args.print_attempt_errors:
                tqdm.write(f"[EXCEPTION] contract_id={cid} err={type(e).__name__}: {e}")
            if debug_path:
                append_jsonl(str(debug_path), {"contract_id": cid, "ok": False, "exception": f"{type(e).__name__}: {e}"})

        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
