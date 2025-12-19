# deepseek_strictify_schema.py
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SUPPORTED_TYPES = {"object", "string", "number", "integer", "boolean", "array", "null"}

DROP_KEYS = {
    "$schema",
    "title",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "contentEncoding",
    "contentMediaType",
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
}

BOOL_HINT_WORDS = ("布尔", "boolean", "flag", "true/false", "True/False")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _walk_all_refs_and_fix_defs(root: Dict[str, Any]) -> None:
    """
    统一 $defs -> $def，并把 #/$defs/... ref 统一改成 #/$def/...
    """
    if "$defs" in root and "$def" not in root and isinstance(root["$defs"], dict):
        root["$def"] = root.pop("$defs")

    def rec(x: Any):
        if isinstance(x, dict):
            if "$ref" in x and isinstance(x["$ref"], str) and x["$ref"].startswith("#/$defs/"):
                x["$ref"] = x["$ref"].replace("#/$defs/", "#/$def/")
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(root)


def resolve_ref(root: Dict[str, Any], ref: str) -> Optional[Dict[str, Any]]:
    if not (isinstance(ref, str) and ref.startswith("#/")):
        return None
    parts = ref.lstrip("#/").split("/")
    cur: Any = root
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur if isinstance(cur, dict) else None


def infer_ref_type(root: Dict[str, Any], ref: str) -> str:
    target = resolve_ref(root, ref)
    if not target:
        return "object"
    t = target.get("type")
    if isinstance(t, str) and t in SUPPORTED_TYPES:
        return t
    return "object"


def drop_unwanted_keys(node: Dict[str, Any]) -> None:
    for k in list(node.keys()):
        if k in DROP_KEYS:
            node.pop(k, None)
    # default: null 也可能导致工具校验奇怪，保守移除
    if node.get("default") is None:
        node.pop("default", None)


def is_empty_object_schema(node: Dict[str, Any]) -> bool:
    if node.get("type") != "object":
        return False
    props = node.get("properties", None)
    return (props is None) or (isinstance(props, dict) and len(props) == 0)


def is_boolish_description(desc: str) -> bool:
    d = (desc or "").lower()
    return any(w.lower() in d for w in BOOL_HINT_WORDS)


def make_kv_list_schema(
    root: Dict[str, Any],
    value_schema: Optional[Dict[str, Any]] = None,
    *,
    prefer_bool: bool = False,
) -> Dict[str, Any]:
    """
    把 Dict / 开口 object / 空 object 转成 KV 列表（strict 最稳）
    """
    if not value_schema or not isinstance(value_schema, dict):
        value_schema = {"type": "boolean"} if prefer_bool else {"type": "string"}

    if is_empty_object_schema(value_schema):
        value_schema = {"type": "boolean"} if prefer_bool else {"type": "string"}

    if "$ref" in value_schema and "type" not in value_schema:
        value_schema["type"] = infer_ref_type(root, value_schema["$ref"])

    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "k": {"type": "string"},
                "v": value_schema,
            },
            "required": ["k", "v"],
            "additionalProperties": False,
        },
    }


def make_nullable(root: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    s = strictify_node(root, schema)
    if isinstance(s, dict) and "anyOf" in s and isinstance(s["anyOf"], list):
        if any(isinstance(x, dict) and x.get("type") == "null" for x in s["anyOf"]):
            return s

    if isinstance(s, dict) and "$ref" in s and "type" not in s:
        s["type"] = infer_ref_type(root, s["$ref"])

    return {"anyOf": [s, {"type": "null"}]}


def collapse_allof(root: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    allof = node.get("allOf")
    if not isinstance(allof, list) or not allof:
        return node

    items = [strictify_node(root, x) for x in allof]
    rest = copy.deepcopy(node)
    rest.pop("allOf", None)

    if len(items) == 1 and isinstance(items[0], dict):
        merged = copy.deepcopy(items[0])
        merged.update(rest)
        return merged

    all_obj = all(isinstance(x, dict) and x.get("type") == "object" for x in items)
    if all_obj:
        props: Dict[str, Any] = {}
        for x in items:
            props.update(x.get("properties", {}) or {})
        merged = {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),
            "additionalProperties": False,
        }
        if "description" in rest:
            merged["description"] = rest["description"]
        return merged

    return {"anyOf": items}


def strictify_node(root: Dict[str, Any], node: Any) -> Any:
    if isinstance(node, list):
        return [strictify_node(root, x) for x in node]
    if not isinstance(node, dict):
        return node

    node = copy.deepcopy(node)
    drop_unwanted_keys(node)

    # allOf
    if "allOf" in node:
        node = collapse_allof(root, node)
        if not isinstance(node, dict):
            return node
        drop_unwanted_keys(node)

    # type: ["x","null"] -> anyOf
    t = node.get("type")
    if isinstance(t, list):
        anyof = [{"type": tt} for tt in t if isinstance(tt, str) and tt in SUPPORTED_TYPES]
        if anyof:
            node.pop("type", None)
            node["anyOf"] = anyof

    # anyOf
    if "anyOf" in node and isinstance(node["anyOf"], list):
        new_any = []
        for opt in node["anyOf"]:
            opt2 = strictify_node(root, opt)
            if isinstance(opt2, dict) and "$ref" in opt2 and "type" not in opt2:
                opt2["type"] = infer_ref_type(root, opt2["$ref"])
            new_any.append(opt2)
        node["anyOf"] = new_any
        return node

    # array
    if node.get("type") == "array":
        if "items" in node:
            node["items"] = strictify_node(root, node["items"])
        return node

    # object
    if node.get("type") == "object":
        props = node.get("properties", None)
        desc = node.get("description", "") or ""
        prefer_bool = is_boolish_description(desc)

        # 空 / 开口 object -> KV list
        if props is None or (isinstance(props, dict) and len(props) == 0):
            ap = node.get("additionalProperties", None)
            v_schema = strictify_node(root, ap) if isinstance(ap, dict) else None
            return make_kv_list_schema(root, v_schema, prefer_bool=prefer_bool)

        if not isinstance(props, dict):
            return make_kv_list_schema(root, {"type": "boolean" if prefer_bool else "string"}, prefer_bool=prefer_bool)

        original_required = set(node.get("required") or []) if isinstance(node.get("required"), list) else set()

        new_props: Dict[str, Any] = {}
        for k, v in props.items():
            sv = strictify_node(root, v)

            # optional -> nullable（因为 strict 下我们会把 required 强制为全字段）
            if k not in original_required:
                if isinstance(sv, dict):
                    sv = make_nullable(root, sv)
                else:
                    sv = make_nullable(root, {"type": "string"})

            if isinstance(sv, dict) and "$ref" in sv and "type" not in sv:
                sv["type"] = infer_ref_type(root, sv["$ref"])

            new_props[k] = sv

        node["properties"] = new_props
        node["required"] = list(new_props.keys())
        node["additionalProperties"] = False

        # ⭐关键：如果 object 里还带 $def（通常只在根），也要递归 strictify
        if "$def" in node and isinstance(node["$def"], dict):
            node["$def"] = {k: strictify_node(root, v) for k, v in node["$def"].items()}

        return node

    # 仅 $ref
    if "$ref" in node:
        if "type" not in node:
            node["type"] = infer_ref_type(root, node["$ref"])
        return node

    # ⭐通用递归：避免像 $def 这种“不是 object properties”的子树漏处理
    for k, v in list(node.items()):
        node[k] = strictify_node(root, v)

    return node


def strictify_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    schema = copy.deepcopy(schema)
    _walk_all_refs_and_fix_defs(schema)

    out = strictify_node(schema, schema)
    if not isinstance(out, dict):
        raise ValueError("schema root must be dict")

    # 工具 parameters 必须是 object
    if out.get("type") != "object":
        out = {
            "type": "object",
            "properties": {"data": out},
            "required": ["data"],
            "additionalProperties": False,
        }

    # 根也不能空
    props = out.get("properties") or {}
    if not isinstance(props, dict) or len(props) == 0:
        out = {
            "type": "object",
            "properties": {"data": {"type": "string"}},
            "required": ["data"],
            "additionalProperties": False,
        }
    else:
        out["required"] = list(props.keys())
        out["additionalProperties"] = False

    return out


def lint_schema(schema: Dict[str, Any]) -> List[str]:
    bad: List[str] = []

    def rec(x: Any, path: str):
        if isinstance(x, dict):
            if "anyOf" in x and isinstance(x["anyOf"], list):
                for i, opt in enumerate(x["anyOf"]):
                    if isinstance(opt, dict) and "type" not in opt:
                        bad.append(f"{path}.anyOf[{i}] missing type keys={list(opt.keys())}")
            if x.get("type") == "object":
                props = x.get("properties", None)
                if props is None or (isinstance(props, dict) and len(props) == 0):
                    bad.append(f"{path} is object with no properties")
            for k, v in x.items():
                rec(v, f"{path}.{k}" if path else k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, f"{path}[{i}]")

    rec(schema, "")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--lint", action="store_true")
    ap.add_argument("--fail", action="store_true")
    args = ap.parse_args()

    s = load_json(Path(args.inp))
    out = strictify_schema(s)
    dump_json(Path(args.out), out)
    print(f"[OK] wrote strict schema -> {Path(args.out).resolve()}")

    if args.lint:
        bad = lint_schema(out)
        if bad:
            print(f"[WARN] lint found {len(bad)} issues (show first 50):")
            for x in bad[:50]:
                print("  -", x)
            if args.fail:
                raise SystemExit(1)
        else:
            print("[OK] lint passed")


if __name__ == "__main__":
    main()
