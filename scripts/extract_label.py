#!/usr/bin/env python3
# extract_labels.py
import json, argparse
from typing import Any, Iterable, List, Optional

def load_json_any(path: str) -> Any:
    # 既兼容大 JSON，也兼容 NDJSON（每行一个 JSON）
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 作为 NDJSON 逐行解析
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

def get_root(data: Any) -> Any:
    # 你的示例是 { code, msg, data: [...] }
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data

def iter_nodes(x: Any) -> Iterable[dict]:
    if x is None:
        return
    if isinstance(x, dict):
        yield x
        kids = x.get("childrens") or x.get("children") or []
        if isinstance(kids, list):
            for k in kids:
                yield from iter_nodes(k)
    elif isinstance(x, list):
        for item in x:
            yield from iter_nodes(item)

def to_label_tree(node: dict) -> Optional[dict]:
    label = node.get("label")
    kids = node.get("childrens") or node.get("children") or []
    child_list = []
    if isinstance(kids, list):
        for k in kids:
            t = to_label_tree(k)
            if t is not None:
                child_list.append(t)
    if label is None and not child_list:
        return None
    out = {}
    if label is not None:
        out["label"] = label
    if child_list:
        out["childrens"] = child_list
    return out

def main():
    ap = argparse.ArgumentParser(description="Extract only 'label' from nested JSON.")
    ap.add_argument("--input", required=True, help="输入 JSON 文件路径")
    ap.add_argument("--output", required=True, help="输出文件路径")
    ap.add_argument("--mode", choices=["flat", "tree"], default="flat",
                    help="flat=拉平所有 label；tree=保持树结构但仅保留 label/childrens")
    ap.add_argument("--unique", action="store_true", help="flat 模式下去重（保持首次出现顺序）")
    ap.add_argument("--out-format", choices=["txt", "json"], default="txt",
                    help="flat 模式下的输出格式")
    args = ap.parse_args()

    data = load_json_any(args.input)
    root = get_root(data)

    if args.mode == "flat":
        labels: List[str] = []
        seen = set()
        for node in iter_nodes(root):
            lab = node.get("label")
            if lab is None:
                continue
            if args.unique:
                if lab in seen:
                    continue
                seen.add(lab)
            labels.append(lab)
        if args.out_format == "txt":
            with open(args.output, "w", encoding="utf-8") as w:
                w.write("\n".join(labels))
        else:  # json
            with open(args.output, "w", encoding="utf-8") as w:
                json.dump(labels, w, ensure_ascii=False, indent=2)
        print(f"[OK] extracted {len(labels)} labels -> {args.output}")
    else:
        # tree
        if isinstance(root, list):
            out = [t for n in root if (t := to_label_tree(n)) is not None]
        elif isinstance(root, dict):
            out = to_label_tree(root)
        else:
            out = []
        with open(args.output, "w", encoding="utf-8") as w:
            json.dump(out, w, ensure_ascii=False, indent=2)
        print(f"[OK] tree written -> {args.output}")

if __name__ == "__main__":
    main()
