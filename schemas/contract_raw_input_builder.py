# schemas/build_raw_input_from_contract_files.py
# -*- coding: utf-8 -*-
"""
把 contract_template_chunker 产出的 docs.jsonl + chunks.jsonl 转成 deepseek_contractgraph 的 raw_input.jsonl。

特性（偏工程化，跑大库更稳）：
- 支持断点续跑（--resume）
- 支持只处理某个 doc_id（--doc_id）
- 支持按 clause_type 过滤（--keep_clause_types）
- 支持把一个合同按“三部分标题”拆成 3 段 raw_input：
    第一部分 合同协议书
    第二部分 通用合同条款
    第三部分 专用合同条款
  并支持段间 overlap（--part_overlap）
- 支持每段拼接 text 的最大字符数（--max_chars_per_part），避免超长输入导致 tool_calls 截断
- 可选输出 seed_clauses（--emit_seed_clauses），便于 deepseek_contractgraph 强制沿用 chunk_id 作为 clause.id（强烈推荐）
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm


# -------------------------
# IO utils
# -------------------------

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_done_ids(out_jsonl: Path, id_key: str = "doc_id") -> Set[str]:
    done: Set[str] = set()
    if not out_jsonl.exists():
        return done
    for row in read_jsonl(out_jsonl):
        if isinstance(row, dict):
            did = row.get(id_key)
            if isinstance(did, str) and did:
                done.add(did)
    return done


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def norm_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


# -------------------------
# clause modeling
# -------------------------

def build_seed_clause_from_chunk(ch: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容你可能有的字段：
    - chunk_id / doc_id / order / chunk_index
    - clause_no / section / parent_section / section_path
    - clause_type
    - text
    """
    clause_id = ch.get("chunk_id") or ch.get("clause_id") or ""
    section_path = ch.get("section_path")
    if not isinstance(section_path, list):
        section_path = []

    # order：优先 order，其次 chunk_index，再次 0
    order = ch.get("order")
    if order is None:
        order = ch.get("chunk_index")
    order_i = safe_int(order, 0)

    title = ch.get("section") or ch.get("title") or ""
    parent_section = ch.get("parent_section") or ""
    text = norm_str(ch.get("text")).strip()

    # clause_type：chunker 若有输出，用它；否则留空
    clause_type = ch.get("clause_type") or ch.get("category") or None

    return {
        "id": norm_str(clause_id),
        "order": order_i,
        "clause_no": ch.get("clause_no"),
        "title": norm_str(title).strip(),
        "parent_section": norm_str(parent_section).strip(),
        "section_path": section_path,
        "clause_type": clause_type,
        "text": text,
    }


def dedup_keep_first(items: List[Dict[str, Any]], key: str = "id") -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        k = norm_str(it.get(key))
        if k and k in seen:
            continue
        if k:
            seen.add(k)
        out.append(it)
    return out


def join_clauses_to_text(
    clauses: List[Dict[str, Any]],
    *,
    max_chars: int = 0,
    sep_style: str = "header",
    keep_empty_titles: bool = True,
) -> str:
    """
    把条款拼成给 LLM 的 text。
    - sep_style=header: 每条款带明显 header（便于模型定位）
    - max_chars>0: 截断总长度（从前往后保留）
    """
    parts: List[str] = []
    total = 0

    # 保证顺序
    clauses = sorted(clauses, key=lambda x: (safe_int(x.get("order"), 0), norm_str(x.get("id"))))

    for c in clauses:
        title = norm_str(c.get("title")).strip()
        clause_no = norm_str(c.get("clause_no")).strip()
        text = norm_str(c.get("text")).strip()

        if not text and not (keep_empty_titles and title):
            continue

        if sep_style == "header":
            head = " ".join([x for x in [clause_no, title] if x]).strip()
            if head:
                block = f"\n{text}".strip()
            else:
                block = text
        else:
            block = text

        block = block.strip()
        if not block:
            continue

        if max_chars > 0 and total + len(block) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            parts.append(block[:remain])
            break

        parts.append(block)
        total += len(block) + 2

    return "\n\n".join(parts).strip()


# -------------------------
# split into 3 parts
# -------------------------

@dataclass(frozen=True)
class PartSpec:
    part_id: str
    part_name: str
    order: int


PARTS: List[PartSpec] = [
    PartSpec("p1", "第一部分 合同协议书", 1),
    PartSpec("p2", "第二部分 通用合同条款", 2),
    PartSpec("p3", "第三部分 专用合同条款", 3),
]

PART_NAME: Dict[str, str] = {p.part_id: p.part_name for p in PARTS}

# 更鲁棒：允许“通用条款/专用条款”简写；允许空格/全半角差异
PART_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"第\s*一\s*部\s*分.*合同协议书"), "p1"),
    (re.compile(r"第\s*二\s*部\s*分.*(通用合同条款|通用条款)"), "p2"),
    (re.compile(r"第\s*三\s*部\s*分.*(专用合同条款|专用条款)"), "p3"),
]


def detect_part_marker(clause: Dict[str, Any]) -> Optional[str]:
    """
    在 clause 的多个字段里查找分段标题：
    - title
    - parent_section
    - section_path 拼接
    - text 前若干字符（很多文档分段标题就在条款正文里）
    """
    title = norm_str(clause.get("title")).strip()
    parent = norm_str(clause.get("parent_section")).strip()
    sp = clause.get("section_path")
    sp_join = ""
    if isinstance(sp, list) and sp:
        sp_join = " ".join([norm_str(x).strip() for x in sp if norm_str(x).strip()])

    text = norm_str(clause.get("text")).strip()
    text_head = text[:200]  # 足够识别标题

    blob = "\n".join([title, parent, sp_join, text_head]).strip()

    if not blob:
        return None

    for rgx, pid in PART_PATTERNS:
        if rgx.search(blob):
            return pid
    return None


def split_seed_clauses_into_parts(
    seed: List[Dict[str, Any]],
    *,
    overlap_tail: int = 2,
    default_part: str = "p1",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    把 seed clauses（已排序）按分段标题切成 p1/p2/p3。
    - overlap_tail：把上一段最后 N 条复制到下一段开头，减少跨段引用丢上下文
      合并时按 clause.id 去重即可
    """
    parts: Dict[str, List[Dict[str, Any]]] = {p.part_id: [] for p in PARTS}
    current = default_part

    for c in seed:
        pid = detect_part_marker(c)
        if pid:
            current = pid
        parts[current].append(c)

    # 如果完全没识别到 p2/p3，就退化为单段
    if not parts["p2"] and not parts["p3"]:
        return {"p1": seed, "p2": [], "p3": []}

    # overlap：p1 tail -> p2 head, p2 tail -> p3 head（去重）
    def prepend_unique(dst: List[Dict[str, Any]], src_tail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = {norm_str(x.get("id")) for x in dst if norm_str(x.get("id"))}
        add = []
        for x in src_tail:
            xid = norm_str(x.get("id"))
            if xid and xid not in seen:
                add.append(x)
                seen.add(xid)
        return add + dst

    if overlap_tail > 0:
        p1_tail = parts["p1"][-overlap_tail:] if parts["p1"] else []
        p2_tail = parts["p2"][-overlap_tail:] if parts["p2"] else []
        if parts["p2"] and p1_tail:
            parts["p2"] = prepend_unique(parts["p2"], p1_tail)
        if parts["p3"] and p2_tail:
            parts["p3"] = prepend_unique(parts["p3"], p2_tail)

    # 最后保证每段内部按 order 排序，并且去重（overlap 可能导致局部重复）
    for pid in list(parts.keys()):
        parts[pid] = dedup_keep_first(
            sorted(parts[pid], key=lambda x: (safe_int(x.get("order"), 0), norm_str(x.get("id")))),
            key="id",
        )

    return parts


# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--docs", required=True, help="chunker 输出的 docs.jsonl")
    ap.add_argument("--chunks", required=True, help="chunker 输出的 chunks.jsonl")
    ap.add_argument("--out", required=True, help="输出 raw_input.jsonl")
    ap.add_argument("--err", default="", help="输出 errors.jsonl（可选）")

    ap.add_argument("--resume", action="store_true", help="断点续跑：out 已有 doc_id 不重复写")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 doc（0=不限制）")
    ap.add_argument("--doc_id", default="", help="只处理指定 doc_id（调试用）")

    ap.add_argument("--keep_clause_types", default="", help="仅保留这些 clause_type（逗号分隔，留空=全部）")

    ap.add_argument("--split_mode", choices=["none", "three_parts"], default="three_parts",
                    help="none=不拆分；three_parts=按三部分标题拆分")
    ap.add_argument("--part_overlap", type=int, default=2, help="段间 overlap：复制上一段末尾 N 条到下一段开头")
    ap.add_argument("--max_chars_per_part", type=int, default=0,
                    help="每段拼接 text 最大字符数（0=不截断）。建议给一个值以避免超长输入")
    ap.add_argument("--skip_empty_parts", action="store_true", help="拆分后如果某段为空则跳过该段输出")

    ap.add_argument("--emit_seed_clauses", action="store_true",
                    help="在 raw_input 中额外写入 seed_clauses（推荐：让后续抽取沿用 chunk_id 作为 clause.id）")

    ap.add_argument("--emit_clause_type_stats", action="store_true",
                    help="在 meta 中写入 clause_type 统计（便于调试/评估）")

    ap.add_argument("--sep_style", choices=["header", "plain"], default="header",
                    help="header=每条款前加【条号 标题】；plain=仅拼文本")

    args = ap.parse_args()

    docs_path = Path(args.docs)
    chunks_path = Path(args.chunks)
    out_path = Path(args.out)
    err_path = Path(args.err) if args.err else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if err_path:
        err_path.parent.mkdir(parents=True, exist_ok=True)

    keep_types: Optional[Set[str]] = None
    if args.keep_clause_types.strip():
        keep_types = set([x.strip() for x in args.keep_clause_types.split(",") if x.strip()])

    done: Set[str] = set()
    if args.resume:
        done = read_done_ids(out_path, id_key="doc_id")
        print(f"[INFO] resume: already have {len(done)} rows in {out_path}")

    # 1) 先读 docs：选出要处理的 doc_id 列表 & meta
    doc_meta_map: Dict[str, Dict[str, Any]] = {}
    doc_ids: List[str] = []

    for d in read_jsonl(docs_path):
        doc_id = d.get("doc_id") or d.get("contract_id")
        if not doc_id:
            continue
        doc_id = str(doc_id)

        if args.doc_id and doc_id != args.doc_id:
            continue

        # 如果拆分模式：输出 doc_id 会变成 doc_id#p1/#p2/#p3，所以这里不能仅用 parent 判断 done
        # 我们让 done 的判断放到写出时（每个 part 自己判断）
        doc_meta_map[doc_id] = d
        doc_ids.append(doc_id)

        if args.limit and len(doc_ids) >= args.limit:
            break

    if not doc_ids:
        print("[WARN] no docs selected")
        return

    target_ids = set(doc_ids)

    # 2) 再读 chunks：只聚合需要的 doc_id
    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for ch in tqdm(read_jsonl(chunks_path), desc="load chunks"):
        did = ch.get("doc_id")
        if not did:
            continue
        did = str(did)
        if did not in target_ids:
            continue

        if keep_types is not None:
            ct = ch.get("clause_type") or ch.get("category")
            if ct and str(ct) not in keep_types:
                continue

        chunks_by_doc[did].append(ch)

    # 3) 写 raw_input
    n_ok = 0
    n_err = 0

    it = tqdm(doc_ids, desc="write raw_input")
    for parent_doc_id in it:
        d = doc_meta_map[parent_doc_id]
        try:
            ch_list = chunks_by_doc.get(parent_doc_id, [])
            if not ch_list:
                raise RuntimeError("no chunks found for doc_id")

            seed = [build_seed_clause_from_chunk(ch) for ch in ch_list]
            seed = sorted(seed, key=lambda x: (safe_int(x.get("order"), 0), norm_str(x.get("id"))))
            seed = dedup_keep_first(seed, key="id")

            # doc-level meta：尽量多保留（你后面可能要过滤/回溯）
            meta: Dict[str, Any] = {}
            # 常见字段：business_type/legal_type/title/source/language/presence_summary
            for k in [
                "doc_type", "business_type", "legal_type", "title", "source",
                "language", "presence_summary", "consult_category"
            ]:
                if k in d:
                    meta[k] = d.get(k)
            if isinstance(d.get("meta"), dict):
                meta.update(d["meta"])

            # contract_type：优先用 business_type
            contract_type = meta.get("contract_type") or meta.get("business_type") or d.get("business_type") or "other"
            meta["contract_type"] = contract_type
            meta["parent_doc_id"] = parent_doc_id

            # clause_type 统计（可选）
            if args.emit_clause_type_stats:
                ct_counter = Counter()
                for c in seed:
                    ct = c.get("clause_type")
                    if ct:
                        ct_counter[str(ct)] += 1
                meta["clause_type_stats"] = dict(ct_counter)

            # split
            parts_map: Dict[str, List[Dict[str, Any]]]
            if args.split_mode == "three_parts":
                parts_map = split_seed_clauses_into_parts(seed, overlap_tail=args.part_overlap, default_part="p1")
            else:
                parts_map = {"p1": seed, "p2": [], "p3": []}

            # 输出：每个 part 一行 raw_input（doc_id#pX）
            for pid in ["p1", "p2", "p3"]:
                clauses = parts_map.get(pid, [])
                if not clauses and args.skip_empty_parts:
                    continue

                out_doc_id = f"{parent_doc_id}#{pid}"
                if args.resume and out_doc_id in done:
                    continue

                text = join_clauses_to_text(
                    clauses,
                    max_chars=args.max_chars_per_part,
                    sep_style=args.sep_style,
                    keep_empty_titles=True,
                )

                if not text and args.skip_empty_parts:
                    continue

                # part meta：独立字段，便于你后面 merge
                part_meta = dict(meta)
                part_meta["part_id"] = pid
                part_meta["part_name"] = PART_NAME.get(pid, pid)

                # 回溯锚点
                part_meta["chunk_ids"] = [c["id"] for c in clauses if norm_str(c.get("id"))]
                part_meta["sections"] = [c.get("title") for c in clauses if norm_str(c.get("title"))]

                row: Dict[str, Any] = {
                    "doc_id": out_doc_id,
                    "doc_type": d.get("doc_type") or "contract_template",
                    "source": d.get("source") or meta.get("source") or "template",
                    "text": text,
                    "meta": part_meta,
                }

                if args.emit_seed_clauses:
                    # 你后续 deepseek_contractgraph 可以用这个强制 clauses[].id 必须沿用 chunk_id
                    row["seed_clauses"] = clauses

                append_jsonl(out_path, row)
                done.add(out_doc_id)
                n_ok += 1
                it.set_postfix(ok=n_ok, err=n_err)

        except Exception as e:
            n_err += 1
            it.set_postfix(ok=n_ok, err=n_err)
            if err_path:
                append_jsonl(err_path, {"doc_id": parent_doc_id, "error": f"{type(e).__name__}: {e}"})

    print(f"[OK] wrote raw_input: {out_path}  ok={n_ok} err={n_err}")
    if err_path:
        print(f"[OK] errors: {err_path}")


if __name__ == "__main__":
    main()
