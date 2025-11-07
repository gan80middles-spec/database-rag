#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf2txt_unified.py — PDF 正文 + 表格线性化，统一输出到一个 TXT
依赖：pip install pymupdf pdfplumber
建议：扫描件先用 OCRmyPDF 加文本层： ocrmypdf --skip-text in.pdf out.pdf
"""

from __future__ import annotations
import argparse, math, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------- 基础清洗与复选框修复 --------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00A0", " ")           # 不换行空格
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\s*\n\s*", "", text)        # 软连字符换行
    text = re.sub(r"\n{3,}", "\n\n", text)       # 压缩多余空行
    return text

def _normalize_line_key(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

def _detect_checkbox_lines(page) -> List[Tuple[str, bool]]:
    if page is None:
        return []
    words = page.get_text("words") or []
    line_map: Dict[Tuple[int, int], List[Tuple[int, float, float, float, float, str]]] = {}
    for x0, y0, x1, y1, word, block, line, word_no in words:
        key = (block, line)
        line_map.setdefault(key, []).append((word_no, x0, y0, x1, y1, word))
    results: List[Tuple[str, bool]] = []
    for key in sorted(line_map.keys()):
        items = sorted(line_map[key], key=lambda t: t[0])
        text = " ".join(it[5] for it in items)
        checked = False
        boxes: List[Tuple[float, float]] = []
        ticks: List[Tuple[float, float]] = []
        for _, x0, y0, x1, y1, word in items:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            if "☑" in word:
                checked = True
            for ch in word:
                if ch == "☐":
                    boxes.append((cx, cy))
                elif ch == "✓":
                    ticks.append((cx, cy))
        if not checked and boxes and ticks:
            for bx, by in boxes:
                for tx, ty in ticks:
                    if math.hypot(bx - tx, by - ty) < 25:
                        checked = True
                        break
                if checked:
                    break
        results.append((text, checked))
    return results

def repair_checkboxes(text: str, page=None) -> str:
    lines = text.splitlines()
    detected = _detect_checkbox_lines(page)
    detected_keys = [_normalize_line_key(t) for t, _ in detected]
    detected_idx = 0

    def lookup_checked(line: str) -> bool:
        nonlocal detected_idx
        if not detected:
            return False
        key = _normalize_line_key(line)
        if not key:
            return False
        for i in range(detected_idx, len(detected)):
            if detected_keys[i] == key:
                detected_idx = i + 1
                return detected[i][1]
        return False

    line_items = []
    for line in lines:
        item_checked = lookup_checked(line)
        if "☑" in line:
            item_checked = True
        line_items.append({"line": line, "checked": item_checked})

    merged: List[Dict[str, object]] = []
    for item in line_items:
        stripped = item["line"].strip()
        if stripped in {"☐✓", "☑", "✓", "☐ ✓"} and merged:
            prev = merged[-1]
            prev_line = str(prev["line"]).replace("☐", "☑", 1)
            prev["line"] = prev_line
            prev["checked"] = True
        else:
            merged.append(item)

    AMT_CN = r"[壹贰叁肆伍陆柒捌玖拾佰仟万亿零]+元"
    amt_pattern = re.compile(AMT_CN + r"|¥\s*\d[\d,]*\.?\d*")
    for idx, item in enumerate(merged):
        line_text = str(item["line"])
        ctx = line_text
        if idx + 1 < len(merged):
            ctx += " " + str(merged[idx + 1]["line"])
        if "固定总价合同" in line_text and amt_pattern.search(ctx):
            item["checked"] = True

    norm = []
    for item in merged:
        line = str(item["line"])
        m = re.match(r"^([ \t]*)([☐☑])\s*(.*)$", line)
        if m:
            indent, box, body = m.groups()
            body = body.replace("✓", "").replace("☐", "").strip("：: \t")
            mark = "x" if item.get("checked") or box == "☑" else " "
            norm.append(f"{indent}- [{mark}] {body}")
        else:
            norm.append(line)
    return "\n".join(norm)

# -------- 表格抽取（pdfplumber）与线性化 --------
def dedupe_doubled_chars(s: str) -> str:
    if not s:
        return s
    pairs = sum(1 for i in range(1, len(s)) if s[i] == s[i - 1])
    if pairs >= 0.4 * len(s):
        return "".join(s[i] for i in range(len(s)) if i % 2 == 0)
    return s

def dedupe_full_runs(s: str) -> str:
    return re.sub(r"([\u4e00-\u9fff])\1+", r"\1", s)

HEADER_TERMS = [
    "编号", "序号", "序", "号",
    "模块", "模块名称", "模块名", "项目", "子项目",
    "内容", "说明", "描述", "服务范围", "服务要求", "服务标准",
    "数量", "单位", "单价", "合价", "金额",
]

def _is_probable_table(rows: List[List[str]]) -> bool:
    if len(rows) < 2:
        return False
    num_cols = max((len(r) for r in rows if r), default=0)
    if num_cols < 2:
        return False
    header = rows[0]
    header_cells = [c or "" for c in header]
    hits = 0
    for term in HEADER_TERMS:
        if any(term in cell for cell in header_cells):
            hits += 1
    if hits < 2:
        return False
    total_cells = sum(len(r) for r in rows if r)
    if total_cells == 0:
        return False
    total_len = sum(len(c or "") for r in rows for c in r)
    avg_len = total_len / total_cells
    if avg_len >= 120:
        return False
    return True

def _extract_tables_on_page(pdf_path: Path, page_index: int) -> List[List[List[str]]]:
    """返回同一页的多张表，每张表是 rows(list[list[str]])。"""
    import pdfplumber  # 文本型 PDF 表格：lines / text 两策略
    tables_all: List[List[List[str]]] = []
    with pdfplumber.open(str(pdf_path)) as doc:
        page = doc.pages[page_index]
        # 先用“线条策略”（有网格线/边框时更稳）
        settings_lines = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
        }
        tbls = page.extract_tables(table_settings=settings_lines) or []
        # 若没抓到，退回“文本策略”（无网格线的表）
        if not tbls:
            settings_text = {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "text_x_tolerance": 2,
                "text_y_tolerance": 2,
            }
            tbls = page.extract_tables(table_settings=settings_text) or []
        for t in tbls:
            # 规范化单元格：去多余空白，合并断行
            rows = []
            for row in t:
                if row is None:
                    continue
                normalized_row = []
                for c in row:
                    cell = re.sub(r"\s+", " ", (c or "").strip())
                    cell = dedupe_doubled_chars(cell)
                    cell = dedupe_full_runs(cell)
                    normalized_row.append(cell)
                rows.append(normalized_row)
            if rows and _is_probable_table(rows):
                tables_all.append(rows)
    return tables_all

def _looks_like_header(cells: List[str]) -> bool:
    """启发式：是否像表头（含汉字、少数字、无长串数值）。"""
    if not cells: 
        return False
    txt = " ".join(cells)
    has_cjk = re.search(r"[\u4e00-\u9fff]", txt) is not None
    many_digits = len(re.findall(r"\d", txt)) > max(4, len(txt) * 0.4)
    return has_cjk and not many_digits

def linearize_table(rows: List[List[str]], known_order: Optional[List[str]] = None) -> str:
    """
    将表格改写为可读文本。优先识别 '编号/序号'、'模块名称'、'内容说明' 等列名；
    若无法确定表头，则以“第i行”形式输出并按列顺序列出。
    """
    if not rows:
        return ""

    # 1) 表头检测
    header = rows[0]
    data = rows[1:]
    if not _looks_like_header(header):
        # 首行不像表头：用默认列名 C1,C2...
        header = [f"列{j+1}" for j in range(len(rows[0]))]
        data = rows
    # 列名标准化
    header_norm = [re.sub(r"\s+", "", h or "") for h in header]
    alias = {
        "编号": ["编号", "序号", "編號", "序", "序號", "序列", "号"],
        "模块名称": [
            "模块名称", "模块名", "模块", "名称", "模塊名稱", "项目名称", "项目", "子项目", "模组",
            "品目", "品目编码", "服务范围", "服务要求", "服务标准"
        ],
        "内容说明": [
            "内容说明", "说明", "描述", "內容說明", "内容", "详情", "详细", "服务范围",
            "服务要求", "服务标准", "标准", "要求"
        ],
    }
    def find_col(name: str) -> Optional[int]:
        cands = alias.get(name, []) + [name]
        for i, h in enumerate(header_norm):
            if any(k == h or k in h for k in cands):
                return i
        return None

    idx_no   = find_col("编号")
    idx_mod  = find_col("模块名称")
    idx_desc = find_col("内容说明")

    num_cols = len(header)
    desc_index = idx_desc if idx_desc is not None else (num_cols - 1 if num_cols else None)

    def row_values(row: List[str]) -> List[str]:
        return [(row[j] if j < len(row) else "").strip() for j in range(num_cols)]

    merged_rows: List[Dict[str, object]] = []
    for row in data:
        values = row_values(row)
        no_val = values[idx_no] if idx_no is not None and idx_no < len(values) else ""
        mod_val = values[idx_mod] if idx_mod is not None and idx_mod < len(values) else ""
        desc_val = values[desc_index] if desc_index is not None and desc_index < len(values) else ""
        should_merge = False
        if desc_index is not None and desc_val:
            if (not no_val and not mod_val) and merged_rows:
                others = [values[j] for j in range(num_cols) if j != desc_index and values[j]]
                if not others:
                    should_merge = True
        if should_merge:
            last = merged_rows[-1]
            last_desc: List[str] = last.setdefault("desc", [])  # type: ignore
            last_desc.append(desc_val)
            if desc_index is not None:
                joined = "；".join(last_desc)
                last_values: List[str] = last["values"]  # type: ignore
                last_values[desc_index] = joined
            continue
        entry: Dict[str, object] = {
            "values": values,
            "desc": [desc_val] if (desc_index is not None and desc_val) else [],
            "no": no_val,
            "mod": mod_val,
        }
        merged_rows.append(entry)

    out_lines: List[str] = []
    if idx_desc is not None:
        for ridx, entry in enumerate(merged_rows, start=1):
            values = entry["values"]  # type: ignore
            number = str(entry.get("no")) if entry.get("no") else str(ridx)
            out_lines.append(f"编号{number}：")
            if idx_mod is not None:
                mod_val = values[idx_mod]
                if mod_val:
                    out_lines.append(f"模块名称: {mod_val}")
            desc_items: List[str] = entry.get("desc", [])  # type: ignore
            desc_val = "；".join(desc_items) if desc_items else (values[idx_desc] if values[idx_desc] else "")
            if desc_val:
                out_lines.append(f"内容说明: {desc_val}")
            used = {c for c in [idx_no, idx_mod, idx_desc] if c is not None}
            for j, h in enumerate(header):
                if j in used:
                    continue
                val = values[j]
                if val:
                    label = h or f"列{j+1}"
                    out_lines.append(f"{label}: {val}")
            out_lines.append("")
    else:
        for ridx, entry in enumerate(merged_rows, start=1):
            values = entry["values"]  # type: ignore
            out_lines.append(f"第{ridx}行：")
            for j, h in enumerate(header):
                val = values[j]
                if val:
                    label = h or f"列{j+1}"
                    out_lines.append(f"{label}: {val}")
            out_lines.append("")
    return "\n".join(out_lines).rstrip()

# -------- 整体流程：逐页正文 + 表格线性化 并入同一 TXT --------
def pdf_to_unified_txt(pdf_path: Path, out_txt: Path, sort_text: bool = True) -> None:
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    pages_out: List[str] = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text = page.get_text("text", sort=sort_text)
        text = clean_text(text)
        text = repair_checkboxes(text, page)

        # 同页表格
        tables = _extract_tables_on_page(pdf_path, pno)
        if tables:
            text += "\n\n" + f"【表格（第{pno+1}页）】".strip() + "\n"
            for ti, tbl in enumerate(tables, start=1):
                text += f"—— 表 {ti} ——\n"
                text += linearize_table(tbl) + "\n"

        pages_out.append(f"=== Page {pno+1} ===\n{text.strip()}\n")

    out_txt.write_text("\n\n".join(pages_out), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="PDF 正文+表格线性化 → 单一 TXT")
    ap.add_argument("pdf", help="输入 PDF")
    ap.add_argument("-o", "--output", help="输出 TXT（默认与PDF同名）")
    ap.add_argument("--no-sort", action="store_true", help="禁用 PyMuPDF 文本排序（默认开启）")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_txt  = Path(args.output) if args.output else pdf_path.with_suffix(".txt")

    pdf_to_unified_txt(pdf_path, out_txt, sort_text=(not args.no_sort))
    print(f"[OK] {pdf_path.name} → {out_txt}")

if __name__ == "__main__":
    main()
