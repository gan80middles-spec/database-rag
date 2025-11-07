#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf2txt_unified.py — PDF 正文 + 表格线性化，统一输出到一个 TXT
依赖：pip install pymupdf pdfplumber
建议：扫描件先用 OCRmyPDF 加文本层： ocrmypdf --skip-text in.pdf out.pdf
"""

from __future__ import annotations
import argparse, math, re, statistics
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

# -------- 表格遮蔽辅助工具 --------
def find_table_bboxes(pdf_path: Path, page_index: int):
    import pdfplumber

    bboxes = []
    with pdfplumber.open(str(pdf_path)) as doc:
        page = doc.pages[page_index]
        settings_lines = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
        }
        tables = page.find_tables(table_settings=settings_lines) or []
        if not tables:
            settings_text = {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "text_x_tolerance": 2,
                "text_y_tolerance": 2,
            }
            tables = page.find_tables(table_settings=settings_text) or []
        for t in tables:
            bboxes.append(tuple(t.bbox))
    return bboxes


def _rect_intersect(a, b, pad=0.0):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ax0 -= pad
    ay0 -= pad
    ax1 += pad
    ay1 += pad
    bx0 -= pad
    by0 -= pad
    bx1 += pad
    by1 += pad
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def extract_page_text_without_tables(doc, page_index: int, sort=True, table_bboxes=None):
    page = doc.load_page(page_index)
    words = page.get_text("words", sort=sort) or []
    keep = []
    if table_bboxes:
        for w in words:
            wx0, wy0, wx1, wy1, wtext, *_ = w
            wbox = (wx0, wy0, wx1, wy1)
            if any(_rect_intersect(wbox, tb, pad=0.5) for tb in table_bboxes):
                continue
            keep.append(w)
    else:
        keep = words

    keep.sort(key=lambda x: (x[5], x[6], x[1], x[0]))
    lines = []
    cur_key = None
    buf = []
    for w in keep:
        key = (w[5], w[6])
        if key != cur_key and buf:
            lines.append(" ".join(buf))
            buf = []
        buf.append(w[4])
        cur_key = key
    if buf:
        lines.append(" ".join(buf))
    text = "\n".join(lines)
    return text

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

def _col_stats(col_vals: List[str]):
    lens = [len(v) for v in col_vals if v]
    avg_len = statistics.mean(lens) if lens else 0.0
    zh_ratio = (
        statistics.mean([
            sum(1 for ch in v if "\u4e00" <= ch <= "\u9fff") / max(1, len(v))
            for v in col_vals if v
        ])
        if col_vals
        else 0.0
    )
    num_like = [bool(re.fullmatch(r"[0-9一二三四五六七八九十]+", (v or "").strip())) for v in col_vals]
    num_ratio = sum(num_like) / max(1, len(col_vals))
    pure_num_ratio = sum(bool(re.fullmatch(r"\d+", (v or "").strip())) for v in col_vals) / max(1, len(col_vals))
    return dict(avg_len=avg_len, zh_ratio=zh_ratio, num_ratio=num_ratio, pure_num_ratio=pure_num_ratio)


def _is_id_column(vals: List[str]) -> bool:
    stripped = []
    for v in vals:
        v = (v or "").strip()
        if v == "":
            continue
        m = re.fullmatch(r"\d+", v)
        if m:
            stripped.append(int(v))
        else:
            return False
    if len(stripped) < max(2, math.ceil(0.5 * len(vals))):
        return False
    inc = all(stripped[i] <= stripped[i + 1] for i in range(len(stripped) - 1))
    return inc


def _merge_cont_rows(rows, idx_no, idx_name, idx_desc):
    merged = []
    for row in rows:
        def get(i):
            return (row[i].strip() if i is not None and i < len(row) and row[i] else "")

        no = get(idx_no)
        name = get(idx_name)
        desc = get(idx_desc)
        if (not no and not name) and desc and merged:
            merged[-1]["desc"].append(desc)
        else:
            merged.append({"no": no, "name": name, "desc": [desc] if desc else [], "raw": row})
    return merged


def linearize_table(rows: List[List[str]], known_order: Optional[List[str]] = None) -> str:
    if not rows:
        return ""

    rows = [[re.sub(r"\s+", " ", dedupe_doubled_chars((c or "").strip())) for c in row] for row in rows]

    header = rows[0]
    data = rows[1:]

    def looks_like_header(cells):
        if not cells:
            return False
        txt = " ".join(cells)
        has_zh = re.search(r"[\u4e00-\u9fff]", txt) is not None
        many_digits = len(re.findall(r"\d", txt)) > max(4, len(txt) * 0.3)
        return has_zh and not many_digits and (max(len(c) for c in cells) if cells else 0) <= 20

    if not looks_like_header(header):
        header = [f"列{j+1}" for j in range(len(rows[0]))]
        data = rows

    cols = list(zip(*data)) if data else [[] for _ in header]
    stats = [_col_stats(list(c)) for c in cols] if cols else []

    idx_id_candidates = [j for j, c in enumerate(cols) if _is_id_column(list(c))]
    idx_no = idx_id_candidates[0] if idx_id_candidates else None

    idx_desc = max(range(len(header)), key=lambda j: stats[j]["avg_len"]) if header else None

    candidates = [j for j in range(len(header)) if j != idx_desc]
    idx_name = max(candidates, key=lambda j: (stats[j]["zh_ratio"], -abs(stats[j]["avg_len"] - 12))) if candidates else None

    merged = _merge_cont_rows(data, idx_no, idx_name, idx_desc)

    out = []
    for ridx, item in enumerate(merged, start=1):
        no = item["no"] or str(ridx)
        out.append(f"编号{no}：")
        if idx_name is not None:
            name = item["name"]
            if name:
                out.append(f"模块名称: {name}")
        if idx_desc is not None:
            desc = "；".join([d for d in item["desc"] if d])
            if desc:
                out.append(f"内容说明: {desc}")
        if (idx_name is None and idx_desc is None) or (not item["name"] and not item["desc"]):
            row = item["raw"]
            for j, h in enumerate(header):
                cell = row[j] if j < len(row) else ""
                if cell.strip():
                    out.append(f"{h}: {cell.strip()}")
        out.append("")
    return "\n".join(out).rstrip()

# -------- 整体流程：逐页正文 + 表格线性化 并入同一 TXT --------
def pdf_to_unified_txt(pdf_path: Path, out_txt: Path, sort_text: bool = True) -> None:
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    pages_out: List[str] = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        table_bboxes = find_table_bboxes(pdf_path, pno)
        text = extract_page_text_without_tables(doc, pno, sort=sort_text, table_bboxes=table_bboxes)
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
