#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf2txt_unified.py — PDF 正文 + 表格线性化，统一输出到一个 TXT
依赖：pip install pymupdf pdfplumber
建议：扫描件先用 OCRmyPDF 加文本层： ocrmypdf --skip-text in.pdf out.pdf
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import List, Optional

# -------- 基础清洗与复选框修复 --------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00A0", " ")           # 不换行空格
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\s*\n\s*", "", text)        # 软连字符换行
    text = re.sub(r"\n{3,}", "\n\n", text)       # 压缩多余空行
    return text

def repair_checkboxes(text: str) -> str:
    lines = text.splitlines()
    out = []
    for l in lines:
        s = l.strip()
        if s in {"☐✓", "☑", "✓", "☐ ✓"} and out:
            out[-1] = out[-1].replace("☐", "☑", 1)
        else:
            out.append(l)
    norm = []
    for l in out:
        m = re.match(r"^([ \t]*)([☐☑])\s*(.*)$", l)
        if m:
            indent, box, body = m.groups()
            body = body.replace("✓", "").replace("☐", "").strip("：: \t")
            mark = "x" if box == "☑" else " "
            norm.append(f"{indent}- [{mark}] {body}")
        else:
            norm.append(l)
    return "\n".join(norm)

# -------- 表格抽取（pdfplumber）与线性化 --------
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
                rows.append([re.sub(r"\s+", " ", (c or "").strip()) for c in row])
            if rows:
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
    header_norm = [re.sub(r"\s+", "", h) for h in header]
    # 同义映射
    alias = {
        "编号": ["编号", "序号", "編號"],
        "模块名称": ["模块名称", "模块名", "模块", "名称", "模塊名稱"],
        "内容说明": ["内容说明", "说明", "描述", "內容說明"],
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

    # 2) 行改写
    out_lines: List[str] = []
    for ridx, row in enumerate(data, start=1):
        def safe(i): 
            return (row[i] if i is not None and i < len(row) else "").strip()
        no = safe(idx_no) or str(ridx)
        out_lines.append(f"编号{no}：")
        # 优先输出三大列；其余补充
        if idx_mod is not None:
            val = safe(idx_mod)
            if val:
                out_lines.append(f"模块名称: {val}")
        if idx_desc is not None:
            val = safe(idx_desc)
            if val:
                out_lines.append(f"内容说明: {val}")
        # 其他列
        used = set([c for c in [idx_no, idx_mod, idx_desc] if c is not None])
        for j, h in enumerate(header):
            if j in used: 
                continue
            val = safe(j)
            if val:
                out_lines.append(f"{h}: {val}")
        out_lines.append("")  # 空行分隔
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
        text = repair_checkboxes(text)

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
