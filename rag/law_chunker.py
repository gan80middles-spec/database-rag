# -*- coding: utf-8 -*-
"""
通用法规抽取+切块脚本（12部法规可共用）
- 解析层级：编/章/节/条（自动识别，若缺编/节也能工作）
- 切块：以“条”为最小原子，合并相邻条凑到 400–900 字；块间重叠默认 100 字
- 兜底：若单条 > 1200 字，按 款/项/分段（（一）（二）/换行/；/。）在条内温和切分
用法示例：
python law_chunker.py \
  --in 民法典_20210101.txt \
  --law_name 民法典 \
  --version_date 2021-01-01 \
  --doc_id LAW-CIVIL-20210101 \
  --out 20_chunks/民法典.jsonl
"""
import re
import os
import json
import argparse
from datetime import datetime
from hashlib import sha1
from tqdm import tqdm

CN_NUM = "〇零一二三四五六七八九十百千两"
CN_NUM_MAP = {c:i for i,c in enumerate("零一二三四五六七八九")}
CN_NUM_MAP.update({"〇":0,"两":2,"十":10,"百":100,"千":1000})

def normalize_text(t: str) -> str:
    t = t.replace("\u3000", " ").replace("\xa0", " ").replace("\t", " ")
    t = re.sub(r"[ \r\f]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"附\s*则", "附则", t)
    t = re.sub(r"附\s*则", "附则", t)
    t = re.sub(r"附\s*件\s*([一二三四五六七八九十]+)", r"附件\1", t)
    # 清掉页码行（如：－151－）
    t = re.sub(r"^\s*[-－]\s*\d+\s*[-－]\s*$", "", t, flags=re.M)

    return t.strip()

RE_COMP    = re.compile(r"^\s*第\s*([\d%s]+)\s*编\s*(.*)$" % CN_NUM)
RE_CHAPTER = re.compile(r"^\s*第\s*([\d%s]+)\s*章\s*(.*)$" % CN_NUM)
RE_SECTION = re.compile(r"^\s*第\s*([\d%s]+)\s*节\s*(.*)$" % CN_NUM)
RE_ARTICLE  = re.compile(r"^\s*第\s*([\d%s]+)\s*条[：:．.\s]?(.*)$" % CN_NUM)
RE_APPENDIX = re.compile(r"^\s*附\s*则\s*$")
RE_APPENDIX   = re.compile(r"^\s*附\s*则\s*$")                   # 附 则 / 附则
RE_ATTACHMENT = re.compile(r"^\s*附\s*件\s*([一二三四五六七八九十]+)\s*(.*)$")


# 款/项/分段提示（用于长条“温和切分”）
RE_PARA_HINT = re.compile(r"(（\d+）|（[一二三四五六七八九十]+）|^\s*[一二三四五六七八九十]+、)", re.M)

def is_heading(line:str):
    for kind,pat in (("编",RE_COMP),("章",RE_CHAPTER),("节",RE_SECTION),("附则", RE_APPENDIX),("附件", RE_ATTACHMENT),("条",RE_ARTICLE)):
        m = pat.match(line)
        if m:
            return kind, m
    return None, None

def _tight(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

def norm_chapter(ch: str):
    if not ch: return ch
    m = re.match(r'^([一二三四五六七八九十百千两]+)\s*(.*)$', ch)
    return f"第{m.group(1)}章 " + _tight(m.group(2) or "") if m else ch

def norm_section(sec: str):
    if not sec: return sec
    m = re.match(r'^([一二三四五六七八九十百千两]+)\s*(.*)$', sec)
    return f"第{m.group(1)}节 " + _tight(m.group(2) or "") if m else sec

def norm_compilation(comp: str):
    if not comp: return comp
    m = re.match(r'^([一二三四五六七八九十百千两]+)\s*(.*)$', comp)
    return f"第{m.group(1)}编 " + _tight(m.group(2) or "") if m else comp


def read_articles(txt: str):
    """按层级解析为条列表，每条= {article_no, path:{编/章/节}, text}"""
    comp = chap = sec = None
    cur_article_no = None
    cur_buf = []
    articles = []

    def flush_article():
        nonlocal cur_article_no, cur_buf, comp, chap, sec
        if cur_article_no is not None:
            text = "\n".join(cur_buf).strip()
            if text:
                articles.append({
                    "article_no": cur_article_no,
                    "path": {"编": comp, "章": chap, "节": sec},
                    "text": text
                })
        cur_article_no, cur_buf = None, []

    lines = txt.splitlines()
    for raw_line in tqdm(lines, desc="解析条文", unit="行"):
        line = raw_line.rstrip()
        if not line.strip():
            if cur_article_no is not None:
                cur_buf.append("")  # 保留段落空行
            continue

        kind, m = is_heading(line)
        if kind == "编":
            flush_article()
            comp_raw = m.group(1).strip() + (" " + m.group(2).strip() if m.group(2) else "")
            comp = norm_compilation(comp_raw)
            chap = None
            sec = None
        elif kind == "章":
            flush_article()
            chap_raw = m.group(1).strip() + (" " + m.group(2).strip() if m.group(2) else "")
            chap = norm_chapter(chap_raw)
            sec = None
        elif kind == "附则":
            flush_article()
            chap = "附则"
            comp = None
            sec = None
        elif kind == "附件":
            flush_article()
            num = m.group(1) or ""
            tail = (m.group(2) or "").strip()
            sec = f"附件{num}" + (f" {tail}" if tail else "")
            cur_article_no = sec
            cur_buf = []
        elif kind == "节":
            flush_article()
            sec_raw = m.group(1).strip() + (" " + m.group(2).strip() if m.group(2) else "")
            sec = norm_section(sec_raw)
        elif kind == "条":
            flush_article()
            cur_article_no = "第" + m.group(1).strip() + "条"
            first = m.group(2).strip()
            cur_buf = [first] if first else []
        else:
            # 普通正文
            if cur_article_no is None:
                # 某些法规开头“总则/目的”未显式‘条’，并入上一章简介；通常可忽略或合并到下一条前导
                continue
            cur_buf.append(line)

    flush_article()
    return articles

def gentle_split_long_article(a_text: str, hard_limit=1200, target=700):
    text = a_text.strip()
    if len(text) <= hard_limit:
        return [text]

    # 2.1 先按“（一）（二）/ 一、二、三、”等提示切（这些常见于司法解释）
    hints = [p for p in re.split(r"(?=（\d+）|（[一二三四五六七八九十]+）|^\s*[一二三四五六七八九十]+、)", 
                                text, flags=re.M) if p.strip()]
    parts = hints if len(hints) > 1 else None

    # 2.2 若没切开，再退回按空行切
    if not parts:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if sum(len(p) for p in parts) == 0:
            parts = [text]

    # 2.3 组合到接近 target
    merged, buf = [], ""
    def push():
        nonlocal buf
        if buf.strip():
            merged.append(buf.strip()); buf = ""
    for p in parts:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) < target * 1.4:
            buf += "\n" + p
        else:
            push(); buf = p
    push()

    # 2.4 仍太长就按句号/分号再温和细分
    refined = []
    for seg in merged:
        if len(seg) <= hard_limit:
            refined.append(seg)
        else:
            clauses = re.split(r"(。|；)", seg)
            tmp = ""
            for i in range(0, len(clauses), 2):
                chunk = clauses[i] + (clauses[i+1] if i+1 < len(clauses) else "")
                if len(tmp) + len(chunk) < target * 1.3:
                    tmp += chunk
                else:
                    if tmp: refined.append(tmp); tmp = chunk
            if tmp: refined.append(tmp)
    return [s.strip() for s in refined if s.strip()]
    # """对超长条按 款/项/分段 提示温和切分，保持条内语义"""
    # text = a_text.strip()
    # if len(text) <= hard_limit:
    #     return [text]
    # # 优先按空行分段
    # parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # if sum(len(p) for p in parts) == 0:
    #     parts = [text]

    # # 合并/再切，目标段长 ~ target
    # merged = []
    # buf = ""
    # def push():
    #     nonlocal buf
    #     if buf.strip():
    #         merged.append(buf.strip())
    #         buf = ""

    # for p in parts:
    #     if not buf:
    #         buf = p
    #     elif len(buf) + 1 + len(p) < target * 1.4:
    #         buf += "\n" + p
    #     else:
    #         push()
    #         buf = p
    # push()

    # # 如果仍然很长，再按分号/句号温和切
    # refined = []
    # for seg in merged:
    #     if len(seg) <= hard_limit:
    #         refined.append(seg)
    #     else:
    #         clauses = re.split(r"(。|；)", seg)
    #         tmp = ""
    #         for i in range(0, len(clauses), 2):
    #             chunk = clauses[i] + (clauses[i+1] if i+1 < len(clauses) else "")
    #             if len(tmp) + len(chunk) < target * 1.3:
    #                 tmp += chunk
    #             else:
    #                 if tmp:
    #                     refined.append(tmp)
    #                 tmp = chunk
    #         if tmp:
    #             refined.append(tmp)
    # return [s.strip() for s in refined if s.strip()]
    

def make_chunks(articles, min_chars=400, max_chars=900, overlap=100):
    """
    以‘条’为原子合并出 400–900 字块；对超长条先做条内温和切分。
    返回列表：每个元素包含 text, start_article_idx, end_article_idx, path_info, article_span
    """
    # 先把每条可能拆成多个“条内段”
    expanded = []
    for idx, a in tqdm(enumerate(articles), total=len(articles), desc="条内切分", unit="条"):
        segs = gentle_split_long_article(a["text"])
        for j, seg in enumerate(segs):
            expanded.append({
                "article_idx": idx,
                "article_no": a["article_no"] + (f"（续{j+1}）" if len(segs) > 1 else ""),
                "path": a["path"],
                "text": seg
            })

    chunks = []
    buf = []
    buf_len = 0
    start_idx = 0

    def flush(end_idx, *, allow_overlap = True):
        nonlocal buf, buf_len, start_idx
        if not buf:
            return
        text = "\n".join([b["text"] for b in buf]).strip()
        arts = [b["article_no"] for b in buf]
        path = buf[0]["path"]
        chunks.append({
            "text": text,
            "article_span": f"{arts[0]}~{arts[-1]}" if len(set(arts)) > 1 else arts[0],
            "start_idx": start_idx,
            "end_idx": end_idx,
            "path": path
        })
        # 处理重叠：基于字符数退回 overlap
        if allow_overlap and overlap > 0 and text:
            # 简单从末尾截取 overlap 作为下个块的开头对齐
            remain = text[-overlap:]
            buf = [{**buf[-1], "text": remain}]
            buf_len = len(remain)
            start_idx = end_idx  # 逻辑索引续上
        else:
            buf, buf_len = [], 0
            start_idx = end_idx

    def same_scope(a,b):
        return (a["path"].get("编")==b["path"].get("编") and
                a["path"].get("章")==b["path"].get("章") and
                a["path"].get("节")==b["path"].get("节"))

    for i, seg in tqdm(enumerate(expanded), total=len(expanded), desc="合并切块", unit="段"):
        if buf and not same_scope(seg, buf[-1]):
            flush(i, allow_overlap=False)

        seglen = len(seg["text"])
        if buf_len == 0:
            buf = [seg]
            buf_len = seglen
            start_idx = i
            # 单段太大，直接输出
            if buf_len >= max_chars:
                flush(i)
            continue

        if buf_len + 1 + seglen <= max_chars:
            buf.append(seg)
            buf_len += 1 + seglen
        else:
            # 已超上限，先落块
            if buf_len >= min_chars:
                flush(i)
                # 新起一块
                buf = [seg]
                buf_len = seglen
                if buf_len >= max_chars:
                    flush(i)
            else:
                # buf 太短但再加就超上限：直接落地当前（可能略短），并起新块
                flush(i)
                buf = [seg]
                buf_len = seglen
                if buf_len >= max_chars:
                    flush(i)

    # 尾块
    if buf_len > 0:
        flush(len(expanded)-1)

    return chunks

def make_chunk_id(doc_id: str, article_span: str, idx: int):
    base = f"{doc_id}#{article_span}#{idx:03d}"
    h = sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{h}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="法规TXT文本（UTF-8）")
    ap.add_argument("--law_name", required=True, help="法名（如：民法典）")
    ap.add_argument("--version_date", required=True, help="版本日期 YYYY-MM-DD")
    ap.add_argument("--doc_id", required=True, help="文档ID（全库唯一）")
    ap.add_argument("--doc_type", default="statute",
                choices=["statute", "judicial_interpretation"])
    ap.add_argument("--issuer", default="", help="发布机关：SPC/SPP/联合等")
    ap.add_argument("--doc_no", default="", help="发文字号：如 法释〔2020〕15号")
    ap.add_argument("--title", default="", help="标题（可与 law_name 相同）")
    ap.add_argument("--status", default="有效", help="时效性：有效/失效/部分失效")
    ap.add_argument("--out", required=True, help="输出 JSONL 路径")
    ap.add_argument("--min_chars", type=int, default=400)
    ap.add_argument("--max_chars", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        raw = f.read()
    text = normalize_text(raw)
    articles = read_articles(text)
    if not articles:
        raise RuntimeError("未解析到任何‘第×条’：请检查输入是否为纯文本、条号是否规范。")

    chunks = make_chunks(
        articles,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap=args.overlap
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for i, c in tqdm(enumerate(chunks), total=len(chunks), desc="写出切块", unit="块"):
            rec = {
                "chunk_id": make_chunk_id(args.doc_id, c["article_span"], i),
                "doc_id": args.doc_id,
                "doc_type": args.doc_type,
                "issuer":args.issuer,
                "doc_no":args.doc_no,
                "title": args.title or args.law_name,
                "law_name":args.law_name,
                "article_no": c["article_span"],
                "version_date": args.version_date,
                "validity_status": args.status,
                "path": c["path"],           # {编,章,节}
                "chunk_index": i,
                "text": c["text"]
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] {args.law_name} 解析到 {len(articles)} 条；导出切块 {len(chunks)} 条 → {args.out}")

if __name__ == "__main__":
    main()
