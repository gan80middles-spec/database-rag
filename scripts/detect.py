#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
detect.py

功能：
- 针对“××关于集中修改和废止……的决定”这类文本，
  解析出每一部被“修改/废止”的法规，并逐条输出：
    -《某某条例》[已被修改]
    -《某某条例》[现已失效]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

# ========= 通用工具 =========

CN_NUM = "一二三四五六七八九十百千零〇○两"

STRICT_MODIFY_KWS = [
    "修改为",
    "改为",
    "变更为",
    "删去",
    "删除",
    "增加一条",
    "增加一款",
    "增加一项",
    "增加一条作为",
    "增加一款作为",
    "增加一项作为",
]


def normalize_space(text: str) -> str:
    """把各种空白折叠成一个空格，便于正则匹配。"""
    return re.sub(r"\s+", " ", text)


def norm_title(name: str) -> str:
    """
    归一化标题用于去重：
    - 中英文括号统一成中文括号
    - 去掉所有空格
    """
    name = name.replace("(", "（").replace(")", "）")
    name = re.sub(r"\s+", "", name)
    return name


# ========= 1. 抽取“被修改”的法规 =========

def extract_modified_laws(text: str) -> List[str]:
    """
    返回所有被“修改”的法规名（带《》），例如：
    [
        "《鞍山市城市市容和环境卫生管理条例》",
        "《鞍山市市政工程设施管理条例》",
        ...
    ]

    覆盖的几种常见结构：
      1) 标题：××关于修改《A》…的决定
      2) 条目：一、修改《A》……
      3) 条目：一、对《A》作出/作如下/予以/相应修改
      4) 条目：（一）《A》 …… 将第N条修改为/删去/增加……
    """
    normalized = normalize_space(text)
    hits: List[tuple[int, str]] = []

    # --- 1) 标题里的“关于修改《……》的决定”（仅看前 400 字符） ---
    header = normalized[:400]
    m = re.search(r"关于修改[《〈](?P<law>.+?)[》〉].*?的决定", header)
    if m:
        law = f"《{m.group('law').strip()}》"
        hits.append((m.start(), law))

    # --- 2) 一、修改《××》…… ---
    pat_bullet_modify = re.compile(
        rf"[{CN_NUM}]{{1,3}}、\s*修改[《〈](?P<law>.+?)[》〉]"
    )
    for m in pat_bullet_modify.finditer(normalized):
        law = f"《{m.group('law').strip()}》"
        hits.append((m.start(), law))

    # --- 3) 一、对《××》作出/作如下/予以/相应修改 ---
    pat_bullet_modify2 = re.compile(
        rf"[{CN_NUM}]{{1,3}}、\s*对[《〈](?P<law>.+?)[》〉](?:作出|作如下|予以|相应)修改"
    )
    for m in pat_bullet_modify2.finditer(normalized):
        law = f"《{m.group('law').strip()}》"
        hits.append((m.start(), law))

    # --- 4) （一）《××》 …… 将第N条修改为/删去/增加…… ---
    pat_enum_law = re.compile(
        r"（[一二三四五六七八九十百千零〇○两]{1,3}）\s*[《〈](?P<law>.+?)[》〉]"
    )
    for m in pat_enum_law.finditer(normalized):
        law = f"《{m.group('law').strip()}》"

        # 只看这条后面的一个小窗口，必须真的出现“修改为/删去/增加一条…”这类操作词
        start = m.end()
        end = min(len(normalized), start + 240)
        window = normalized[start:end]
        no_paren = re.sub(r"[（(][^）)]*[）)]", "", window)

        if not any(kw in no_paren for kw in STRICT_MODIFY_KWS):
            continue

        hits.append((m.start(), law))

    if not hits:
        return []

    # 按顺序排序 + 通过“规范化标题”去重（解决 (二)/(二) 这种差异）
    hits.sort(key=lambda x: x[0])
    laws: List[str] = []
    seen_norm = set()
    for _, law in hits:
        key = norm_title(law)
        if key in seen_norm:
            continue
        seen_norm.add(key)
        laws.append(law)

    return laws


# ========= 2. 抽取“被废止”的法规 =========

def extract_repealed_laws(text: str) -> List[str]:
    """
    抽取“被废止的法规列表”，典型句式：

      决定废止下列地方性法规：
      一、《A条例》……
      二、《B条例》……
      …

    或者其它包含“废止……。”的句子。
    """
    normalized = normalize_space(text)
    raw_laws: List[str] = []

    # 1) 主通路：所有包含“废止”的句子
    for m in re.finditer(r"废止(?P<body>.+?)(?:。|；|;|$)", normalized):
        seg = m.group("body")
        titles = re.findall(r"[《〈](.+?)[》〉]", seg)
        raw_laws.extend([t.strip() for t in titles])

    # 2) 兜底：如果完全没匹配到句子，就用 “废止《……》” 简单形式
    if not raw_laws:
        for m in re.finditer(r"废止[《〈](.+?)[》〉]", normalized):
            raw_laws.append(m.group(1).strip())

    # 过滤掉明显不是“法规本体名”的条目
    bad_kw = [
        "关于废止",
        "关于修改",
        "提请废止",
        "提请修改",
        "的决定",
        "的议案",
        "公告",
    ]

    def is_non_statute(name: str) -> bool:
        return any(k in name for k in bad_kw)

    laws: List[str] = []
    seen_norm = set()
    for name in raw_laws:
        if is_non_statute(name):
            continue
        full = f"《{name}》"
        key = norm_title(full)
        if key in seen_norm:
            continue
        seen_norm.add(key)
        laws.append(full)

    return laws


def analyze_law_text(text: str) -> dict:
    """如果你后面要程序里调用，可以用这个函数。"""
    return {
        "modified_laws": extract_modified_laws(text),
        "repealed_laws": extract_repealed_laws(text),
    }


# ========= 3. 文件读取 + CLI =========

def read_text_file(path: Path, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        for enc in ("gb18030", "gbk", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_bytes().decode("latin-1", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检测法规‘决定类’文本中每一部被修改/被废止的法规。"
    )
    parser.add_argument("file", type=str, help="待检查的文本文件路径")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"文件不存在：{path}")

    text = read_text_file(path)

    modified = extract_modified_laws(text)
    repealed = extract_repealed_laws(text)

    # 你之前要的输出格式：
    # -《……》[已被修改]
    # （空行）
    # -《……》[现已失效]
    for law in modified:
        print(f"-{law}[已被修改]")

    if modified and repealed:
        print()

    for law in repealed:
        print(f"-{law}[现已失效]")


if __name__ == "__main__":
    main()
