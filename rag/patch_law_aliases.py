# -*- coding: utf-8 -*-
# 补充 law_alias / law_name_norm / law_alias_norm，并建索引
import re, sys
from pymongo import MongoClient, UpdateOne

MONGO_URI = "mongodb://adminUser:~Q2w3e4r@192.168.110.36:27019"
DB_NAME   = "lawKB"
DOC_COL   = "law_kb_docs"

CN_PREFIX = re.compile(r"^\s*中华人民共和国")
BRACKETS  = re.compile(r"[()（）\[\]【】＜＞<>]")  # 先去括号，再去其中内容
MULTI_WS  = re.compile(r"\s+")


def norm_name(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("《", "").replace("》", "")
    s = CN_PREFIX.sub("", s)                   # 去“中华人民共和国”前缀
    s = re.sub(r"（[^）]*修正[^）]*）", "", s)   # 去“（2021年修正）”等括注
    s = re.sub(r"\([^)]*修正[^)]*\)", "", s)
    s = BRACKETS.sub("", s)                    # 去残余括号符号
    s = s.replace("　", " ").replace("·", " ")
    s = MULTI_WS.sub("", s)                    # 全去空白（也可以保留空格，看你喜好）
    return s


def add_alias_ops(col, finder, aliases):
    ops = []
    # 兼容字符串与列表传参
    if isinstance(finder, str):
        q = {"law_name": {"$regex": finder}}
    else:
        q = finder
    cur = list(col.find(q, {"_id": 1, "law_name": 1, "law_alias": 1}))
    if not cur:
        print(f"[WARN] no doc matched: {finder}")
        return ops

    for d in cur:
        law_alias = list(set((d.get("law_alias") or []) + aliases))
        # 归一化：主名 + 全部别名
        name_norm = norm_name(d.get("law_name", ""))
        alias_norm = sorted(set(norm_name(x) for x in law_alias + [d.get("law_name", "")]))
        ops.append(
            UpdateOne(
                {"_id": d["_id"]},
                {
                    "$set": {
                        "law_alias": law_alias,
                        "law_name_norm": name_norm,
                        "law_alias_norm": alias_norm,
                    }
                },
            )
        )
    return ops


def main():
    mc = MongoClient(MONGO_URI)
    col = mc[DB_NAME][DOC_COL]
    # 索引（幂等）
    col.create_index("law_name_norm")
    col.create_index("law_alias_norm")

    ops = []

    # --- 关键修复 1：民法典（合同编） —— 吞并“合同法”简称 ---
    # 你的库里民法典 doc_id 是 "CODE-CIVIL-20200528"（整部法），合同编体现在 chunk.path
    ops += add_alias_ops(
        col,
        r"民法典",
        [
            "民法典合同编",
            "合同编",
            "民法典 第三编 合同",
            "第三编 合同",
            "合同法",  # 旧称，判决书常用
        ],
    )

    # --- 关键修复 2：民诉法总解释（2022 版） ---
    ops += add_alias_ops(col, r"关于适用.*民事诉讼法.*的解释", ["民事诉讼法解释", "民诉法解释", "民诉解释"])

    # --- 关键修复 3：民法典时间效力规定 ---
    ops += add_alias_ops(col, r"关于适用.*民法典.*时间效力.*规定", ["时间效力规定", "民法典时间效力规定"])

    # --- 常见简称：国家赔偿法 / 道路交通安全法 / 保险法 / 行政许可法 ---
    ops += add_alias_ops(col, r"国家赔偿法", ["国家赔偿法"])  # 补 norm 字段
    ops += add_alias_ops(col, r"道路交通安全法", ["道交法"])
    ops += add_alias_ops(col, r"保险法", ["保险法"])  # 补 norm 字段
    ops += add_alias_ops(col, r"行政许可法", ["行许法"])

    # 民间借贷规定：把“全称”也作为别名
    ops += add_alias_ops(
        col,
        {"law_name": {"$regex": r"民间借贷规定"}},
        [
            "最高人民法院关于审理民间借贷案件适用法律若干问题的规定",
            "最高法民间借贷规定",
        ],
    )

    # --- 地方性法规示例：西安市集中供热条例（匹配“集中供热条例”更稳） ---
    ops += add_alias_ops(col, r"集中供热条例", ["集中供热条例"])

    # --- 部门规章示例：因工死亡职工供养亲属范围规定（劳社部令第18号） ---
    ops += add_alias_ops(
        col,
        r"供养亲属范围规定",
        ["因工死亡职工供养亲属范围规定", "劳社部令第18号"],
    )

    # —— 民法典总解释（建议补个“民法典解释”的简称）
    ops += add_alias_ops(
        col,
        r"关于适用.*民法典.*的解释",
        ["民法典解释"]
    )

    # —— 人格权纠纷解释（2020）
    ops += add_alias_ops(
        col,
        r"关于审理人格权纠纷案件适用法律若干问题的解释",
        ["人格权纠纷解释", "人格权解释"]
    )

    # —— 金融借款合同纠纷规定（2015）
    ops += add_alias_ops(
        col,
        r"关于审理金融借款合同纠纷案件依法适用法律若干问题的规定",
        ["金融借款合同纠纷规定", "金融借款合同规定", "金融借款合同解释"]
    )

    # —— 建设工程施工合同纠纷解释（注意：有《2004解释》与《解释（一）（2020）》）
    ops += add_alias_ops(
        col,
        {
            "law_name": {"$regex": r"建设工程施工合同纠纷案件适用法律问题的解释（一）"},
            "version_date": {"$gte": "2021-01-01"},
        },
        [
            "最高人民法院关于审理建设工程施工合同纠纷案件适用法律问题的解释",
            "建设工程施工合同纠纷解释",
            "建工施工合同纠纷解释",
            "建工解释一",
        ],
    )

    # —— 人民调解协议司法确认（2011）
    ops += add_alias_ops(
        col,
        r"人民调解协议司法确认程序.*若干规定",
        ["人民调解协议司法确认规定", "调解协议司法确认规定"]
    )

    # —— 劳动法（如果你库里只有《劳动合同法》，可临时把“劳动法”加为它的别名或补录原法）
    ops += add_alias_ops(col, r"劳动法", ["劳动法"])


    # 一次性批量提交
    if ops:
        res = col.bulk_write(ops, ordered=False)
        print(f"[OK] modified={res.modified_count}, upserted={getattr(res, 'upserted_count', 0)}")
    else:
        print("[OK] nothing to update")


if __name__ == "__main__":
    main()
