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


def ensure_doc_from_chunks(mc, short_name: str, title_regex: str,
                           doc_type="judicial_interpretation", validity="valid"):
    """
    用 chunk.title 的正则找到 doc_id，在 law_kb_docs 里 upsert：
      - law_name 用短名（你希望的规范名）
      - law_alias 加入该解释的“全称标题”
      - *不会* 改 chunk，只建/修 docs
    """
    C = mc[DB_NAME]["law_kb_chunks"]
    D = mc[DB_NAME]["law_kb_docs"]
    q = {"doc_type": doc_type, "title": {"$regex": title_regex}}
    cur = list(C.find(q, {"doc_id": 1, "title": 1, "version_date": 1}))
    if not cur:
        print(f"[WARN] chunks not found by title /{title_regex}/")
        return 0

    ops = []
    for d in cur:
        did = d.get("doc_id")
        title = d.get("title") or short_name
        if not did:
            continue
        ops.append(UpdateOne(
            {"doc_id": did},
            {
                "$setOnInsert": {
                    "doc_id": did,
                    "doc_type": doc_type,
                    "law_name": short_name,
                    "law_name_norm": norm_name(short_name),
                    "law_alias": [title],
                    "law_alias_norm": sorted({norm_name(short_name), norm_name(title)}),
                    "version_date": d.get("version_date", ""),
                    "validity_status": validity,
                },
                "$set": {
                    "validity_status": validity,
                },
                "$addToSet": {
                    "law_alias": {"$each": [title]},
                    "law_alias_norm": {"$each": [norm_name(title)]},
                },
            },
            upsert=True,
        ))
    if ops:
        res = D.bulk_write(ops, ordered=False)
        print(f"[OK] ensured docs from chunks for /{title_regex}/ -> "
              f"modified={res.modified_count}, upserted={getattr(res,'upserted_count',0)}")
        return res.modified_count + getattr(res, "upserted_count", 0)
    return 0


def ensure_alias_to_doc(mc, title_regex: str, target_doc_regex: str):
    """
    把“无序号/全称”的解释作为别名，**只**挂到目标文档（比如《建工解释（一）》）。
    - title_regex: 在 chunks 里找到“无序号/全称”的解释标题
    - target_doc_regex: 在 docs 里找到真正要挂接的目标（带（一））
    """
    C = mc[DB_NAME]["law_kb_chunks"]
    D = mc[DB_NAME]["law_kb_docs"]
    src = C.find_one({"doc_type": "judicial_interpretation", "title": {"$regex": title_regex}},
                     {"title": 1})
    dst = D.find_one({"law_name": {"$regex": target_doc_regex}}, {"_id": 1, "law_name": 1})
    if not src or not dst:
        print(f"[WARN] alias attach miss: src=/{title_regex}/ dst=/{target_doc_regex}/")
        return 0
    title = src["title"]
    res = D.update_one(
        {"_id": dst["_id"]},
        {
            "$addToSet": {
                "law_alias": title,
                "law_alias_norm": norm_name(title)
            }
        }
    )
    print(f"[OK] alias attached '{title}' => {dst['law_name']}")
    return res.modified_count


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

    # --- 常见简称：国家赔偿法 / 道路交通安全法 / 保险法 / 行政许可法 ---
    ops += add_alias_ops(col, r"国家赔偿法", ["国家赔偿法"])  # 补 norm 字段
    ops += add_alias_ops(col, r"道路交通安全法", ["道交法"])
    ops += add_alias_ops(col, r"保险法", ["保险法"])  # 补 norm 字段
    ops += add_alias_ops(col, r"行政许可法", ["行许法"])

    # --- 地方性法规示例：西安市集中供热条例（匹配“集中供热条例”更稳） ---
    ops += add_alias_ops(col, r"集中供热条例", ["集中供条例"])

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

    # —— 三大诉讼法解释（用短名锁定，加全称/常用简称为别名）
    ops += add_alias_ops(col, {"law_name": "民事诉讼法解释"},
        ["最高人民法院关于适用中华人民共和国民事诉讼法的解释", "民诉法解释"])

    ops += add_alias_ops(col, {"law_name": "行政诉讼法解释"},
        ["最高人民法院关于适用中华人民共和国行政诉讼法的解释", "行诉法解释"])

    ops += add_alias_ops(col, {"law_name": "刑事诉讼法解释"},
        ["最高人民法院关于适用中华人民共和国刑事诉讼法的解释", "刑诉法解释"])

    # —— 民法典“时间效力规定”
    ops += add_alias_ops(col, {"law_name": "民法典时间效力规定"},
        ["最高人民法院关于适用中华人民共和国民法典时间效力的若干规定", "时间效力规定"])

    # —— 人身损害赔偿解释
    ops += add_alias_ops(col, {"law_name": "人身损害赔偿解释"},
        ["最高人民法院关于审理人身损害赔偿案件适用法律若干问题的解释"])

    # —— 道交赔偿解释
    ops += add_alias_ops(col, {"law_name": "道路交通事故损害赔偿解释"},
        ["最高人民法院关于审理道路交通事故损害赔偿案件适用法律若干问题的解释", "道交赔偿解释"])

    # —— 行政赔偿规定
    ops += add_alias_ops(col, {"law_name": "行政赔偿规定"},
        ["最高人民法院关于审理行政赔偿案件若干问题的规定"])

    # —— 建工解释：只加到《解释（一）》
    ops += add_alias_ops(
        col,
        {"law_name": {"$regex": r"建设工程施工合同纠纷解释（一）"},
         "version_date": {"$gte": "2021-01-01"}},
        ["最高人民法院关于审理建设工程施工合同纠纷案件适用法律问题的解释",
         "建设工程施工合同纠纷解释", "建工施工合同纠纷解释", "建工解释一"]
    )

    # —— 民间借贷规定（你库里短名是“民间借贷规定”）
    ops += add_alias_ops(col, {"law_name": "民间借贷规定"},
        ["最高人民法院关于审理民间借贷案件适用法律若干问题的规定", "最高法民间借贷规定"])

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

    # 1) 人身损害赔偿解释
    ensure_doc_from_chunks(mc,
        short_name="人身损害赔偿解释",
        title_regex=r"人身损害赔偿案件适用法律若干问题的解释")

    # 2) 道路交通事故损害赔偿解释
    ensure_doc_from_chunks(mc,
        short_name="道路交通事故损害赔偿解释",
        title_regex=r"道路交通事故损害赔偿案件适用法律若干问题的解释")

    # 3) 行政赔偿规定
    ensure_doc_from_chunks(mc,
        short_name="行政赔偿规定",
        title_regex=r"审理行政赔偿案件若干问题的规定")

    # 4) 刑事诉讼法解释
    ensure_doc_from_chunks(mc,
        short_name="刑事诉讼法解释",
        title_regex=r"适用中华人民共和国刑事诉讼法的解释")

    # 5) 建工解释（先确保《解释（一）》的文档存在；然后把“无序号全称”挂到（一））
    ensure_doc_from_chunks(mc,
        short_name="建设工程施工合同纠纷解释（一）",
        title_regex=r"建设工程施工合同纠纷案件适用法律问题的解释（一）")

    ensure_alias_to_doc(mc,
        title_regex=r"建设工程施工合同纠纷案件适用法律问题的解释(?!（一）)",  # 无（一）
        target_doc_regex=r"建设工程施工合同纠纷解释（一）")


if __name__ == "__main__":
    main()
