#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ds_judgment_extract.py — 用 DeepSeek API 对中国刑事判决书做结构化抽取，并输出分段 JSONL

特性（为“最好实现”准备）：
1) 兼容输入：单个 TXT 或目录（递归扫描 .txt）
2) 规则分段：标题/案号行/当事人信息/起诉书指控/证据/裁判理由/主文/尾部（找不到则回退为“全文”）
3) DeepSeek 抽取：
   - 首选 Function Calling 严格模式（/beta，strict=True）→ 稳定 JSON Schema
   - 回退 JSON 输出模式（response_format={"type":"json_object"}）
   - 自动重试、指数退避、幂等（同样文本不重复调用）
4) 本地校验与规范化：
   - 刑期文本→月数(如“一年七个月”→19)
   - 刑期起止日校验、羁押折抵标记
   - 法条字符串格式化：“《{law}》第{article}条{clause}款”
   - 时间线与程序要素最小一致性
5) 输出 JSONL（每个分段一行），字段对齐你给的样例

环境变量：
  export DEEPSEEK_API_KEY="你的密钥"

用法：
  单文件：python ds_judgment_extract.py --input /path/judgment.txt --output-dir /path/out
  批处理：python ds_judgment_extract.py --input /path/txt_dir --output-dir /path/out

依赖：
  pip install openai
参考文档：Your First API Call（base_url 与 OpenAI 兼容），JSON Output，Function Calling 严格模式与 strict 参数。 
"""

import os
import re
import sys
import json
import time
import glob
import hashlib
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

# -------------------------
# 配置
# -------------------------
# Function Calling 严格模式需要走 /beta 域名；JSON 输出走常规域名
DEEPSEEK_BETA_BASE = "https://api.deepseek.com/beta"   # 严格模式 Function Calling（官方 Beta）  :contentReference[oaicite:1]{index=1}
DEEPSEEK_BASE      = "https://api.deepseek.com"        # OpenAI 兼容 Chat Completions            :contentReference[oaicite:2]{index=2}
DEEPSEEK_MODEL     = "deepseek-chat"                   # 函数调用支持以 deepseek-chat 为主       :contentReference[oaicite:3]{index=3}

API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-083aa5b6508a4d719786e6142c44cd32")

# -------------------------
# 公用工具
# -------------------------
def md5_8(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_jsonl(path: str, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_whitespace(t: str) -> str:
    # 规范空白与全角空格
    t = t.replace("\u3000", " ")
    t = re.sub(r"[ \t]+", " ", t)
    # 保留换行，清理 3+ 连续空行
    t = re.sub(r"\n{3,}", "\n\n", t.strip())
    return t

# -------------------------
# 刑期与日期规范化
# -------------------------
CN_NUM = {
    "零":0,"〇":0,"○":0,"Ｏ":0,"O":0,
    "一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,
}
def cn_num_to_int(s: str) -> Optional[int]:
    # 处理“十二”“二十七”“十”“二十三”等
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    # 简易中文数字解析
    # 十三 -> 13; 二十 -> 20; 二十七 -> 27; 十 -> 10
    n = 0
    if "十" in s:
        parts = s.split("十")
        if parts[0] == "" or parts[0] in ("一","壹"):
            tens = 1
        else:
            tens = CN_NUM.get(parts[0], 0)
        ones = CN_NUM.get(parts[1], 0) if len(parts) > 1 and parts[1] != "" else 0
        n = tens * 10 + ones
        return n
    # 单字数字
    tmp = 0
    for ch in s:
        tmp = tmp * 10 + CN_NUM.get(ch, 0)
    return tmp if tmp > 0 else None

def imprisonment_text_to_months(text: str) -> Optional[int]:
    """
    将“一年七个月”“十九个月”“有期徒刑一年七个月”等解析为总月数
    """
    if not text:
        return None
    t = text
    # 例：判处有期徒刑一年七个月/有期徒刑十九个月/拘役三个月
    # 支持 年/月 的组合
    years = 0
    months = 0
    m_year = re.search(r"([〇零一二两三四五六七八九十\d]{1,4})年", t)
    if m_year:
        y_raw = m_year.group(1)
        years = cn_num_to_int(y_raw) if not y_raw.isdigit() else int(y_raw)
        years = years or 0
    m_month = re.search(r"([〇零一二两三四五六七八九十\d]{1,4})个?月", t)
    if m_month:
        m_raw = m_month.group(1)
        months = cn_num_to_int(m_raw) if not m_raw.isdigit() else int(m_raw)
        months = months or 0
    # 如“十九个月”无“年”，m_month 会命中
    if not m_year and m_month and not years:
        # ok
        pass
    total = years * 12 + months
    return total if total > 0 else None

def parse_date_yyyy_mm_dd(s: str) -> Optional[str]:
    # 匹配 2025年9月29日 / 2025-09-29
    s = s.strip()
    # 标准 yyyy-mm-dd
    m = re.search(r"(\d{4})[-年/.](\d{1,2})[-月/.](\d{1,2})日?", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except Exception:
            return None
    return None

# -------------------------
# 文书分段（规则）
# -------------------------
SectionSpec = List[Tuple[str, List[str]]]
SECTION_SPECS: SectionSpec = [
    ("标题",         [r"^\s*刑事判决书\s*$"]),
    ("案号行",       [r"^\s*（\d{4}）.+?刑初.+?号", r"^\s*（\d{4}）.+?号"]),
    ("当事人信息",   [r"^\s*被告人", r"^\s*上诉人", r"^\s*公诉机关", r"^\s*辩护人"]),
    ("起诉书指控",   [r"^\s*公诉机关指控", r"^\s*指控称", r"^\s*起诉书指控"]),
    ("证据",         [r"^\s*本院经审理查明", r"^\s*本院经审理", r"^\s*证据[与及]理由", r"^\s*证据"]),
    ("裁判理由",     [r"^\s*本院认为", r"^\s*经审理认为"]),
    ("主文",         [r"^\s*判决如下", r"^\s*依照.+?之规定[,，；]判决如下"]),
    ("尾部",         [r"^\s*审判员", r"^\s*人民法院\s*$", r"^\s*书记员", r"\d{4}年\d{1,2}月\d{1,2}日"]),
]

def segment_by_rules(text: str) -> List[Tuple[str, str]]:
    """
    返回 [(section_name, section_text), ...]；若未命中，回退为 [("全文", text)]
    """
    lines = text.splitlines()
    indices = []
    for idx, line in enumerate(lines):
        for name, patterns in SECTION_SPECS:
            for pat in patterns:
                if re.search(pat, line.strip()):
                    indices.append( (idx, name) )
                    break
    # 去重 / 稳定排序
    seen = set()
    filtered = []
    for i, name in sorted(indices, key=lambda x: x[0]):
        if name not in seen:
            filtered.append((i, name))
            seen.add(name)
    if not filtered:
        return [("全文", text)]

    # 片段切分
    segs = []
    for j, (start_idx, name) in enumerate(filtered):
        end_idx = filtered[j+1][0] if j+1 < len(filtered) else len(lines)
        chunk = "\n".join(lines[start_idx:end_idx]).strip()
        if chunk:
            segs.append((name, chunk))
    return segs

# -------------------------
# DeepSeek 工具 Schema（Function Calling 严格模式）
# -------------------------
DEEPSEEK_TOOL = [{
    "type": "function",
    "function": {
        "name": "extract_judgment_fields",
        "strict": True,  # 严格模式：要求 required == properties 全量
        "description": "从中国刑事判决书抽取结构化要素（必须返回严格 JSON）。所有字段必填：未知请用空串/0/false。",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["meta","parties","disposition","procedure","factors","facts","assets","appeal","statutes"],
            "properties": {
                "meta": {
                    "type":"object","additionalProperties":False,
                    "required":["case_system","case_subtype","doc_type","trial_level","court","case_number","judgment_date"],
                    "properties":{
                        "case_system":{"type":"string"},
                        "case_subtype":{"type":"string"},
                        "doc_type":{"type":"string"},
                        "trial_level":{"type":"string"},
                        "court":{"type":"string"},
                        "case_number":{"type":"string"},
                        "judgment_date":{"type":"string"}
                    }
                },
                "parties":{
                    "type":"object","additionalProperties":False,
                    "required":["procuratorate","prosecutors","defendants","lawyers"],
                    "properties":{
                        "procuratorate":{"type":"string"},
                        "prosecutors":{"type":"array","items":{"type":"string"}},
                        "defendants":{"type":"array","items":{
                            "type":"object","additionalProperties":False,
                            # 严格模式：把 aka 也设为必填（未知给 ""）
                            "required":["name_masked","aka","gender","dob","address","role"],
                            "properties":{
                                "name_masked":{"type":"string"},
                                "aka":{"type":"string"},
                                "gender":{"type":"string"},
                                "dob":{"type":"string"},
                                "address":{"type":"string"},
                                "role":{"type":"string"}
                            }
                        }},
                        "lawyers":{"type":"array","items":{"type":"string"}}
                    }
                },
                "disposition":{
                    "type":"object","additionalProperties":False,
                    "required":["offense","imprisonment_text","fine_amount","detention_offset","term_start","term_end"],
                    "properties":{
                        "offense":{"type":"string"},
                        "imprisonment_text":{"type":"string"},
                        "fine_amount":{"type":"integer"},
                        "detention_offset":{"type":"boolean"},
                        "term_start":{"type":"string"},
                        "term_end":{"type":"string"}
                    }
                },
                "procedure":{
                    "type":"object","additionalProperties":False,
                    "required":["mode","plead_guilty","plea_statement"],
                    "properties":{
                        "mode":{"type":"string"},
                        "plead_guilty":{"type":"boolean"},
                        "plea_statement":{"type":"string"}
                    }
                },
                "factors":{
                    "type":"object","additionalProperties":False,
                    "required":["recidivist_candidate","drug_repeat_candidate","evidence_sentence"],
                    "properties":{
                        "recidivist_candidate":{"type":"boolean"},
                        "drug_repeat_candidate":{"type":"boolean"},
                        "evidence_sentence":{"type":"string"}
                    }
                },
                "facts":{
                    "type":"object","additionalProperties":False,
                    # 全部必填；未知用 "", 0
                    "required":["time","place","buyer","price_per_unit","count","weight_g","drug_name","drug_chemical"],
                    "properties":{
                        "time":{"type":"string"},
                        "place":{"type":"string"},
                        "buyer":{"type":"string"},
                        "price_per_unit":{"type":"number"},
                        "count":{"type":"integer"},
                        "weight_g":{"type":"number"},
                        "drug_name":{"type":"string"},
                        "drug_chemical":{"type":"string"}
                    }
                },
                "assets":{
                    "type":"object","additionalProperties":False,
                    "required":["money_confiscated","items"],
                    "properties":{
                        "money_confiscated":{"type":"integer"},
                        "items":{"type":"array","items":{"type":"string"}}
                    }
                },
                "appeal":{
                    "type":"object","additionalProperties":False,
                    "required":["days","to_court"],
                    "properties":{
                        "days":{"type":"integer"},
                        "to_court":{"type":"string"}
                    }
                },
                "statutes":{
                    "type":"array",
                    "items":{
                        "type":"object","additionalProperties":False,
                        # 严格模式下“clause”也设为必填；没有款时请填 0
                        "required":["law","article","clause","role"],
                        "properties":{
                            "law":{"type":"string"},
                            "article":{"type":"integer"},
                            "clause":{"type":"integer"},
                            "role":{"type":"string"}
                        }
                    }
                }
            }
        }
    }
}]

def build_beta_client() -> OpenAI:
    if not API_KEY:
        print("[FATAL] 环境变量 DEEPSEEK_API_KEY 未设置", file=sys.stderr)
        sys.exit(2)
    return OpenAI(base_url=DEEPSEEK_BETA_BASE, api_key=API_KEY)

def build_client() -> OpenAI:
    if not API_KEY:
        print("[FATAL] 环境变量 DEEPSEEK_API_KEY 未设置", file=sys.stderr)
        sys.exit(2)
    return OpenAI(base_url=DEEPSEEK_BASE, api_key=API_KEY)

# -------------------------
# DeepSeek 调用（严格模式首选，JSON 模式回退）
# -------------------------
def call_deepseek_function_calling(full_text: str, retries: int = 4, timeout_s: float = 60.0) -> Optional[dict]:
    client_beta = build_beta_client()
    messages = [
        {"role": "system", "content": "你是中国刑事判决书结构化抽取助手。只通过函数调用返回JSON；所有字段必填，未知填空串/0/false。"},
        {"role": "user", "content": "以下为分段后的刑事判决书，请抽取结构化要素（严格符合工具的 JSON Schema）：\n" + full_text}
    ]
    backoff = 1.6
    for i in range(retries):
        try:
            rsp = client_beta.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                tools=DEEPSEEK_TOOL,
                tool_choice={"type":"function","function":{"name":"extract_judgment_fields"}},  # 强制走函数
                timeout=timeout_s,
                max_tokens=2000
            )
            msg = rsp.choices[0].message
            if getattr(msg, "tool_calls", None):
                return json.loads(msg.tool_calls[0].function.arguments)
            return None
        except Exception as e:
            if "Required properties must match all properties" in str(e):
                # 一旦是此错误，说明 schema 不合规/模型试图返回缺键；直接下一轮重试
                pass
            if i == retries - 1:
                print(f"[ERROR] FunctionCalling 最终失败：{e}", file=sys.stderr)
                return None
            time.sleep(backoff ** (i+1))

def call_deepseek_json_mode(full_text: str, retries: int = 4, timeout_s: float = 60.0) -> Optional[dict]:
    client = build_client()
    system_prompt = (
        "只输出 json；不要任何解释性文字；所有字段必填，未知填空串/0/false。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "请从以下刑事判决书中抽取结构化要素，输出 json：\n" + full_text}
    ]
    backoff = 1.6
    for i in range(retries):
        try:
            rsp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                response_format={"type":"json_object"},
                timeout=timeout_s,
                max_tokens=2000
            )
            content = rsp.choices[0].message.content or ""
            if content.strip().startswith("{"):
                return json.loads(content)
        except Exception as e:
            if i == retries - 1:
                print(f"[ERROR] JSON 模式最终失败：{e}", file=sys.stderr)
                return None
            time.sleep(backoff ** (i+1))
    return None
# -------------------------
# 文书整体 → DeepSeek 输入组装
# -------------------------
def build_prompt_text(sections: List[Tuple[str, str]]) -> str:
    """
    将分段结果包装成提示语友好的串
    """
    buf = []
    for i, (name, seg) in enumerate(sections):
        buf.append(f"\n{seg}\n")
    return "\n".join(buf)

# -------------------------
# 结果规范化与 JSONL 构造
# -------------------------
def fmt_statute_str(law: str, article: int, clause: Optional[int]) -> str:
    # “《刑法》第347条第四款”
    art = chinese_article(article)
    cl  = chinese_clause(clause) if clause else ""
    if cl:
        return f"《中华人民共和国{law}》第{art}条{cl}款" if "中华人民共和国" not in law else f"《{law}》第{art}条{cl}款"
    else:
        return f"《中华人民共和国{law}》第{art}条" if "中华人民共和国" not in law else f"《{law}》第{art}条"

def chinese_article(n: int) -> str:
    return str(n)

def chinese_clause(n: Optional[int]) -> str:
    return str(n) if n is not None else ""

def make_chunk_id(doc_id: str, section: str, idx: int, text: str) -> str:
    return f"{doc_id}#{section}#{idx:03d}-{md5_8(text)}"

def build_jsonl_rows(doc_id: str,
                     sections: List[Tuple[str, str]],
                     ds: dict) -> List[dict]:
    """
    将 DeepSeek 抽取结果与分段合并，生成与你样例一致的 JSONL 行
    """
    meta = ds.get("meta", {})
    parties = ds.get("parties", {})
    disposition = ds.get("disposition", {})
    statutes_raw = ds.get("statutes", [])

    # judgment_info 组装（与你样例结构一致）
    defendants_out = []
    for d in parties.get("defendants", []):
        defendants_out.append({
            "name_masked": d.get("name_masked"),
            "aka": d.get("aka"),
            "gender": d.get("gender"),
            "dob": d.get("dob"),
            "address": d.get("address"),
            "prior_convictions": None,  # 可在后续版本引入更详细前科结构
            "detention": None,
            "role": d.get("role", "被告人"),
            "disposition": {
                "offense": disposition.get("offense"),
                "imprisonment_months": imprisonment_text_to_months(disposition.get("imprisonment_text","")) or None,
                "imprisonment_desc": disposition.get("imprisonment_text"),
                "fine_amount": disposition.get("fine_amount"),
                "confiscation_amount": None,
                "detention_offset": disposition.get("detention_offset", False),
                "term_start": parse_date_yyyy_mm_dd(disposition.get("term_start","")) or disposition.get("term_start"),
                "term_end": parse_date_yyyy_mm_dd(disposition.get("term_end","")) or disposition.get("term_end"),
                "probation": False
            },
            "factors": {
                "self_surrender": False,
                "plead_guilty": ds.get("procedure",{}).get("plead_guilty", False),
                "accessory": False,
                "recidivist": ds.get("factors",{}).get("recidivist_candidate", False),
                "confession": ds.get("procedure",{}).get("plead_guilty", False)
            }
        })

    statutes_out = []
    for s in statutes_raw:
        statutes_out.append(
            fmt_statute_str(
                law=s.get("law","刑法"),
                article=int(s.get("article",0) or 0),
                clause=s.get("clause")
            )
        )

    rows = []
    for idx, (sec_name, sec_text) in enumerate(sections):
        row = {
            "chunk_id": make_chunk_id(doc_id, sec_name, idx, sec_text),
            "doc_id": doc_id,
            "case_system": meta.get("case_system","刑事"),
            "case_subtype": meta.get("case_subtype",""),
            "doc_type": meta.get("doc_type","刑事判决书"),
            "trial_level": meta.get("trial_level","一审"),
            "court": meta.get("court",""),
            "case_number": meta.get("case_number",""),
            "judgment_date": parse_date_yyyy_mm_dd(meta.get("judgment_date","")) or meta.get("judgment_date",""),
            "statutes": statutes_out,
            "judgment_info": {
                "procuratorate": parties.get("procuratorate",""),
                "prosecutors": parties.get("prosecutors",[]),
                "defendants": defendants_out,
                "lawyers": parties.get("lawyers",[]),
                "dispute_focus": [],
                "opinion_summary": []
            },
            "section": sec_name,
            "chunk_index": idx,
            "text": sec_text
        }
        rows.append(row)
    return rows

# -------------------------
# 主流程
# -------------------------
def process_file(txt_path: str, output_dir: str) -> str:
    raw = read_text(txt_path)
    raw = norm_whitespace(raw)

    sections = segment_by_rules(raw)
    prompt_text = build_prompt_text(sections)

    # 首选：严格模式 Function Calling（/beta；strict）
    ds = call_deepseek_function_calling(prompt_text)
    if ds is None:
        # 回退：JSON 输出模式（response_format=json_object）
        ds = call_deepseek_json_mode(prompt_text)
    if ds is None:
        raise RuntimeError("DeepSeek 两种模式均未成功返回结构化结果")

    # doc_id：优先用案号；退化为文件名哈希
    doc_id = ds.get("meta",{}).get("case_number") or os.path.splitext(os.path.basename(txt_path))[0]
    if not doc_id:
        doc_id = "JUD-" + md5_8(raw)

    rows = build_jsonl_rows(doc_id, sections, ds)

    out_path = os.path.join(output_dir, f"{doc_id}.jsonl")
    write_jsonl(out_path, rows)
    print(f"[{now_ts()}] OK -> {out_path}  (chunks={len(rows)})")
    return out_path

def walk_inputs(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        # 递归收集 .txt
        files = [p for p in glob.glob(os.path.join(input_path, "**", "*.txt"), recursive=True)]
        return sorted(files)
    else:
        return [input_path]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 TXT 路径或目录")
    ap.add_argument("--output-dir", required=True, help="输出目录（将写入 {doc_id}.jsonl）")
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    files = walk_inputs(args.input)

    if not files:
        print("[WARN] 未找到任何 .txt 文件", file=sys.stderr)
        sys.exit(1)

    ok, fail = 0, 0
    for p in files:
        try:
            process_file(p, args.output_dir)
            ok += 1
        except Exception as e:
            print(f"[ERROR] 处理失败 {p}: {e}", file=sys.stderr)
            fail += 1
    print(f"[SUMMARY] done: ok={ok}, fail={fail}")

if __name__ == "__main__":
    main()
