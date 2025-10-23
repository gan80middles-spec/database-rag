# -*- coding: utf-8 -*-
"""
把切块 JSONL 中的 validity_status 批量改为指定值（obsolete / repealed / valid / amended）
两种模式（二选一）：
  1) --doc_id 多次给 / 或 --doc_id_file 提供列表
  2) --law_name + --before YYYY-MM-DD   （把该法名下，早于日期的版本全标旧）

默认把结果写到 --out_dir；不传 --out_dir 则原地覆盖并生成 .bak 备份。
"""
import os, glob, json, argparse, time
from datetime import date

def parse_date(s):
    y,m,d = s.split("-")
    return date(int(y), int(m), int(d))

def iter_files(in_dir, pattern):
    for fp in sorted(glob.glob(os.path.join(in_dir, pattern))):
        yield fp

def load_docids(path):
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.strip().startswith("#")]

def should_mark(rec, mode, args, cutoff=None, docid_set=None):
    if mode == "docid":
        return rec.get("doc_id") in docid_set
    else:
        if rec.get("law_name") != args.law_name:
            return False
        vd = rec.get("version_date") or ""
        try:
            vd = parse_date(vd)
        except Exception:
            return False
        return vd < cutoff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--status", required=True, choices=["obsolete","repealed","valid","amended"])
    ap.add_argument("--out_dir", default="")  # 为空=原地覆盖，生成 .bak
    # 模式1：按 doc_id
    ap.add_argument("--doc_id", action="append", default=[])
    ap.add_argument("--doc_id_file", default="")
    # 模式2：按 law_name + before
    ap.add_argument("--law_name", default="")
    ap.add_argument("--before", default="")  # YYYY-MM-DD
    args = ap.parse_args()

    mode = None
    docid_set = set()
    cutoff = None
    if args.doc_id or args.doc_id_file:
        mode = "docid"
        if args.doc_id_file:
            docid_set.update(load_docids(args.doc_id_file))
        docid_set.update(args.doc_id)
    elif args.law_name and args.before:
        mode = "lawdate"
        cutoff = parse_date(args.before)
    else:
        raise SystemExit("请使用 模式1(--doc_id/--doc_id_file) 或 模式2(--law_name + --before)")

    os.makedirs(args.out_dir, exist_ok=True) if args.out_dir else None
    total_files, total_rows, changed = 0, 0, 0
    now = int(time.time())

    for fp in iter_files(args.in_dir, args.pattern):
        total_files += 1
        out_fp = fp if not args.out_dir else os.path.join(args.out_dir, os.path.basename(fp))
        if not args.out_dir:
            os.replace(fp, fp + ".bak")  # 先备份
            src_fp = fp + ".bak"
        else:
            src_fp = fp

        with open(src_fp, "r", encoding="utf-8") as r, open(out_fp, "w", encoding="utf-8") as w:
            for line in r:
                if not line.strip():
                    continue
                rec = json.loads(line)
                total_rows += 1
                if should_mark(rec, mode, args, cutoff=cutoff, docid_set=docid_set):
                    rec["validity_status"] = args.status
                    rec["updated_at"] = now
                    changed += 1
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] files={total_files}, rows={total_rows}, changed={changed}, status='{args.status}'")

if __name__ == "__main__":
    main()
