"""Command-line tool to classify contract text files into predefined categories.

Usage examples:

    python classify_contracts.py --input_dir ./contracts --output_dir ./out_contracts

The script scans the input directory for text-based contract files, scores them
against six keyword-driven categories, and copies each file into the matching
sub-directory under the output directory. Classification details are stored in
``classified.jsonl`` and contracts that remain uncertain are also listed in
``uncertain.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


RAW_KEYWORDS = {
    "buy_sell": "采购｜买卖｜购销｜供货｜交货｜验收",
    "lease": "租赁｜出租｜承租｜租金｜押金｜续租｜退租",
    "labor": "劳动合同｜用工｜聘用｜试用期｜社保｜工资｜加班｜解除｜竞业",
    "outsourcing": "服务外包｜技术服务｜咨询服务｜运维｜SLA｜服务等级｜里程碑｜服务费",
    "nda": "保密协议｜不披露协议｜NDA｜披露方｜接收方｜保密信息｜保密期限",
    "software": "软件许可｜使用许可｜授权协议｜SaaS｜API｜数据许可｜数据使用协议｜DPA｜著作权｜用户数",
}


KEYWORDS = {
    label: tuple(keyword.strip() for keyword in keywords.split("｜") if keyword.strip())
    for label, keywords in RAW_KEYWORDS.items()
}


NDA_STRONG_ANCHORS = [
    r"保密协议",
    r"不披露协议",
    r"\bNDA\b",
    r"保密信息",
    r"披露方",
    r"接收方",
    r"保密期限",
]

NDA_EXCLUDES_INSURANCE = [
    r"保险",
    r"保单",
    r"被保险人",
    r"投保人",
    r"身故",
    r"保险金额",
    r"现金价值",
    r"核保",
    r"退保",
    r"宽限期",
    r"保费",
]

_anchor_re = [re.compile(pattern, re.IGNORECASE) for pattern in NDA_STRONG_ANCHORS]
_excl_re = [re.compile(pattern, re.IGNORECASE) for pattern in NDA_EXCLUDES_INSURANCE]


def _count_matches(patterns, text: str) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _nda_gate(title: str, body: str) -> bool:
    """Return True if the NDA label should be allowed based on strong anchors."""

    title = title or ""
    body = body or ""

    anchors = _count_matches(_anchor_re, title) + _count_matches(_anchor_re, body)
    has_insurance = any(pattern.search(title) or pattern.search(body) for pattern in _excl_re)

    return anchors >= 2 and not has_insurance


SUPPORTED_TEXT_EXTENSIONS = {".txt"}
SPECIAL_SKIP_EXTENSIONS = {".pdf", ".docx"}

ALL_LABELS = list(KEYWORDS.keys()) + ["uncertain", "error"]


def load_text(path: Path, encoding: str, max_chars: int) -> str:
    """Load up to ``max_chars`` characters from ``path`` using the given encoding.

    If decoding with the preferred encoding fails, a GBK fallback is attempted.
    The caller is expected to catch ``UnicodeError`` or ``OSError`` for failure
    cases.
    """

    try:
        with path.open("r", encoding=encoding) as fh:
            return fh.read(max_chars)
    except UnicodeDecodeError:
        with path.open("r", encoding="gbk") as fh:
            return fh.read(max_chars)


def score(title: str, body: str) -> Dict[str, int]:
    """Return category scores for the provided title and body snippets."""

    title_lower = title.lower()
    body_lower = body.lower()
    scores: Dict[str, int] = {label: 0 for label in KEYWORDS}

    for label, keywords in KEYWORDS.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title_lower:
                scores[label] += 2

            occurrences = body_lower.count(keyword_lower)
            if occurrences:
                scores[label] += min(occurrences, 2)

    return scores


def _classify_scores(scores: Dict[str, int]) -> str:
    """Classify based on the provided ``scores`` dict."""

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    runner_up_score = max(
        (value for label, value in scores.items() if label != best_label),
        default=0,
    )

    if best_score >= 3 and (best_score - runner_up_score) >= 1:
        return best_label
    return "uncertain"


def classify(title: str, body: str) -> Tuple[str, Dict[str, int]]:
    """Compute scores for ``title`` and ``body`` and return the label and scores."""

    scores = score(title, body)
    label = _classify_scores(scores)
    if label == "nda" and not _nda_gate(title, body):
        label = "uncertain"
    return label, scores


def ensure_unique_name(target_dir: Path, filename: str) -> Path:
    """Generate a unique file path within ``target_dir`` for ``filename``."""

    base = Path(filename)
    candidate = target_dir / base.name
    if not candidate.exists():
        return candidate

    stem = base.stem
    suffix = base.suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_to(out_dir: Path, label: str, src_path: Path, dry_run: bool = False) -> Path:
    """Copy ``src_path`` into ``out_dir / label`` and return the destination path."""

    dest_dir = out_dir / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = ensure_unique_name(dest_dir, src_path.name)

    if not dry_run:
        shutil.copy2(src_path, dest_path)

    return dest_path


def iter_files(directory: Path) -> Iterable[Path]:
    """Yield files within ``directory`` recursively, sorted for stability."""

    for path in sorted(directory.rglob("*")):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("./contracts"))
    parser.add_argument("--output_dir", type=Path, default=Path("./out_contracts"))
    parser.add_argument("--read_bytes", type=int, default=2000,
                        help="Maximum number of characters to read from each file body.")
    parser.add_argument("--encoding", default="utf-8",
                        help="Preferred encoding when reading text files (fallback to GBK).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process files without copying them to the output directory.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    classified_path = output_dir / "classified.jsonl"
    uncertain_path = output_dir / "uncertain.jsonl"

    stats = Counter({label: 0 for label in ALL_LABELS})
    uncertain_entries = []

    with classified_path.open("w", encoding="utf-8") as classified_file, \
            uncertain_path.open("w", encoding="utf-8") as uncertain_file:

        for file_path in iter_files(input_dir):
            relative = file_path.relative_to(input_dir)
            logging.debug("Processing %s", relative)

            ext = file_path.suffix.lower()
            title = file_path.stem
            scores: Dict[str, int] = {label: 0 for label in KEYWORDS}
            label = "uncertain"
            err_message = None

            if ext in SPECIAL_SKIP_EXTENSIONS:
                logging.info("Skipping unsupported file type %s -> marked as uncertain", relative)
            elif ext in SUPPORTED_TEXT_EXTENSIONS:
                try:
                    body = load_text(file_path, args.encoding, args.read_bytes)
                except (OSError, UnicodeError) as err:
                    label = "error"
                    err_message = str(err)
                    logging.error("Failed to read %s: %s", relative, err)
                else:
                    label, scores = classify(title, body)
            else:
                logging.info("Unknown extension for %s -> marked as uncertain", relative)

            if label != "error":
                copy_label = label
            else:
                copy_label = "error"

            if not args.dry_run:
                copy_to(output_dir, copy_label, file_path, dry_run=False)
            else:
                logging.info("Dry run: skipping copy of %s", relative)

            record = {
                "path": str(file_path.resolve()),
                "label": label,
                "scores": scores,
            }
            if err_message:
                record["err"] = err_message

            classified_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if label == "uncertain":
                uncertain_entries.append(record)

            stats[label] += 1

        for entry in uncertain_entries:
            uncertain_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logging.info("Classification complete.")
    for label in ALL_LABELS:
        logging.info("%s: %d", label, stats[label])

    total_processed = sum(stats.values())
    print("Processed files:", total_processed)
    for label in ALL_LABELS:
        print(f"  {label}: {stats[label]}")


if __name__ == "__main__":
    main()
