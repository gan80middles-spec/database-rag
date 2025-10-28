#!/usr/bin/env python3
"""Update doc_type and create doctype_detail in JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def find_jsonl_files(root: Path) -> Iterable[Path]:
    """Yield all .jsonl files under ``root`` recursively."""
    for path in root.rglob("*.jsonl"):
        if path.is_file():
            yield path


def process_file(path: Path) -> int:
    """Update a JSONL file, returning the number of records modified."""
    updated_lines = []
    modified_count = 0

    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                updated_lines.append(line)
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_number} of {path}") from exc

            original_doc_type = record.get("doc_type")
            record["doc_type"] = "judgment"

            if original_doc_type is not None:
                record["doctype_detail"] = original_doc_type
            else:
                record.pop("doctype_detail", None)

            updated_lines.append(json.dumps(record, ensure_ascii=False) + "\n")
            modified_count += 1

    with path.open("w", encoding="utf-8") as outfile:
        outfile.writelines(updated_lines)

    return modified_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Update every JSON object in JSONL files so that doc_type becomes "
            "'judgment' and the previous value is stored in doctype_detail."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to the directory containing JSONL files to update.",
    )
    args = parser.parse_args()

    directory = args.directory
    if not directory.exists():
        raise SystemExit(f"Directory does not exist: {directory}")

    total_records = 0
    total_files = 0

    for jsonl_file in find_jsonl_files(directory):
        total_files += 1
        total_records += process_file(jsonl_file)

    print(
        f"Processed {total_records} records across {total_files} JSONL files "
        f"under {directory.resolve()}"
    )


if __name__ == "__main__":
    main()
