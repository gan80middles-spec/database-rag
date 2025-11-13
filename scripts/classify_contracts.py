"""Command-line tool to classify contract text files into predefined categories.

Usage examples:

    python classify_contracts.py --input_dir ./contracts --output_dir ./out_contracts

The script scans the input directory for text-based contract files, scores them
against six keyword-driven categories (Step 1: rule-based keywords), and then
optionally applies a vector-based semantic expansion for uncertain documents
(Step 2: centroid similarity using a Chinese embedding model).

Classification details are stored in ``classified.jsonl`` and contracts that
remain uncertain are also listed in ``uncertain.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

# ---- Optional vector dependencies -------------------------------------------------
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    np = None
    SentenceTransformer = None


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


# -----------------------------------------------------------------------------
# Step 2: Vector semantic expansion for uncertain docs
# -----------------------------------------------------------------------------


def build_embedding_text(title: str, body: str, max_chars: int) -> str:
    """Build the text fed into the embedding model: title + key body snippet."""
    title = (title or "").strip()
    body = (body or "").strip()
    if body:
        text = f"{title}\n\n{body}" if title else body
    else:
        text = title

    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def semantic_vector_step(
    docs: List[dict],
    model_name: str,
    threshold: float,
    max_chars: int,
) -> Tuple[int, int]:
    """Re-label 'uncertain' docs using vector centroids of rule-based seeds.

    Returns:
        (num_uncertain, num_upgraded)
    """
    if SentenceTransformer is None or np is None:
        logging.warning(
            "sentence-transformers or numpy is not available; "
            "skipping vector semantic step."
        )
        return 0, 0

    # Collect seeds (Step 1 confident docs) and uncertain docs
    seed_indices_by_label: Dict[str, List[int]] = {label: [] for label in KEYWORDS}
    uncertain_indices: List[int] = []

    for idx, doc in enumerate(docs):
        if doc["final_label"] == "error":
            continue

        label_step1 = doc["label_step1"]

        # Seeds: anything被 Step 1 分到六大类的
        if label_step1 in KEYWORDS and doc.get("body"):
            seed_indices_by_label[label_step1].append(idx)

        # Uncertain: Step 1 未命中的文档
        if label_step1 == "uncertain" and doc.get("body"):
            uncertain_indices.append(idx)

    labels_with_seeds = [label for label, ids in seed_indices_by_label.items() if ids]

    if not labels_with_seeds or not uncertain_indices:
        logging.info(
            "Vector step: no seeds (%d labels) or no uncertain docs (%d); skipping.",
            len(labels_with_seeds),
            len(uncertain_indices),
        )
        return len(uncertain_indices), 0

    logging.info(
        "Vector step: seeds=%d docs over %d labels, uncertain=%d docs.",
        sum(len(v) for v in seed_indices_by_label.values()),
        len(labels_with_seeds),
        len(uncertain_indices),
    )

    model = SentenceTransformer(model_name)

    # ---- Compute centroids per label ----------------------------------------
    centroids = []
    centroid_labels = []

    for label in labels_with_seeds:
        ids = seed_indices_by_label[label]
        texts = [
            build_embedding_text(docs[i]["title"], docs[i].get("body") or "", max_chars)
            for i in ids
        ]
        if not texts:
            continue

        embs = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embs = np.asarray(embs)
        centroid = embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            logging.warning("Vector step: centroid norm=0 for label %s; skipping.", label)
            continue
        centroid = centroid / norm
        centroids.append(centroid)
        centroid_labels.append(label)

    if not centroids:
        logging.warning("Vector step: no valid centroids; skipping.")
        return len(uncertain_indices), 0

    centroids_matrix = np.stack(centroids, axis=0)  # [C, D]

    # ---- Encode uncertain docs ----------------------------------------------
    texts_unc = [
        build_embedding_text(docs[i]["title"], docs[i].get("body") or "", max_chars)
        for i in uncertain_indices
    ]
    embs_unc = model.encode(
        texts_unc,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embs_unc = np.asarray(embs_unc)

    upgraded = 0

    for row_idx, doc_idx in enumerate(uncertain_indices):
        vec = embs_unc[row_idx]  # [D]
        sims = centroids_matrix @ vec  # [C], cosine similarity

        best_pos = int(np.argmax(sims))
        best_label = centroid_labels[best_pos]
        best_sim = float(sims[best_pos])

        doc = docs[doc_idx]
        doc["vector_label"] = best_label
        doc["vector_score"] = best_sim
        doc["vector_scores"] = {
            centroid_labels[i]: float(sims[i]) for i in range(len(centroid_labels))
        }

        if best_sim < threshold:
            # 语义上离各类都不够近 -> 维持 uncertain
            continue

        # 对 NDA 类，仍然套一层 anchor gate，避免保险等误伤
        title = doc["title"]
        body = doc.get("body") or ""
        candidate_label = best_label
        if candidate_label == "nda" and not _nda_gate(title, body):
            continue

        # 升级为向量预测的类别
        doc["final_label"] = candidate_label
        upgraded += 1

    return len(uncertain_indices), upgraded


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("./contracts"))
    parser.add_argument("--output_dir", type=Path, default=Path("./out_contracts"))
    parser.add_argument(
        "--read_bytes",
        type=int,
        default=2000,
        help="Maximum number of characters to read from each file body.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Preferred encoding when reading text files (fallback to GBK).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process files without copying them to the output directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    # ---- Step 2 config -------------------------------------------------------
    parser.add_argument(
        "--use_vector_step",
        action="store_true",
        help="Enable Step 2: vector semantic expansion for docs labeled 'uncertain' "
        "after keyword-based scoring.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-zh-v1.5",
        help="Name or local path of a SentenceTransformer model "
        "(e.g. 'BAAI/bge-large-zh-v1.5', 'gte-large-zh').",
    )
    parser.add_argument(
        "--vector_threshold",
        type=float,
        default=0.4,
        help="Cosine similarity threshold for Step 2 (0.35–0.45 is usually reasonable).",
    )

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

    # ---- Pass 1: Step 1 关键词分类 + 收集文本 --------------------------------
    docs: List[dict] = []

    file_paths = list(iter_files(input_dir))
    for file_path in tqdm(file_paths, desc="Classifying contracts"):
        relative = file_path.relative_to(input_dir)
        logging.debug("Processing %s", relative)

        ext = file_path.suffix.lower()
        title = file_path.stem
        scores: Dict[str, int] = {label: 0 for label in KEYWORDS}
        label = "uncertain"
        err_message = None
        body = ""

        if ext in SPECIAL_SKIP_EXTENSIONS:
            logging.info(
                "Skipping unsupported file type %s -> marked as uncertain", relative
            )
        elif ext in SUPPORTED_TEXT_EXTENSIONS:
            try:
                body = load_text(file_path, args.encoding, args.read_bytes)
            except (OSError, UnicodeError) as err:  # noqa: PERF203
                label = "error"
                err_message = str(err)
                logging.error("Failed to read %s: %s", relative, err)
            else:
                label, scores = classify(title, body)
        else:
            logging.info(
                "Unknown extension for %s -> marked as uncertain", relative
            )

        doc = {
            "path": file_path,
            "relative": str(relative),
            "ext": ext,
            "title": title,
            "body": body,
            "scores": scores,
            "label_step1": label,  # 纯规则结果
            "final_label": label,  # 最终可被 Step 2 覆盖
            "err": err_message,
        }
        docs.append(doc)

    # ---- Step 2: 向量语义扩展（可选） ---------------------------------------
    if args.use_vector_step:
        total_uncertain, upgraded = semantic_vector_step(
            docs,
            model_name=args.embedding_model,
            threshold=args.vector_threshold,
            max_chars=args.read_bytes * 2,  # 向量可读稍长一点
        )
        logging.info(
            "Vector step finished: uncertain=%d, upgraded=%d (model=%s, threshold=%.3f).",
            total_uncertain,
            upgraded,
            args.embedding_model,
            args.vector_threshold,
        )

    # ---- 输出阶段：复制文件 + 写 JSONL --------------------------------------
    classified_path = output_dir / "classified.jsonl"
    uncertain_path = output_dir / "uncertain.jsonl"

    stats = Counter({label: 0 for label in ALL_LABELS})

    with classified_path.open("w", encoding="utf-8") as classified_file, \
            uncertain_path.open("w", encoding="utf-8") as uncertain_file:

        for doc in docs:
            label = doc["final_label"]

            if label not in ALL_LABELS:
                logging.warning(
                    "Unknown final label '%s' for %s; forcing to 'uncertain'.",
                    label,
                    doc["relative"],
                )
                label = "uncertain"
                doc["final_label"] = label

            # Copy target目录
            copy_label = "error" if label == "error" else label

            if not args.dry_run:
                copy_to(output_dir, copy_label, doc["path"], dry_run=False)
            else:
                logging.info("Dry run: skipping copy of %s", doc["relative"])

            record = {
                "path": str(doc["path"].resolve()),
                "label": label,
                "scores": doc["scores"],
                "label_step1": doc["label_step1"],
            }
            if doc["err"]:
                record["err"] = doc["err"]

            if args.use_vector_step:
                record["vector_model"] = args.embedding_model
                if "vector_label" in doc:
                    record["vector_label"] = doc["vector_label"]
                    record["vector_score"] = round(
                        float(doc.get("vector_score") or 0.0), 6
                    )
                    record["vector_scores"] = doc.get("vector_scores", {})

            classified_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if label == "uncertain":
                uncertain_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats[label] += 1

    logging.info("Classification complete.")
    for label in ALL_LABELS:
        logging.info("%s: %d", label, stats[label])

    total_processed = sum(stats.values())
    print("Processed files:", total_processed)
    for label in ALL_LABELS:
        print(f"  {label}: {stats[label]}")


if __name__ == "__main__":
    main()
