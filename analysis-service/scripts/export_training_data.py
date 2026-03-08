"""Export training data from Claude-labeled SQLite + corpus .md files.

Pairs Claude's 8-dimension state profile scores with the raw entry text
(preprocessed identically to the batch analysis pipeline) to create a
supervised learning dataset for fine-tuning.

Usage:
  python -m scripts.export_training_data \
    --db analysis.sqlite \
    --corpus /path/to/corpus \
    --output training_data.jsonl \
    --splits splits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path

# Add parent directory so we can import corpus_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.corpus_utils.text_processing import strip_markdown

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("export-training-data")

ALL_DIMENSIONS = (
    "valence",
    "activation",
    "agency",
    "certainty",
    "relational_openness",
    "self_trust",
    "time_orientation",
    "integration",
)

CLAUDE_PROMPT_VERSION = "state-label-prompt-claude-v1"


def _extract_labels(payload_json: str) -> dict[str, float] | None:
    """Extract 8-dimension scores from a state label payload."""
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    dims = data.get("state_profile", {}).get("dimensions", [])
    if len(dims) != 8:
        return None

    labels: dict[str, float] = {}
    for dim in dims:
        name = dim.get("dimension", "")
        score = dim.get("score")
        if name not in ALL_DIMENSIONS or score is None:
            return None
        labels[name] = float(score)

    # Skip flatlined entries (all zeros = failed analysis)
    if all(v == 0.0 for v in labels.values()):
        return None

    return labels


def _find_corpus_file(entry_id: str, corpus_dir: Path) -> Path | None:
    """Find the .md file matching an entry_id (filename stem)."""
    # Try direct match first
    candidate = corpus_dir / f"{entry_id}.md"
    if candidate.exists():
        return candidate

    # Search recursively (corpus may have subdirectories)
    matches = list(corpus_dir.rglob(f"{entry_id}.md"))
    if matches:
        return matches[0]

    return None


def _stable_split_key(entry_id: str) -> int:
    """Deterministic hash for reproducible train/test splits."""
    h = hashlib.md5(entry_id.encode()).hexdigest()
    return int(h[:8], 16)


def export_training_data(
    db_path: str,
    corpus_path: str,
    output_path: str,
    splits_path: str,
    test_fraction: float = 0.1,
    n_folds: int = 5,
) -> None:
    corpus_dir = Path(corpus_path)
    if not corpus_dir.exists():
        logger.error(f"Corpus path does not exist: {corpus_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT entry_id, payload_json FROM state_labels WHERE prompt_version = ?",
        (CLAUDE_PROMPT_VERSION,),
    )

    records: list[dict] = []
    skipped_no_file = 0
    skipped_empty = 0
    skipped_bad_labels = 0

    for entry_id, payload_json in cursor:
        # Extract labels
        labels = _extract_labels(payload_json)
        if labels is None:
            skipped_bad_labels += 1
            continue

        # Find corpus file
        file_path = _find_corpus_file(entry_id, corpus_dir)
        if file_path is None:
            skipped_no_file += 1
            continue

        # Read and preprocess text (same pipeline as batch_analyze.py)
        try:
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            continue

        text = strip_markdown(raw_text)
        if not text.strip():
            skipped_empty += 1
            continue

        records.append({
            "entry_id": entry_id,
            "text": text,
            "word_count": len(text.split()),
            "labels": labels,
        })

    conn.close()

    logger.info(
        f"Exported {len(records)} training records "
        f"(skipped: {skipped_no_file} no file, {skipped_empty} empty, "
        f"{skipped_bad_labels} bad labels)"
    )

    if not records:
        logger.error("No records exported!")
        sys.exit(1)

    # Sort by entry_id for reproducibility
    records.sort(key=lambda r: r["entry_id"])

    # Write training data
    output = Path(output_path)
    with output.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Wrote {len(records)} records to {output_path}")

    # Create train/test splits
    # Deterministic split based on entry_id hash
    test_ids: list[str] = []
    train_ids: list[str] = []

    n_test = max(1, int(len(records) * test_fraction))

    # Sort by hash to get deterministic ordering, take first n_test as test
    sorted_by_hash = sorted(records, key=lambda r: _stable_split_key(r["entry_id"]))
    test_set = {r["entry_id"] for r in sorted_by_hash[:n_test]}

    for rec in records:
        if rec["entry_id"] in test_set:
            test_ids.append(rec["entry_id"])
        else:
            train_ids.append(rec["entry_id"])

    # Create k-fold CV indices for training set
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, entry_id in enumerate(train_ids):
        fold_idx = _stable_split_key(entry_id) % n_folds
        folds[fold_idx].append(entry_id)

    splits = {
        "total_records": len(records),
        "test_ids": test_ids,
        "test_count": len(test_ids),
        "train_ids": train_ids,
        "train_count": len(train_ids),
        "n_folds": n_folds,
        "folds": {str(i): fold_ids for i, fold_ids in enumerate(folds)},
    }

    splits_file = Path(splits_path)
    with splits_file.open("w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        f"Splits: {len(train_ids)} train, {len(test_ids)} test, "
        f"{n_folds} folds (sizes: {[len(f) for f in folds]})"
    )

    # Print label statistics
    import statistics

    for dim in ALL_DIMENSIONS:
        scores = [r["labels"][dim] for r in records]
        logger.info(
            f"  {dim:25s}: mean={statistics.mean(scores):+.3f} "
            f"std={statistics.stdev(scores):.3f} "
            f"min={min(scores):+.2f} max={max(scores):+.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Export training data from Claude-labeled SQLite")
    parser.add_argument("--db", required=True, help="Path to Claude-labeled analysis.sqlite")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory with .md files")
    parser.add_argument("--output", default="training_data.jsonl", help="Output JSONL file")
    parser.add_argument("--splits", default="splits.json", help="Output splits JSON file")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction for test set")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    export_training_data(
        db_path=args.db,
        corpus_path=args.corpus,
        output_path=args.output,
        splits_path=args.splits,
        test_fraction=args.test_fraction,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
