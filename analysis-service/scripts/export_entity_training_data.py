"""Export entity type classification training data from Claude-labeled SQLite.

For each Claude-labeled entry, extracts typed entities and matches them back
to the source text to build context windows. Produces a dataset of
(entity_name, entity_type, context) examples for fine-tuning an entity type
classifier.

Usage:
  python -m scripts.export_entity_training_data \
    --db analysis.sqlite.bak-claude \
    --corpus /path/to/corpus \
    --output entity_training_data.jsonl \
    [--reuse-state-splits splits.json] \
    [--context-window 200]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# Add parent directory so we can import corpus_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.corpus_utils.text_processing import strip_markdown

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("export-entity-training-data")

CLAUDE_PROMPT_VERSION_PREFIX = "entry-summary-prompt-claude"
VALID_ENTITY_TYPES = {"person", "place", "organization", "concept", "spiritual"}


def _find_corpus_file(entry_id: str, corpus_dir: Path) -> Path | None:
    """Find the .md file matching an entry_id (filename stem)."""
    candidate = corpus_dir / f"{entry_id}.md"
    if candidate.exists():
        return candidate

    matches = list(corpus_dir.rglob(f"{entry_id}.md"))
    if matches:
        return matches[0]

    return None


def _extract_entities(payload_json: str) -> list[dict] | None:
    """Extract typed entities from an entry summary payload."""
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    entities = data.get("entities")
    if not isinstance(entities, list) or not entities:
        return None

    result = []
    for e in entities:
        if isinstance(e, dict):
            name = str(e.get("name", "")).strip()
            etype = str(e.get("type", "concept")).strip().lower()
            if name and etype in VALID_ENTITY_TYPES:
                result.append({"name": name, "type": etype})
        elif isinstance(e, str) and e.strip():
            # v4 compat: plain string entities default to "concept"
            result.append({"name": e.strip(), "type": "concept"})

    return result if result else None


def _find_context(text: str, entity_name: str, window: int = 200) -> str | None:
    """Find entity name in text and return a context window around it.

    Uses case-insensitive search. Returns None if not found.
    """
    text_lower = text.lower()
    name_lower = entity_name.lower()

    idx = text_lower.find(name_lower)
    if idx == -1:
        # Try word-boundary-relaxed search (e.g. "Mom" in "Mom's")
        pattern = re.escape(name_lower)
        match = re.search(pattern, text_lower)
        if match:
            idx = match.start()
        else:
            return None

    # Extract context window centered on the entity
    half = window // 2
    start = max(0, idx - half)
    end = min(len(text), idx + len(entity_name) + half)
    return text[start:end].strip()


def _stable_hash(entry_id: str) -> int:
    return int(hashlib.md5(entry_id.encode()).hexdigest()[:8], 16)


def export_entity_training_data(
    db_path: str,
    corpus_path: str,
    output_path: str,
    context_window: int = 200,
    reuse_state_splits: str | None = None,
    n_folds: int = 5,
    test_fraction: float = 0.1,
) -> None:
    corpus_dir = Path(corpus_path)
    if not corpus_dir.exists():
        logger.error(f"Corpus path does not exist: {corpus_path}")
        sys.exit(1)

    # Step 1: Load Claude-labeled entries from SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT entry_id, payload_json FROM entry_summaries "
        "WHERE prompt_version LIKE ?",
        (f"{CLAUDE_PROMPT_VERSION_PREFIX}%",),
    )

    raw_entries: list[tuple[str, list[dict]]] = []
    skipped_no_entities = 0

    for entry_id, payload_json in cursor:
        entities = _extract_entities(payload_json)
        if entities is None:
            skipped_no_entities += 1
            continue
        raw_entries.append((entry_id, entities))

    conn.close()
    logger.info(
        f"Found {len(raw_entries)} Claude-analyzed entries with entities "
        f"(skipped {skipped_no_entities} without entities)"
    )

    # Step 2: Match to corpus files and build training examples
    records: list[dict] = []
    entry_ids_with_data: set[str] = set()
    skipped_no_file = 0
    skipped_empty = 0
    total_entities = 0
    matched_entities = 0
    unmatched_entities = 0
    type_counts: Counter[str] = Counter()
    unmatched_examples: list[tuple[str, str]] = []

    for entry_id, entities in raw_entries:
        file_path = _find_corpus_file(entry_id, corpus_dir)
        if file_path is None:
            skipped_no_file += 1
            continue

        try:
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            continue

        text = strip_markdown(raw_text)
        if not text.strip():
            skipped_empty += 1
            continue

        entry_ids_with_data.add(entry_id)

        for entity in entities:
            total_entities += 1
            context = _find_context(text, entity["name"], window=context_window)

            if context is None:
                unmatched_entities += 1
                if len(unmatched_examples) < 20:
                    unmatched_examples.append((entry_id, entity["name"]))
                continue

            matched_entities += 1
            type_counts[entity["type"]] += 1
            records.append({
                "entry_id": entry_id,
                "entity_name": entity["name"],
                "entity_type": entity["type"],
                "context": context,
            })

    logger.info(
        f"Matched {len(entry_ids_with_data)} entries to corpus files "
        f"(skipped: {skipped_no_file} no file, {skipped_empty} empty text)"
    )
    match_rate = 100 * matched_entities / total_entities if total_entities else 0
    logger.info(
        f"Entity match rate: {matched_entities}/{total_entities} "
        f"({match_rate:.1f}%) — {unmatched_entities} unmatched"
    )

    if unmatched_examples:
        logger.info("\n=== Sample unmatched entities ===")
        for entry_id, name in unmatched_examples[:10]:
            logger.info(f"  {entry_id}: {name!r}")

    if not records:
        logger.error("No records to export!")
        sys.exit(1)

    # Sort for reproducibility
    records.sort(key=lambda r: (r["entry_id"], r["entity_name"]))

    # Step 3: Create train/test splits (entry-level, not example-level)
    all_entry_ids = sorted(entry_ids_with_data)

    if reuse_state_splits:
        with open(reuse_state_splits) as f:
            state_splits = json.load(f)
        state_test_ids = set(state_splits["test_ids"])
        test_ids = sorted(set(all_entry_ids) & state_test_ids)
        train_ids = sorted(set(all_entry_ids) - state_test_ids)
        logger.info(
            f"Reusing state classifier splits: {len(test_ids)} test entries matched "
            f"(of {len(state_test_ids)} in state splits)"
        )
    else:
        n_test = max(1, int(len(all_entry_ids) * test_fraction))
        sorted_by_hash = sorted(all_entry_ids, key=_stable_hash)
        test_set = set(sorted_by_hash[:n_test])
        test_ids = sorted(test_set)
        train_ids = sorted(set(all_entry_ids) - test_set)

    test_set = set(test_ids)

    # Create k-fold CV indices for training set
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for entry_id in train_ids:
        fold_idx = _stable_hash(entry_id) % n_folds
        folds[fold_idx].append(entry_id)

    # Count examples per split
    train_examples = [r for r in records if r["entry_id"] not in test_set]
    test_examples = [r for r in records if r["entry_id"] in test_set]

    # Step 4: Write output files
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training data JSONL
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Wrote {len(records)} records to {output_path}")

    # Entity type index
    type_index = {t: i for i, t in enumerate(sorted(VALID_ENTITY_TYPES))}
    type_index_path = output_dir / "entity_type_index.json"
    with open(type_index_path, "w") as f:
        json.dump(type_index, f, indent=2)
    logger.info(f"Wrote entity type index to {type_index_path}")

    # Splits
    splits_path = output_dir / "entity_splits.json"
    splits = {
        "total_records": len(records),
        "total_entries": len(all_entry_ids),
        "test_ids": test_ids,
        "test_count": len(test_ids),
        "test_examples": len(test_examples),
        "train_ids": train_ids,
        "train_count": len(train_ids),
        "train_examples": len(train_examples),
        "n_folds": n_folds,
        "folds": {str(i): fold_ids for i, fold_ids in enumerate(folds)},
        "context_window": context_window,
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        f"Splits: {len(train_ids)} train entries ({len(train_examples)} examples), "
        f"{len(test_ids)} test entries ({len(test_examples)} examples), "
        f"{n_folds} folds (sizes: {[len(f) for f in folds]})"
    )

    # Step 5: Print class distribution
    logger.info("\n=== Entity type distribution ===")
    for etype in sorted(type_counts, key=lambda t: -type_counts[t]):
        count = type_counts[etype]
        pct = 100 * count / len(records)
        logger.info(f"  {etype:15s}  {count:5d} ({pct:5.1f}%)")

    # Print train/test class distributions
    train_type_counts: Counter[str] = Counter()
    test_type_counts: Counter[str] = Counter()
    for r in train_examples:
        train_type_counts[r["entity_type"]] += 1
    for r in test_examples:
        test_type_counts[r["entity_type"]] += 1

    logger.info("\n=== Train set class distribution ===")
    for etype in sorted(VALID_ENTITY_TYPES):
        count = train_type_counts.get(etype, 0)
        logger.info(f"  {etype:15s}  {count:5d}")

    logger.info("\n=== Test set class distribution ===")
    for etype in sorted(VALID_ENTITY_TYPES):
        count = test_type_counts.get(etype, 0)
        logger.info(f"  {etype:15s}  {count:5d}")


def main():
    parser = argparse.ArgumentParser(
        description="Export entity type classification training data"
    )
    parser.add_argument("--db", required=True, help="Path to Claude-labeled analysis.sqlite")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory with .md files")
    parser.add_argument("--output", default="entity_training_data.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--context-window", type=int, default=200,
        help="Character window around entity mention for context",
    )
    parser.add_argument(
        "--reuse-state-splits", default=None,
        help="Path to state classifier splits.json to reuse test IDs",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--test-fraction", type=float, default=0.1,
        help="Fraction for test set (only used if not reusing state splits)",
    )
    args = parser.parse_args()

    export_entity_training_data(
        db_path=args.db,
        corpus_path=args.corpus,
        output_path=args.output,
        context_window=args.context_window,
        reuse_state_splits=args.reuse_state_splits,
        n_folds=args.n_folds,
        test_fraction=args.test_fraction,
    )


if __name__ == "__main__":
    main()
