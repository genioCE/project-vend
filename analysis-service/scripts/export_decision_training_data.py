"""Export decision detection training data from Claude-labeled SQLite.

Splits each Claude-labeled entry into sentences, then matches Claude's
decision/action strings to individual sentences via fuzzy token overlap.
Produces a binary classification dataset (is_decision / not_decision) for
fine-tuning a sentence-level decision detector.

Usage:
  python -m scripts.export_decision_training_data \
    --db analysis.sqlite.bak-claude \
    --corpus /path/to/corpus \
    --output decision_training_data.jsonl \
    [--reuse-state-splits splits.json] \
    [--match-threshold 0.4]
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
logger = logging.getLogger("export-decision-training-data")

CLAUDE_PROMPT_VERSION_PREFIX = "entry-summary-prompt-claude"

# Words to ignore when computing token overlap
_OVERLAP_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "the", "a", "an", "is", "am",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "can",
    "may", "might", "shall", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "during", "before",
    "after", "and", "but", "or", "if", "not", "so", "that", "this",
    "it", "its", "he", "she", "they", "them", "their", "his", "her",
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (same pattern used in entry_summary_provider)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _tokenize(text: str) -> set[str]:
    """Extract lowercased content words from text."""
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w not in _OVERLAP_STOP_WORDS and len(w) > 1}


def _jaccard_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _find_corpus_file(entry_id: str, corpus_dir: Path) -> Path | None:
    """Find the .md file matching an entry_id (filename stem)."""
    candidate = corpus_dir / f"{entry_id}.md"
    if candidate.exists():
        return candidate

    matches = list(corpus_dir.rglob(f"{entry_id}.md"))
    if matches:
        return matches[0]

    return None


def _extract_decisions(payload_json: str) -> list[str] | None:
    """Extract decisions_actions list from an entry summary payload."""
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    decisions = data.get("decisions_actions")
    if not isinstance(decisions, list) or not decisions:
        return None

    result = [str(d).strip() for d in decisions if isinstance(d, str) and d.strip()]
    return result if result else None


def _stable_hash(entry_id: str) -> int:
    return int(hashlib.md5(entry_id.encode()).hexdigest()[:8], 16)


def export_decision_training_data(
    db_path: str,
    corpus_path: str,
    output_path: str,
    match_threshold: float = 0.4,
    reuse_state_splits: str | None = None,
    n_folds: int = 5,
    test_fraction: float = 0.1,
    min_sentence_words: int = 4,
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

    raw_entries: list[tuple[str, list[str]]] = []
    skipped_no_decisions = 0

    for entry_id, payload_json in cursor:
        decisions = _extract_decisions(payload_json)
        if decisions is None:
            skipped_no_decisions += 1
            continue
        raw_entries.append((entry_id, decisions))

    conn.close()
    logger.info(
        f"Found {len(raw_entries)} Claude-analyzed entries with decisions "
        f"(skipped {skipped_no_decisions} without decisions)"
    )

    # Step 2: Build sentence-level training examples
    records: list[dict] = []
    entry_ids_with_data: set[str] = set()
    skipped_no_file = 0
    skipped_empty = 0
    total_decisions = 0
    matched_decisions = 0
    total_sentences = 0
    positive_sentences = 0
    unmatched_examples: list[tuple[str, str]] = []

    for entry_id, decisions in raw_entries:
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
        normalized = " ".join(text.split())
        if not normalized.strip():
            skipped_empty += 1
            continue

        entry_ids_with_data.add(entry_id)
        sentences = _split_sentences(normalized)

        # Pre-tokenize all Claude decision strings
        decision_tokens = [_tokenize(d) for d in decisions]
        total_decisions += len(decisions)

        # For each sentence, check if it matches any Claude decision
        decision_matched = [False] * len(decisions)
        sentence_labels: list[tuple[str, int, str | None]] = []

        for sent_idx, sentence in enumerate(sentences):
            if len(sentence.split()) < min_sentence_words:
                continue

            sent_tokens = _tokenize(sentence)
            best_overlap = 0.0
            best_decision_idx = -1

            for d_idx, d_tokens in enumerate(decision_tokens):
                overlap = _jaccard_overlap(sent_tokens, d_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_decision_idx = d_idx

            if best_overlap >= match_threshold:
                is_decision = 1
                matched_decision_text = decisions[best_decision_idx]
                decision_matched[best_decision_idx] = True
                positive_sentences += 1
            else:
                is_decision = 0
                matched_decision_text = None

            total_sentences += 1
            sentence_labels.append((sentence, is_decision, matched_decision_text))

        # Track unmatched decisions
        for d_idx, matched in enumerate(decision_matched):
            if not matched:
                if len(unmatched_examples) < 20:
                    unmatched_examples.append((entry_id, decisions[d_idx]))
            else:
                matched_decisions += 1

        # Add all labeled sentences to records
        for sent_idx, (sentence, is_decision, matched_dec) in enumerate(sentence_labels):
            rec = {
                "entry_id": entry_id,
                "sentence": sentence,
                "sentence_index": sent_idx,
                "is_decision": is_decision,
            }
            if matched_dec is not None:
                rec["matched_decision"] = matched_dec
            records.append(rec)

    logger.info(
        f"Matched {len(entry_ids_with_data)} entries to corpus files "
        f"(skipped: {skipped_no_file} no file, {skipped_empty} empty text)"
    )
    dec_match_rate = 100 * matched_decisions / total_decisions if total_decisions else 0
    pos_rate = 100 * positive_sentences / total_sentences if total_sentences else 0
    logger.info(
        f"Decision match rate: {matched_decisions}/{total_decisions} "
        f"({dec_match_rate:.1f}%)"
    )
    logger.info(
        f"Total sentences: {total_sentences}, "
        f"positive (decision): {positive_sentences} ({pos_rate:.1f}%)"
    )

    if unmatched_examples:
        logger.info("\n=== Sample unmatched decisions ===")
        for entry_id, dec in unmatched_examples[:10]:
            logger.info(f"  {entry_id}: {dec!r}")

    if not records:
        logger.error("No records to export!")
        sys.exit(1)

    # Sort for reproducibility
    records.sort(key=lambda r: (r["entry_id"], r["sentence_index"]))

    # Step 3: Create train/test splits (entry-level)
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
    train_pos = sum(1 for r in train_examples if r["is_decision"] == 1)
    test_pos = sum(1 for r in test_examples if r["is_decision"] == 1)

    # Step 4: Write output files
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training data JSONL
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Wrote {len(records)} records to {output_path}")

    # Splits
    splits_path = output_dir / "decision_splits.json"
    splits = {
        "total_records": len(records),
        "total_entries": len(all_entry_ids),
        "positive_rate": round(pos_rate, 2),
        "match_threshold": match_threshold,
        "min_sentence_words": min_sentence_words,
        "test_ids": test_ids,
        "test_count": len(test_ids),
        "test_examples": len(test_examples),
        "test_positive": test_pos,
        "train_ids": train_ids,
        "train_count": len(train_ids),
        "train_examples": len(train_examples),
        "train_positive": train_pos,
        "n_folds": n_folds,
        "folds": {str(i): fold_ids for i, fold_ids in enumerate(folds)},
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        f"Splits: {len(train_ids)} train entries "
        f"({len(train_examples)} sentences, {train_pos} positive), "
        f"{len(test_ids)} test entries "
        f"({len(test_examples)} sentences, {test_pos} positive), "
        f"{n_folds} folds"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export decision detection training data"
    )
    parser.add_argument("--db", required=True, help="Path to Claude-labeled analysis.sqlite")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory with .md files")
    parser.add_argument("--output", default="decision_training_data.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--match-threshold", type=float, default=0.4,
        help="Jaccard token overlap threshold for matching decisions to sentences",
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
    parser.add_argument(
        "--min-sentence-words", type=int, default=4,
        help="Minimum words for a sentence to be included",
    )
    args = parser.parse_args()

    export_decision_training_data(
        db_path=args.db,
        corpus_path=args.corpus,
        output_path=args.output,
        match_threshold=args.match_threshold,
        reuse_state_splits=args.reuse_state_splits,
        n_folds=args.n_folds,
        test_fraction=args.test_fraction,
        min_sentence_words=args.min_sentence_words,
    )


if __name__ == "__main__":
    main()
