"""Convert v6 Haiku decision labels to sentence-level training data.

Takes the entry-level decision_labels_v6.jsonl (Haiku output) and converts
to the same sentence-level format used by train_decision_classifier.py.

For each entry:
1. Read the corpus file, strip markdown, split into sentences
2. Match Haiku decision strings to sentences via token overlap
3. Write sentence-level JSONL with is_decision labels

Usage:
  python scripts/convert_v6_to_training_data.py \
    --labels decision_labels_v6.jsonl \
    --corpus ../../data \
    --output decision_training_data_v6.jsonl \
    [--reuse-state-splits state_splits.json]
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.corpus_utils.text_processing import strip_markdown

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("convert-v6")

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
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w not in _OVERLAP_STOP_WORDS and len(w) > 1}


def _jaccard_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _find_corpus_file(entry_id: str, corpus_dir: Path) -> Path | None:
    candidate = corpus_dir / f"{entry_id}.md"
    if candidate.exists():
        return candidate
    matches = list(corpus_dir.rglob(f"{entry_id}.md"))
    return matches[0] if matches else None


def _stable_hash(entry_id: str) -> int:
    return int(hashlib.md5(entry_id.encode()).hexdigest()[:8], 16)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert v6 labels to sentence-level training data")
    parser.add_argument("--labels", required=True, help="Path to decision_labels_v6.jsonl")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory")
    parser.add_argument("--output", default="decision_training_data_v6.jsonl", help="Output JSONL")
    parser.add_argument("--reuse-state-splits", default=None, help="Reuse test split from state classifier")
    parser.add_argument("--match-threshold", type=float, default=0.35, help="Jaccard overlap threshold")
    parser.add_argument("--min-sentence-words", type=int, default=4, help="Min words per sentence")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Test fraction")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    labels = [json.loads(l) for l in open(args.labels)]
    logger.info(f"Loaded {len(labels)} entry labels")

    records = []
    entry_ids_with_data = set()
    total_decisions = 0
    matched_decisions = 0
    skipped_no_file = 0
    skipped_empty = 0
    unmatched_examples = []

    for entry in labels:
        entry_id = entry["entry_id"]
        decisions = entry["decisions"]

        file_path = _find_corpus_file(entry_id, corpus_dir)
        if not file_path:
            skipped_no_file += 1
            continue

        try:
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        text = strip_markdown(raw_text)
        normalized = " ".join(text.split())
        if not normalized.strip():
            skipped_empty += 1
            continue

        entry_ids_with_data.add(entry_id)
        sentences = _split_sentences(normalized)

        decision_tokens = [_tokenize(d) for d in decisions]
        total_decisions += len(decisions)

        decision_matched = [False] * len(decisions)
        sentence_labels = []

        for sent_idx, sentence in enumerate(sentences):
            if len(sentence.split()) < args.min_sentence_words:
                continue

            sent_tokens = _tokenize(sentence)
            best_overlap = 0.0
            best_idx = -1

            for d_idx, d_tokens in enumerate(decision_tokens):
                overlap = _jaccard_overlap(sent_tokens, d_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = d_idx

            if best_overlap >= args.match_threshold:
                is_decision = 1
                matched_decision_text = decisions[best_idx]
                decision_matched[best_idx] = True
            else:
                is_decision = 0
                matched_decision_text = None

            sentence_labels.append((sentence, is_decision, matched_decision_text))

        for d_idx, matched in enumerate(decision_matched):
            if not matched:
                if len(unmatched_examples) < 20:
                    unmatched_examples.append((entry_id, decisions[d_idx]))
            else:
                matched_decisions += 1

        # Build context-aware records
        # Filter to keep only sentences that pass min-word threshold
        valid_sentences = [
            (sent, label, matched)
            for sent, label, matched in sentence_labels
        ]

        for sent_idx, (sentence, is_decision, matched_dec) in enumerate(valid_sentences):
            # Build context window: [prev] [SEP] target [SEP] [next]
            prev_sent = valid_sentences[sent_idx - 1][0] if sent_idx > 0 else ""
            next_sent = valid_sentences[sent_idx + 1][0] if sent_idx < len(valid_sentences) - 1 else ""

            parts = []
            if prev_sent:
                parts.append(prev_sent)
            parts.append(sentence)
            if next_sent:
                parts.append(next_sent)
            text_with_context = " [SEP] ".join(parts)

            rec = {
                "entry_id": entry_id,
                "sentence": sentence,
                "text_with_context": text_with_context,
                "sentence_index": sent_idx,
                "is_decision": is_decision,
            }
            if matched_dec is not None:
                rec["matched_decision"] = matched_dec
            records.append(rec)

    total_sentences = len(records)
    positive_sentences = sum(1 for r in records if r["is_decision"] == 1)
    pos_rate = 100 * positive_sentences / total_sentences if total_sentences else 0
    dec_match_rate = 100 * matched_decisions / total_decisions if total_decisions else 0

    logger.info(f"Entries: {len(entry_ids_with_data)} (skipped: {skipped_no_file} no file, {skipped_empty} empty)")
    logger.info(f"Decision match rate: {matched_decisions}/{total_decisions} ({dec_match_rate:.1f}%)")
    logger.info(f"Total sentences: {total_sentences}, positive: {positive_sentences} ({pos_rate:.1f}%)")

    if unmatched_examples:
        logger.info("\n=== Sample unmatched decisions ===")
        for eid, dec in unmatched_examples[:15]:
            logger.info(f"  {eid}: {dec!r}")

    records.sort(key=lambda r: (r["entry_id"], r["sentence_index"]))

    # Splits
    all_entry_ids = sorted(entry_ids_with_data)

    if args.reuse_state_splits:
        state_splits = json.load(open(args.reuse_state_splits))
        state_test_ids = set(state_splits["test_ids"])
        test_ids = sorted(set(all_entry_ids) & state_test_ids)
        train_ids = sorted(set(all_entry_ids) - state_test_ids)
        logger.info(f"Reusing state splits: {len(test_ids)} test entries")
    else:
        n_test = max(1, int(len(all_entry_ids) * args.test_fraction))
        sorted_by_hash = sorted(all_entry_ids, key=_stable_hash)
        test_ids = sorted(sorted_by_hash[:n_test])
        train_ids = sorted(set(all_entry_ids) - set(test_ids))

    test_set = set(test_ids)

    folds = [[] for _ in range(args.n_folds)]
    for eid in train_ids:
        folds[_stable_hash(eid) % args.n_folds].append(eid)

    train_examples = [r for r in records if r["entry_id"] not in test_set]
    test_examples = [r for r in records if r["entry_id"] in test_set]
    train_pos = sum(1 for r in train_examples if r["is_decision"] == 1)
    test_pos = sum(1 for r in test_examples if r["is_decision"] == 1)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Wrote {len(records)} records to {output_path}")

    splits_path = output_path.parent / "decision_splits_v6.json"
    splits = {
        "total_records": len(records),
        "total_entries": len(all_entry_ids),
        "positive_rate": round(pos_rate, 2),
        "match_threshold": args.match_threshold,
        "min_sentence_words": args.min_sentence_words,
        "test_ids": test_ids,
        "test_count": len(test_ids),
        "test_examples": len(test_examples),
        "test_positive": test_pos,
        "train_ids": train_ids,
        "train_count": len(train_ids),
        "train_examples": len(train_examples),
        "train_positive": train_pos,
        "n_folds": args.n_folds,
        "folds": {str(i): fold_ids for i, fold_ids in enumerate(folds)},
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        f"Splits: {len(train_ids)} train ({len(train_examples)} sents, {train_pos} pos), "
        f"{len(test_ids)} test ({len(test_examples)} sents, {test_pos} pos)"
    )


if __name__ == "__main__":
    main()
