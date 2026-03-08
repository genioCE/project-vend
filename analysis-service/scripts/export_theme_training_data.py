"""Export theme training data from Claude-labeled SQLite + corpus .md files.

Extracts themes from Claude's entry summary analysis, clusters them into a
canonical vocabulary via embedding-based agglomerative clustering, and produces
a multi-label training dataset for fine-tuning a theme classifier.

Usage:
  python -m scripts.export_theme_training_data \
    --db analysis.sqlite.bak-claude \
    --corpus /path/to/corpus \
    --output theme_training_data.jsonl \
    --min-cluster-freq 5 \
    --cluster-distance 0.35 \
    [--reuse-state-splits splits.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Add parent directory so we can import corpus_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.corpus_utils.text_processing import strip_markdown

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("export-theme-training-data")

CLAUDE_PROMPT_VERSION_PREFIX = "entry-summary-prompt-claude"


def _find_corpus_file(entry_id: str, corpus_dir: Path) -> Path | None:
    """Find the .md file matching an entry_id (filename stem)."""
    candidate = corpus_dir / f"{entry_id}.md"
    if candidate.exists():
        return candidate

    matches = list(corpus_dir.rglob(f"{entry_id}.md"))
    if matches:
        return matches[0]

    return None


def _extract_themes(payload_json: str) -> list[str] | None:
    """Extract themes list from an entry summary payload."""
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    themes = data.get("themes")
    if not isinstance(themes, list) or not themes:
        return None

    # Filter to non-empty strings
    result = [str(t).strip() for t in themes if isinstance(t, str) and t.strip()]
    return result if result else None


def _cluster_themes(
    raw_themes: list[str],
    distance_threshold: float,
    min_cluster_freq: int,
    entry_theme_lists: list[list[str]],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Cluster raw theme strings into canonical groups using embedding similarity.

    Returns:
        clusters: {canonical_label: [raw_variant_1, raw_variant_2, ...]}
        label_index: {canonical_label: integer_index}
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering

    logger.info(f"Embedding {len(raw_themes)} unique theme strings...")
    encoder = SentenceTransformer("all-mpnet-base-v2")
    embeddings = encoder.encode(raw_themes, show_progress_bar=True, normalize_embeddings=True)

    logger.info(f"Clustering with distance_threshold={distance_threshold}...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    n_clusters_raw = len(set(labels))
    logger.info(f"Agglomerative clustering produced {n_clusters_raw} raw clusters")

    # Count how many entries each raw theme appears in
    theme_entry_count: Counter[str] = Counter()
    for theme_list in entry_theme_lists:
        for t in set(theme_list):  # dedupe within entry
            theme_entry_count[t] += 1

    # Group raw themes by cluster label
    cluster_groups: dict[int, list[str]] = {}
    for theme, cluster_id in zip(raw_themes, labels):
        cluster_groups.setdefault(cluster_id, []).append(theme)

    # For each cluster, count total entry occurrences and pick canonical label
    clusters: dict[str, list[str]] = {}
    for cluster_id, members in sorted(cluster_groups.items()):
        # Total entries this cluster covers (union of member entries)
        cluster_entry_set: set[int] = set()
        for entry_idx, theme_list in enumerate(entry_theme_lists):
            if any(t in members for t in theme_list):
                cluster_entry_set.add(entry_idx)

        if len(cluster_entry_set) < min_cluster_freq:
            continue  # Drop rare clusters

        # Pick the most frequent member as canonical label
        canonical = max(members, key=lambda t: theme_entry_count[t])
        clusters[canonical] = sorted(members)

    # Sort clusters by frequency (descending) for stable label indices
    cluster_freq: dict[str, int] = {}
    for canonical, members in clusters.items():
        freq = 0
        for entry_idx, theme_list in enumerate(entry_theme_lists):
            if any(t in members for t in theme_list):
                freq += 1
        cluster_freq[canonical] = freq

    sorted_labels = sorted(clusters.keys(), key=lambda c: -cluster_freq[c])
    label_index = {label: idx for idx, label in enumerate(sorted_labels)}

    logger.info(f"After frequency filter (min={min_cluster_freq}): {len(clusters)} theme clusters")

    # Print top 20 and bottom 10 clusters for review
    logger.info("\n=== Top 20 clusters by frequency ===")
    for label in sorted_labels[:20]:
        members = clusters[label]
        logger.info(f"  [{cluster_freq[label]:4d} entries] {label}")
        if len(members) > 1:
            others = [m for m in members if m != label][:5]
            logger.info(f"    variants: {others}")

    logger.info("\n=== Bottom 10 clusters (least frequent) ===")
    for label in sorted_labels[-10:]:
        members = clusters[label]
        logger.info(f"  [{cluster_freq[label]:4d} entries] {label}")

    # Print size distribution
    sizes = [len(members) for members in clusters.values()]
    freqs = list(cluster_freq.values())
    logger.info(f"\nCluster size stats: min={min(sizes)}, max={max(sizes)}, "
                f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")
    logger.info(f"Entry frequency stats: min={min(freqs)}, max={max(freqs)}, "
                f"mean={np.mean(freqs):.1f}, median={np.median(freqs):.1f}")

    return clusters, label_index


def export_theme_training_data(
    db_path: str,
    corpus_path: str,
    output_path: str,
    min_cluster_freq: int = 5,
    cluster_distance: float = 0.35,
    reuse_state_splits: str | None = None,
    n_folds: int = 5,
    test_fraction: float = 0.1,
) -> None:
    corpus_dir = Path(corpus_path)
    if not corpus_dir.exists():
        logger.error(f"Corpus path does not exist: {corpus_path}")
        sys.exit(1)

    # Step 1A: Extract raw themes from SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT entry_id, payload_json, prompt_version FROM entry_summaries "
        "WHERE prompt_version LIKE ?",
        (f"{CLAUDE_PROMPT_VERSION_PREFIX}%",),
    )

    entry_themes: list[tuple[str, list[str]]] = []  # (entry_id, themes)
    all_raw_themes: list[str] = []
    skipped_no_themes = 0
    skipped_no_file = 0
    skipped_empty = 0

    # First pass: collect themes and validate corpus files
    raw_entries: list[tuple[str, list[str]]] = []
    for entry_id, payload_json, prompt_version in cursor:
        themes = _extract_themes(payload_json)
        if themes is None:
            skipped_no_themes += 1
            continue
        raw_entries.append((entry_id, themes))

    conn.close()
    logger.info(f"Found {len(raw_entries)} Claude-analyzed entries with themes "
                f"(skipped {skipped_no_themes} without themes)")

    # Second pass: match to corpus files and read text
    records: list[dict] = []
    entry_theme_lists: list[list[str]] = []  # parallel to records, for clustering

    for entry_id, themes in raw_entries:
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

        # Normalize themes: lowercase first letter, strip whitespace
        cleaned_themes = [t.strip() for t in themes if t.strip()]

        records.append({
            "entry_id": entry_id,
            "text": text,
            "word_count": len(text.split()),
            "raw_themes": cleaned_themes,
        })
        entry_theme_lists.append(cleaned_themes)

    logger.info(
        f"Matched {len(records)} entries to corpus files "
        f"(skipped: {skipped_no_file} no file, {skipped_empty} empty text)"
    )

    if not records:
        logger.error("No records to export!")
        sys.exit(1)

    # Collect all unique raw theme strings
    unique_themes: set[str] = set()
    for themes in entry_theme_lists:
        unique_themes.update(themes)
    unique_theme_list = sorted(unique_themes)
    logger.info(f"Total unique raw theme strings: {len(unique_theme_list)}")

    # Step 1B: Cluster themes
    clusters, label_index = _cluster_themes(
        raw_themes=unique_theme_list,
        distance_threshold=cluster_distance,
        min_cluster_freq=min_cluster_freq,
        entry_theme_lists=entry_theme_lists,
    )

    # Build reverse mapping: raw_theme -> canonical_label
    raw_to_canonical: dict[str, str] = {}
    for canonical, members in clusters.items():
        for member in members:
            raw_to_canonical[member] = canonical

    n_labels = len(label_index)
    logger.info(f"Label count: {n_labels}")

    # Step 1C: Create binary label vectors
    for rec in records:
        binary_labels: dict[str, int] = {label: 0 for label in label_index}
        for raw_theme in rec["raw_themes"]:
            canonical = raw_to_canonical.get(raw_theme)
            if canonical and canonical in binary_labels:
                binary_labels[canonical] = 1
        rec["labels"] = binary_labels
        del rec["raw_themes"]  # Don't include in output

    # Sort for reproducibility
    records.sort(key=lambda r: r["entry_id"])

    # Step 1D: Create train/test splits
    if reuse_state_splits:
        # Reuse the same test IDs from state classifier splits
        with open(reuse_state_splits) as f:
            state_splits = json.load(f)
        state_test_ids = set(state_splits["test_ids"])
        all_entry_ids = {r["entry_id"] for r in records}

        test_ids = sorted(all_entry_ids & state_test_ids)
        train_ids = sorted(all_entry_ids - state_test_ids)
        logger.info(f"Reusing state classifier splits: {len(test_ids)} test IDs matched "
                    f"(of {len(state_test_ids)} in state splits)")
    else:
        # Generate new splits using same deterministic hash
        import hashlib

        def _stable_hash(entry_id: str) -> int:
            return int(hashlib.md5(entry_id.encode()).hexdigest()[:8], 16)

        n_test = max(1, int(len(records) * test_fraction))
        sorted_by_hash = sorted(records, key=lambda r: _stable_hash(r["entry_id"]))
        test_set = {r["entry_id"] for r in sorted_by_hash[:n_test]}

        test_ids = sorted(r["entry_id"] for r in records if r["entry_id"] in test_set)
        train_ids = sorted(r["entry_id"] for r in records if r["entry_id"] not in test_set)

    # Create k-fold CV indices for training set
    import hashlib

    def _stable_hash(entry_id: str) -> int:
        return int(hashlib.md5(entry_id.encode()).hexdigest()[:8], 16)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for entry_id in train_ids:
        fold_idx = _stable_hash(entry_id) % n_folds
        folds[fold_idx].append(entry_id)

    # Write output files
    output_dir = Path(output_path).parent
    output_stem = Path(output_path).stem

    # Training data JSONL
    output_file = Path(output_path)
    with output_file.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Wrote {len(records)} records to {output_path}")

    # Clusters mapping
    clusters_path = output_dir / "theme_clusters.json"
    with clusters_path.open("w") as f:
        json.dump(clusters, f, indent=2)
    logger.info(f"Wrote {len(clusters)} clusters to {clusters_path}")

    # Label index
    label_index_path = output_dir / "theme_label_index.json"
    with label_index_path.open("w") as f:
        json.dump(label_index, f, indent=2)
    logger.info(f"Wrote {n_labels} labels to {label_index_path}")

    # Splits
    splits_path = output_dir / "theme_splits.json"
    splits = {
        "total_records": len(records),
        "test_ids": test_ids,
        "test_count": len(test_ids),
        "train_ids": train_ids,
        "train_count": len(train_ids),
        "n_folds": n_folds,
        "folds": {str(i): fold_ids for i, fold_ids in enumerate(folds)},
        "n_labels": n_labels,
        "cluster_distance": cluster_distance,
        "min_cluster_freq": min_cluster_freq,
    }
    with splits_path.open("w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        f"Splits: {len(train_ids)} train, {len(test_ids)} test, "
        f"{n_folds} folds (sizes: {[len(f) for f in folds]})"
    )

    # Print label statistics
    label_counts = Counter()
    for rec in records:
        for label, val in rec["labels"].items():
            if val == 1:
                label_counts[label] += 1

    avg_labels_per_entry = np.mean([sum(r["labels"].values()) for r in records])
    logger.info(f"\nAverage labels per entry: {avg_labels_per_entry:.1f}")
    logger.info(f"Label frequency range: {min(label_counts.values())}-{max(label_counts.values())} entries")

    # Print positive rate for each label
    logger.info("\n=== Label frequencies (top 30) ===")
    for label, count in label_counts.most_common(30):
        pct = 100 * count / len(records)
        logger.info(f"  {label:40s}  {count:4d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Export theme training data from Claude-labeled SQLite")
    parser.add_argument("--db", required=True, help="Path to Claude-labeled analysis.sqlite")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory with .md files")
    parser.add_argument("--output", default="theme_training_data.jsonl", help="Output JSONL file")
    parser.add_argument("--min-cluster-freq", type=int, default=5,
                        help="Minimum entry count for a theme cluster to be included")
    parser.add_argument("--cluster-distance", type=float, default=0.35,
                        help="Cosine distance threshold for agglomerative clustering")
    parser.add_argument("--reuse-state-splits", default=None,
                        help="Path to state classifier splits.json to reuse test IDs")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--test-fraction", type=float, default=0.1,
                        help="Fraction for test set (only used if not reusing state splits)")
    args = parser.parse_args()

    export_theme_training_data(
        db_path=args.db,
        corpus_path=args.corpus,
        output_path=args.output,
        min_cluster_freq=args.min_cluster_freq,
        cluster_distance=args.cluster_distance,
        reuse_state_splits=args.reuse_state_splits,
        n_folds=args.n_folds,
        test_fraction=args.test_fraction,
    )


if __name__ == "__main__":
    main()
