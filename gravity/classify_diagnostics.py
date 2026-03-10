#!/usr/bin/env python3
"""
Fragment Classification Confusion Matrix

Analyzes how reliably Claude Haiku classifies fragments into the 6 types.
Compares Claude's decomposition output against ground-truth annotations.

Usage:
    python classify_diagnostics.py              # Run full analysis
    python classify_diagnostics.py --skip-decompose  # Reuse cached decompositions
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cache import load_cache, save_cache, get_or_decompose
from decompose import embed_decomposition
from fragments import DecompositionResult, FragmentType

from test_queries import TEST_QUERIES, ExpectedFragment

console = Console()
RESULTS_DIR = Path(__file__).parent / "results"

FRAGMENT_TYPES = [t.value for t in FragmentType]


@dataclass
class FragmentMatch:
    """Result of matching a predicted fragment to expected fragments."""
    predicted_type: str
    predicted_text: str
    matched_expected_type: str | None  # None if no match found
    matched_expected_text: str | None
    is_correct_type: bool
    similarity_score: float  # Text similarity to best match


@dataclass
class DecompositionAnalysis:
    """Full analysis of a single query's decomposition."""
    query: str
    expected_fragments: list[ExpectedFragment]
    predicted_fragments: list[tuple[str, str]]  # (type, text)
    fragment_matches: list[FragmentMatch]
    # Metrics
    fragment_count_expected: int
    fragment_count_predicted: int
    fragment_count_diff: int
    coverage: float  # Fraction of expected fragments matched
    primary_mass_correct: bool
    expected_primary_text: str
    predicted_primary_text: str


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def text_similarity(a: str, b: str) -> float:
    """Simple word overlap similarity between two texts."""
    words_a = set(normalize_text(a).split())
    words_b = set(normalize_text(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def find_best_match(
    predicted_text: str,
    expected_fragments: list[ExpectedFragment],
    already_matched: set[int],
) -> tuple[int | None, float]:
    """
    Find the best matching expected fragment for a predicted fragment.
    Returns (index, similarity_score) or (None, 0.0) if no good match.
    """
    best_idx = None
    best_score = 0.0

    for i, exp in enumerate(expected_fragments):
        if i in already_matched:
            continue

        score = text_similarity(predicted_text, exp.text)
        if score > best_score:
            best_score = score
            best_idx = i

    # Require at least some overlap
    if best_score < 0.1:
        return None, 0.0

    return best_idx, best_score


def analyze_decomposition(
    query: str,
    expected_fragments: list[ExpectedFragment],
    expected_primary_mass: str,
    decomposition: DecompositionResult,
) -> DecompositionAnalysis:
    """Analyze a decomposition against ground truth."""
    predicted = [(f.type.value, f.text) for f in decomposition.fragments]

    # Match predicted fragments to expected
    matches = []
    already_matched: set[int] = set()

    for pred_type, pred_text in predicted:
        match_idx, score = find_best_match(pred_text, expected_fragments, already_matched)

        if match_idx is not None:
            exp = expected_fragments[match_idx]
            already_matched.add(match_idx)
            matches.append(FragmentMatch(
                predicted_type=pred_type,
                predicted_text=pred_text,
                matched_expected_type=exp.type,
                matched_expected_text=exp.text,
                is_correct_type=(pred_type == exp.type),
                similarity_score=score,
            ))
        else:
            matches.append(FragmentMatch(
                predicted_type=pred_type,
                predicted_text=pred_text,
                matched_expected_type=None,
                matched_expected_text=None,
                is_correct_type=False,
                similarity_score=0.0,
            ))

    # Coverage: fraction of expected fragments that got matched
    coverage = len(already_matched) / len(expected_fragments) if expected_fragments else 1.0

    # Primary mass correctness
    pred_primary = decomposition.fragments[decomposition.primary_mass_index].text
    primary_correct = text_similarity(pred_primary, expected_primary_mass) > 0.3

    return DecompositionAnalysis(
        query=query,
        expected_fragments=expected_fragments,
        predicted_fragments=predicted,
        fragment_matches=matches,
        fragment_count_expected=len(expected_fragments),
        fragment_count_predicted=len(predicted),
        fragment_count_diff=len(predicted) - len(expected_fragments),
        coverage=coverage,
        primary_mass_correct=primary_correct,
        expected_primary_text=expected_primary_mass,
        predicted_primary_text=pred_primary,
    )


def build_confusion_matrix(analyses: list[DecompositionAnalysis]) -> np.ndarray:
    """
    Build confusion matrix from all analyses.
    Rows = expected type, Columns = predicted type.
    """
    n_types = len(FRAGMENT_TYPES)
    matrix = np.zeros((n_types, n_types), dtype=int)
    type_to_idx = {t: i for i, t in enumerate(FRAGMENT_TYPES)}

    for analysis in analyses:
        for match in analysis.fragment_matches:
            if match.matched_expected_type is not None:
                exp_idx = type_to_idx[match.matched_expected_type]
                pred_idx = type_to_idx[match.predicted_type]
                matrix[exp_idx, pred_idx] += 1

    return matrix


def compute_per_type_metrics(
    confusion_matrix: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 for each type from confusion matrix."""
    metrics = {}

    for i, type_name in enumerate(FRAGMENT_TYPES):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp  # Column sum minus diagonal
        fn = confusion_matrix[i, :].sum() - tp  # Row sum minus diagonal

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[type_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    return metrics


def display_confusion_matrix(matrix: np.ndarray) -> None:
    """Display confusion matrix with rich formatting."""
    table = Table(title="Fragment Type Confusion Matrix (Expected x Predicted)")
    table.add_column("Expected", style="bold", width=12)
    for t in FRAGMENT_TYPES:
        table.add_column(t[:8], width=8, justify="right")
    table.add_column("Total", style="dim", width=8, justify="right")

    for i, type_name in enumerate(FRAGMENT_TYPES):
        row = []
        for j in range(len(FRAGMENT_TYPES)):
            val = matrix[i, j]
            if i == j:
                row.append(f"[bold green]{val}[/bold green]" if val > 0 else "[dim]0[/dim]")
            else:
                row.append(f"[red]{val}[/red]" if val > 0 else "[dim]0[/dim]")
        row_total = matrix[i, :].sum()
        table.add_row(type_name[:12], *row, str(row_total))

    # Add column totals
    col_totals = ["Total"] + [str(matrix[:, j].sum()) for j in range(len(FRAGMENT_TYPES))]
    col_totals.append(str(matrix.sum()))
    table.add_row(*col_totals, style="dim")

    console.print(table)


def display_per_type_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Display per-type precision/recall/F1."""
    table = Table(title="Per-Type Classification Metrics")
    table.add_column("Type", style="bold", width=12)
    table.add_column("Precision", width=10, justify="right")
    table.add_column("Recall", width=10, justify="right")
    table.add_column("F1", width=10, justify="right")
    table.add_column("TP/FP/FN", width=12, justify="right")

    for type_name in FRAGMENT_TYPES:
        m = metrics[type_name]
        p_style = "green" if m["precision"] >= 0.7 else "yellow" if m["precision"] >= 0.4 else "red"
        r_style = "green" if m["recall"] >= 0.7 else "yellow" if m["recall"] >= 0.4 else "red"
        f1_style = "green" if m["f1"] >= 0.7 else "yellow" if m["f1"] >= 0.4 else "red"

        table.add_row(
            type_name,
            f"[{p_style}]{m['precision']:.0%}[/{p_style}]",
            f"[{r_style}]{m['recall']:.0%}[/{r_style}]",
            f"[{f1_style}]{m['f1']:.0%}[/{f1_style}]",
            f"{m['true_positives']}/{m['false_positives']}/{m['false_negatives']}",
        )

    console.print(table)


def display_summary_metrics(analyses: list[DecompositionAnalysis]) -> None:
    """Display aggregate metrics across all queries."""
    table = Table(title="Aggregate Classification Metrics")
    table.add_column("Metric", width=30)
    table.add_column("Value", width=20, justify="right")

    # Fragment count accuracy
    exact_count = sum(1 for a in analyses if a.fragment_count_diff == 0)
    within_1 = sum(1 for a in analyses if abs(a.fragment_count_diff) <= 1)
    avg_diff = np.mean([a.fragment_count_diff for a in analyses])

    table.add_row("Exact fragment count match", f"{exact_count}/{len(analyses)}")
    table.add_row("Within +/-1 fragment", f"{within_1}/{len(analyses)}")
    table.add_row("Avg count difference", f"{avg_diff:+.1f}")

    # Coverage
    coverages = [a.coverage for a in analyses]
    table.add_row("Mean coverage", f"{np.mean(coverages):.0%}")
    table.add_row("Min coverage", f"{np.min(coverages):.0%}")
    table.add_row("Full coverage (100%)", f"{sum(1 for c in coverages if c >= 1.0)}/{len(analyses)}")

    # Primary mass
    primary_correct = sum(1 for a in analyses if a.primary_mass_correct)
    table.add_row("Primary mass agreement", f"{primary_correct}/{len(analyses)} ({primary_correct/len(analyses):.0%})")

    console.print(table)


def display_hybrid_analysis(analyses: list[DecompositionAnalysis]) -> None:
    """Analyze if queries with more fragments decompose worse."""
    table = Table(title="Hybrid Query Analysis (by expected fragment count)")
    table.add_column("# Fragments", width=12)
    table.add_column("Count", width=8, justify="right")
    table.add_column("Avg Coverage", width=12, justify="right")
    table.add_column("Type Accuracy", width=12, justify="right")
    table.add_column("Primary Acc", width=12, justify="right")

    # Group by fragment count
    by_count: dict[int, list[DecompositionAnalysis]] = defaultdict(list)
    for a in analyses:
        by_count[a.fragment_count_expected].append(a)

    for count in sorted(by_count.keys()):
        group = by_count[count]
        avg_coverage = np.mean([a.coverage for a in group])

        # Type accuracy: correctly typed matches / all matches
        correct_types = sum(
            1 for a in group
            for m in a.fragment_matches
            if m.is_correct_type and m.matched_expected_type is not None
        )
        total_matches = sum(
            1 for a in group
            for m in a.fragment_matches
            if m.matched_expected_type is not None
        )
        type_acc = correct_types / total_matches if total_matches > 0 else 0

        primary_acc = sum(1 for a in group if a.primary_mass_correct) / len(group)

        cov_style = "green" if avg_coverage >= 0.8 else "yellow" if avg_coverage >= 0.5 else "red"
        type_style = "green" if type_acc >= 0.7 else "yellow" if type_acc >= 0.4 else "red"
        prim_style = "green" if primary_acc >= 0.7 else "yellow" if primary_acc >= 0.4 else "red"

        table.add_row(
            str(count),
            str(len(group)),
            f"[{cov_style}]{avg_coverage:.0%}[/{cov_style}]",
            f"[{type_style}]{type_acc:.0%}[/{type_style}]",
            f"[{prim_style}]{primary_acc:.0%}[/{prim_style}]",
        )

    console.print(table)


def display_problematic_decompositions(analyses: list[DecompositionAnalysis]) -> None:
    """Show the most problematic decompositions."""
    # Score each analysis: penalize low coverage, wrong types, wrong primary
    scored = []
    for a in analyses:
        type_correct = sum(1 for m in a.fragment_matches if m.is_correct_type and m.matched_expected_type)
        type_total = sum(1 for m in a.fragment_matches if m.matched_expected_type)
        type_acc = type_correct / type_total if type_total > 0 else 0

        score = a.coverage * 0.4 + type_acc * 0.4 + (1.0 if a.primary_mass_correct else 0.0) * 0.2
        scored.append((score, a))

    scored.sort(key=lambda x: x[0])

    table = Table(title="Most Problematic Decompositions (lowest scores)")
    table.add_column("Query", width=45)
    table.add_column("Coverage", width=10, justify="right")
    table.add_column("Type Acc", width=10, justify="right")
    table.add_column("Primary", width=8)
    table.add_column("Issue", width=35)

    for score, a in scored[:8]:
        type_correct = sum(1 for m in a.fragment_matches if m.is_correct_type and m.matched_expected_type)
        type_total = sum(1 for m in a.fragment_matches if m.matched_expected_type)
        type_acc = type_correct / type_total if type_total > 0 else 0

        # Identify main issue
        issues = []
        if a.coverage < 0.8:
            missed = [e.text for i, e in enumerate(a.expected_fragments)
                     if not any(m.matched_expected_text == e.text for m in a.fragment_matches)]
            if missed:
                issues.append(f"missed: {', '.join(missed[:2])}")
        if type_acc < 0.7:
            confused = [(m.matched_expected_type, m.predicted_type)
                       for m in a.fragment_matches
                       if not m.is_correct_type and m.matched_expected_type]
            if confused:
                issues.append(f"{confused[0][0]}->{confused[0][1]}")
        if not a.primary_mass_correct:
            issues.append(f"primary: exp={a.expected_primary_text[:15]}, got={a.predicted_primary_text[:15]}")

        cov_style = "green" if a.coverage >= 0.8 else "yellow" if a.coverage >= 0.5 else "red"
        type_style = "green" if type_acc >= 0.7 else "yellow" if type_acc >= 0.4 else "red"
        prim = "[green]Y[/green]" if a.primary_mass_correct else "[red]N[/red]"

        table.add_row(
            a.query[:45],
            f"[{cov_style}]{a.coverage:.0%}[/{cov_style}]",
            f"[{type_style}]{type_acc:.0%}[/{type_style}]",
            prim,
            "; ".join(issues)[:35] if issues else "[green]OK[/green]",
        )

    console.print(table)


def save_results(
    analyses: list[DecompositionAnalysis],
    confusion_matrix: np.ndarray,
    per_type_metrics: dict[str, dict[str, float]],
) -> Path:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"classification_diagnostics_{timestamp}.json"

    # Convert analyses to JSON-serializable format
    analyses_data = []
    for a in analyses:
        analyses_data.append({
            "query": a.query,
            "fragment_count_expected": a.fragment_count_expected,
            "fragment_count_predicted": a.fragment_count_predicted,
            "fragment_count_diff": a.fragment_count_diff,
            "coverage": a.coverage,
            "primary_mass_correct": a.primary_mass_correct,
            "expected_primary": a.expected_primary_text,
            "predicted_primary": a.predicted_primary_text,
            "expected_fragments": [
                {"type": f.type, "text": f.text, "is_primary": f.is_primary}
                for f in a.expected_fragments
            ],
            "predicted_fragments": [
                {"type": t, "text": txt} for t, txt in a.predicted_fragments
            ],
            "matches": [
                {
                    "predicted_type": m.predicted_type,
                    "predicted_text": m.predicted_text,
                    "expected_type": m.matched_expected_type,
                    "expected_text": m.matched_expected_text,
                    "is_correct_type": m.is_correct_type,
                    "similarity": m.similarity_score,
                }
                for m in a.fragment_matches
            ],
        })

    data = {
        "timestamp": timestamp,
        "total_queries": len(analyses),
        "confusion_matrix": confusion_matrix.tolist(),
        "fragment_types": FRAGMENT_TYPES,
        "per_type_metrics": per_type_metrics,
        "aggregate_metrics": {
            "mean_coverage": float(np.mean([a.coverage for a in analyses])),
            "primary_mass_accuracy": sum(1 for a in analyses if a.primary_mass_correct) / len(analyses),
            "exact_count_match_rate": sum(1 for a in analyses if a.fragment_count_diff == 0) / len(analyses),
        },
        "analyses": analyses_data,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fragment Classification Confusion Matrix")
    parser.add_argument(
        "--skip-decompose",
        action="store_true",
        help="Reuse cached decompositions (skip Claude API calls)",
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Fragment Classification Diagnostics[/bold]\n"
        "Analyzing how reliably Claude classifies fragments into 6 types",
        border_style="blue",
    ))

    # Load cache
    cache = load_cache()
    console.print(f"[dim]Loaded {len(cache)} cached decompositions[/dim]")

    # Run decomposition for each test query
    analyses: list[DecompositionAnalysis] = []
    skipped = 0

    for i, test in enumerate(TEST_QUERIES):
        if not test.expected_fragments:
            console.print(f"[yellow]Skipping query {i}: no expected fragments annotated[/yellow]")
            skipped += 1
            continue

        result = get_or_decompose(test.query, cache, skip_decompose=args.skip_decompose)
        if result is None:
            console.print(f"[yellow]Skipping query {i}: not in cache and --skip-decompose[/yellow]")
            skipped += 1
            continue

        analysis = analyze_decomposition(
            test.query,
            test.expected_fragments,
            test.expected_primary_mass,
            result,
        )
        analyses.append(analysis)
        console.print(f"[dim]  [{i+1}/{len(TEST_QUERIES)}] {test.query[:50]}...[/dim]")

    # Save cache
    save_cache(cache)

    if not analyses:
        console.print("[red]No analyses to display. Run without --skip-decompose first.[/red]")
        return

    console.print()

    # Build and display confusion matrix
    matrix = build_confusion_matrix(analyses)
    display_confusion_matrix(matrix)

    # Per-type metrics
    per_type = compute_per_type_metrics(matrix)
    display_per_type_metrics(per_type)

    # Summary metrics
    display_summary_metrics(analyses)

    # Hybrid query analysis
    display_hybrid_analysis(analyses)

    # Problematic decompositions
    display_problematic_decompositions(analyses)

    # Save results
    output_path = save_results(analyses, matrix, per_type)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
