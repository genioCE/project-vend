#!/usr/bin/env python3
"""
Query-Type Miss Analysis

Analyzes which query archetypes have the worst recall, identifies near-misses
vs systematic failures, and shows which tools consistently fail for certain
query types.

Usage:
    python miss_analysis.py              # Run full analysis
    python miss_analysis.py --skip-decompose  # Reuse cached decompositions
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cache import load_cache, save_cache, get_or_decompose
from decompose import embed_decomposition
from fragments import DecompositionResult
from gravity_field import compute_gravity_field, ActivatedTool
from test_queries import TEST_QUERIES, QueryArchetype
from tool_identities import TOOL_NAMES, load_identity_vectors

console = Console()
RESULTS_DIR = Path(__file__).parent / "results"

# Near miss threshold: within this percentage of the cutoff
NEAR_MISS_THRESHOLD = 0.10
# Clear miss threshold: more than this percentage below cutoff
CLEAR_MISS_THRESHOLD = 0.25


@dataclass
class MissedTool:
    """Details about a tool that should have fired but didn't."""
    name: str
    composite_score: float
    adaptive_cutoff: float
    gap_from_cutoff: float  # Negative = below cutoff
    gap_percentage: float
    fragment_pulls: list[tuple[str, float]]  # (fragment_text, pull)
    query_pull: float
    is_near_miss: bool
    is_clear_miss: bool


@dataclass
class FalseActivation:
    """Details about a tool that fired but shouldn't have."""
    name: str
    composite_score: float
    is_meta: bool


@dataclass
class QueryAnalysis:
    """Full analysis of a single query's activation pattern."""
    query: str
    archetype: QueryArchetype
    recall: float
    precision: float
    total_activated: int
    correct_active: list[str]
    missed_tools: list[MissedTool]
    false_activations: list[FalseActivation]
    adaptive_cutoff: float


@dataclass
class ArchetypeStats:
    """Aggregate statistics for a query archetype."""
    archetype: QueryArchetype
    query_count: int
    mean_recall: float
    mean_precision: float
    min_recall: float
    most_missed_tools: list[tuple[str, int]]  # (tool_name, miss_count)
    most_false_tools: list[tuple[str, int]]
    near_miss_count: int
    clear_miss_count: int


def analyze_query(
    query: str,
    archetype: QueryArchetype,
    expected_active: list[str],
    expected_inactive: list[str],
    decomposition: DecompositionResult,
    identity_vectors: np.ndarray,
) -> QueryAnalysis:
    """Run gravity field and analyze activation results."""
    gfield = compute_gravity_field(identity_vectors, decomposition)

    activated_names = {t.name for t in gfield.activated}

    # Correct activations
    correct = [name for name in expected_active if name in activated_names]

    # Missed tools
    missed: list[MissedTool] = []
    for name in expected_active:
        if name not in activated_names:
            # Find this tool's info
            tool_info = next((t for t in gfield.all_tools if t.name == name), None)
            if tool_info:
                gap = tool_info.composite_score - gfield.adaptive_cutoff
                gap_pct = gap / gfield.adaptive_cutoff if gfield.adaptive_cutoff > 0 else 0

                fragment_pulls = [
                    (decomposition.fragments[i].text, pull)
                    for i, pull in enumerate(tool_info.pulls)
                ]

                missed.append(MissedTool(
                    name=name,
                    composite_score=tool_info.composite_score,
                    adaptive_cutoff=gfield.adaptive_cutoff,
                    gap_from_cutoff=gap,
                    gap_percentage=gap_pct,
                    fragment_pulls=fragment_pulls,
                    query_pull=tool_info.query_pull,
                    is_near_miss=abs(gap_pct) <= NEAR_MISS_THRESHOLD,
                    is_clear_miss=abs(gap_pct) > CLEAR_MISS_THRESHOLD,
                ))

    # False activations
    false_active: list[FalseActivation] = []
    for name in expected_inactive:
        if name in activated_names:
            tool_info = next((t for t in gfield.activated if t.name == name), None)
            if tool_info:
                false_active.append(FalseActivation(
                    name=name,
                    composite_score=tool_info.composite_score,
                    is_meta=tool_info.is_meta,
                ))

    # Metrics
    recall = len(correct) / len(expected_active) if expected_active else 1.0
    precision = len(correct) / len(activated_names) if activated_names else 0.0

    return QueryAnalysis(
        query=query,
        archetype=archetype,
        recall=recall,
        precision=precision,
        total_activated=len(activated_names),
        correct_active=correct,
        missed_tools=missed,
        false_activations=false_active,
        adaptive_cutoff=gfield.adaptive_cutoff,
    )


def compute_archetype_stats(
    analyses: list[QueryAnalysis],
) -> dict[QueryArchetype, ArchetypeStats]:
    """Compute per-archetype statistics."""
    by_archetype: dict[QueryArchetype, list[QueryAnalysis]] = defaultdict(list)
    for a in analyses:
        by_archetype[a.archetype].append(a)

    stats = {}
    for archetype, group in by_archetype.items():
        recalls = [a.recall for a in group]
        precisions = [a.precision for a in group]

        # Count tool misses
        missed_counts: dict[str, int] = defaultdict(int)
        false_counts: dict[str, int] = defaultdict(int)
        near_miss_count = 0
        clear_miss_count = 0

        for a in group:
            for m in a.missed_tools:
                missed_counts[m.name] += 1
                if m.is_near_miss:
                    near_miss_count += 1
                if m.is_clear_miss:
                    clear_miss_count += 1
            for f in a.false_activations:
                false_counts[f.name] += 1

        stats[archetype] = ArchetypeStats(
            archetype=archetype,
            query_count=len(group),
            mean_recall=float(np.mean(recalls)),
            mean_precision=float(np.mean(precisions)),
            min_recall=float(np.min(recalls)),
            most_missed_tools=sorted(missed_counts.items(), key=lambda x: -x[1])[:3],
            most_false_tools=sorted(false_counts.items(), key=lambda x: -x[1])[:3],
            near_miss_count=near_miss_count,
            clear_miss_count=clear_miss_count,
        )

    return stats


def display_archetype_summary(stats: dict[QueryArchetype, ArchetypeStats]) -> None:
    """Display per-archetype summary table."""
    table = Table(title="Per-Archetype Performance")
    table.add_column("Archetype", style="bold", width=22)
    table.add_column("N", width=4, justify="right")
    table.add_column("Mean Recall", width=12, justify="right")
    table.add_column("Min Recall", width=11, justify="right")
    table.add_column("Near/Clear", width=11, justify="right")
    table.add_column("Most Missed", width=30)

    # Sort by mean recall ascending (worst first)
    sorted_stats = sorted(stats.values(), key=lambda s: s.mean_recall)

    for s in sorted_stats:
        recall_style = "green" if s.mean_recall >= 0.8 else "yellow" if s.mean_recall >= 0.5 else "red"
        min_style = "green" if s.min_recall >= 0.8 else "yellow" if s.min_recall >= 0.5 else "red"

        missed_str = ", ".join(f"{name}({count})" for name, count in s.most_missed_tools[:2])
        if not missed_str:
            missed_str = "[green]---[/green]"

        table.add_row(
            s.archetype,
            str(s.query_count),
            f"[{recall_style}]{s.mean_recall:.0%}[/{recall_style}]",
            f"[{min_style}]{s.min_recall:.0%}[/{min_style}]",
            f"{s.near_miss_count}/{s.clear_miss_count}",
            missed_str,
        )

    console.print(table)


def display_near_miss_report(analyses: list[QueryAnalysis]) -> None:
    """Show tools that almost fired."""
    near_misses: list[tuple[QueryAnalysis, MissedTool]] = []
    for a in analyses:
        for m in a.missed_tools:
            if m.is_near_miss:
                near_misses.append((a, m))

    if not near_misses:
        console.print("[green]No near-misses detected![/green]")
        return

    # Sort by gap percentage (closest to cutoff first)
    near_misses.sort(key=lambda x: abs(x[1].gap_percentage))

    table = Table(title=f"Near-Miss Report ({len(near_misses)} near-misses)")
    table.add_column("Query", width=35)
    table.add_column("Tool", width=24)
    table.add_column("Score", width=8, justify="right")
    table.add_column("Cutoff", width=8, justify="right")
    table.add_column("Gap", width=8, justify="right")
    table.add_column("Best Pull", width=20)

    for analysis, miss in near_misses[:12]:
        # Find strongest pull
        if miss.fragment_pulls:
            best_frag, best_pull = max(miss.fragment_pulls, key=lambda x: x[1])
            best_pull_str = f"{best_frag[:12]}={best_pull:.3f}"
        else:
            best_pull_str = f"query={miss.query_pull:.3f}"

        gap_style = "yellow" if abs(miss.gap_percentage) <= 0.05 else "dim"

        table.add_row(
            analysis.query[:35],
            miss.name,
            f"{miss.composite_score:.3f}",
            f"{miss.adaptive_cutoff:.3f}",
            f"[{gap_style}]{miss.gap_percentage:+.0%}[/{gap_style}]",
            best_pull_str,
        )

    console.print(table)


def display_systematic_miss_report(analyses: list[QueryAnalysis]) -> None:
    """Show tools that consistently fail for certain query types."""
    # Track misses by (archetype, tool)
    archetype_tool_misses: dict[tuple[QueryArchetype, str], int] = defaultdict(int)
    archetype_query_count: dict[QueryArchetype, int] = defaultdict(int)

    for a in analyses:
        archetype_query_count[a.archetype] += 1
        for m in a.missed_tools:
            archetype_tool_misses[(a.archetype, m.name)] += 1

    # Find tools that miss >50% of the time for an archetype with >1 query
    systematic: list[tuple[QueryArchetype, str, int, int, float]] = []
    for (archetype, tool), miss_count in archetype_tool_misses.items():
        query_count = archetype_query_count[archetype]
        if query_count > 1:
            miss_rate = miss_count / query_count
            if miss_rate >= 0.5:
                systematic.append((archetype, tool, miss_count, query_count, miss_rate))

    if not systematic:
        console.print("[green]No systematic misses detected![/green]")
        return

    # Sort by miss rate descending
    systematic.sort(key=lambda x: -x[4])

    table = Table(title="Systematic Misses (>50% miss rate per archetype)")
    table.add_column("Archetype", style="bold", width=22)
    table.add_column("Tool", width=24)
    table.add_column("Miss Rate", width=10, justify="right")
    table.add_column("Misses/Total", width=12, justify="right")

    for archetype, tool, miss_count, query_count, miss_rate in systematic:
        rate_style = "red" if miss_rate >= 0.75 else "yellow"
        table.add_row(
            archetype,
            tool,
            f"[{rate_style}]{miss_rate:.0%}[/{rate_style}]",
            f"{miss_count}/{query_count}",
        )

    console.print(table)


def display_global_tool_stats(analyses: list[QueryAnalysis]) -> None:
    """Show global miss/false-activation stats by tool."""
    tool_misses: dict[str, int] = defaultdict(int)
    tool_false: dict[str, int] = defaultdict(int)
    tool_expected: dict[str, int] = defaultdict(int)
    tool_expected_inactive: dict[str, int] = defaultdict(int)

    for test in TEST_QUERIES:
        for name in test.expected_active:
            tool_expected[name] += 1
        for name in test.expected_inactive:
            tool_expected_inactive[name] += 1

    for a in analyses:
        for m in a.missed_tools:
            tool_misses[m.name] += 1
        for f in a.false_activations:
            tool_false[f.name] += 1

    # Calculate miss rates
    miss_rates: list[tuple[str, int, int, float]] = []
    for name in TOOL_NAMES:
        expected = tool_expected.get(name, 0)
        missed = tool_misses.get(name, 0)
        if expected > 0:
            miss_rates.append((name, missed, expected, missed / expected))

    miss_rates.sort(key=lambda x: -x[3])

    table = Table(title="Per-Tool Miss Rates")
    table.add_column("Tool", style="bold", width=26)
    table.add_column("Misses", width=8, justify="right")
    table.add_column("Expected", width=9, justify="right")
    table.add_column("Miss Rate", width=10, justify="right")

    for name, missed, expected, rate in miss_rates[:15]:
        if expected == 0:
            continue
        rate_style = "red" if rate >= 0.3 else "yellow" if rate > 0 else "green"
        table.add_row(
            name,
            str(missed),
            str(expected),
            f"[{rate_style}]{rate:.0%}[/{rate_style}]",
        )

    console.print(table)

    # False activation rates
    if tool_false:
        false_rates: list[tuple[str, int, int, float]] = []
        for name in TOOL_NAMES:
            expected_inactive = tool_expected_inactive.get(name, 0)
            false_count = tool_false.get(name, 0)
            if expected_inactive > 0 and false_count > 0:
                false_rates.append((name, false_count, expected_inactive, false_count / expected_inactive))

        false_rates.sort(key=lambda x: -x[3])

        if false_rates:
            table2 = Table(title="False Activation Rates")
            table2.add_column("Tool", style="bold", width=26)
            table2.add_column("False", width=8, justify="right")
            table2.add_column("Exp Inactive", width=12, justify="right")
            table2.add_column("False Rate", width=10, justify="right")

            for name, false_count, exp_inactive, rate in false_rates[:10]:
                table2.add_row(name, str(false_count), str(exp_inactive), f"{rate:.0%}")

            console.print(table2)


def display_query_details(analyses: list[QueryAnalysis]) -> None:
    """Show per-query details for queries with misses."""
    queries_with_misses = [a for a in analyses if a.missed_tools]

    if not queries_with_misses:
        console.print("[green]No queries with misses![/green]")
        return

    # Sort by recall ascending
    queries_with_misses.sort(key=lambda a: a.recall)

    table = Table(title="Queries with Misses")
    table.add_column("#", width=3)
    table.add_column("Query", width=40)
    table.add_column("Archetype", width=18)
    table.add_column("Recall", width=8, justify="right")
    table.add_column("Missed Tools", width=35)

    for i, a in enumerate(queries_with_misses):
        recall_style = "red" if a.recall < 0.5 else "yellow" if a.recall < 0.8 else "dim"

        missed_str = ", ".join(
            f"{m.name}({'NM' if m.is_near_miss else 'CM' if m.is_clear_miss else 'M'})"
            for m in a.missed_tools
        )

        table.add_row(
            str(i),
            a.query[:40],
            a.archetype,
            f"[{recall_style}]{a.recall:.0%}[/{recall_style}]",
            missed_str[:35],
        )

    console.print(table)


def save_results(
    analyses: list[QueryAnalysis],
    archetype_stats: dict[QueryArchetype, ArchetypeStats],
) -> Path:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"miss_analysis_{timestamp}.json"

    # Convert to JSON-serializable format
    analyses_data = []
    for a in analyses:
        analyses_data.append({
            "query": a.query,
            "archetype": a.archetype,
            "recall": a.recall,
            "precision": a.precision,
            "total_activated": a.total_activated,
            "correct_active": a.correct_active,
            "adaptive_cutoff": a.adaptive_cutoff,
            "missed_tools": [
                {
                    "name": m.name,
                    "composite_score": m.composite_score,
                    "adaptive_cutoff": m.adaptive_cutoff,
                    "gap_from_cutoff": m.gap_from_cutoff,
                    "gap_percentage": m.gap_percentage,
                    "fragment_pulls": [
                        {"fragment": f, "pull": p} for f, p in m.fragment_pulls
                    ],
                    "query_pull": m.query_pull,
                    "is_near_miss": m.is_near_miss,
                    "is_clear_miss": m.is_clear_miss,
                }
                for m in a.missed_tools
            ],
            "false_activations": [
                {"name": f.name, "composite_score": f.composite_score, "is_meta": f.is_meta}
                for f in a.false_activations
            ],
        })

    stats_data = {}
    for archetype, s in archetype_stats.items():
        stats_data[archetype] = {
            "query_count": s.query_count,
            "mean_recall": s.mean_recall,
            "mean_precision": s.mean_precision,
            "min_recall": s.min_recall,
            "most_missed_tools": [{"tool": t, "count": c} for t, c in s.most_missed_tools],
            "most_false_tools": [{"tool": t, "count": c} for t, c in s.most_false_tools],
            "near_miss_count": s.near_miss_count,
            "clear_miss_count": s.clear_miss_count,
        }

    # Aggregate metrics
    all_recalls = [a.recall for a in analyses]
    near_misses = sum(1 for a in analyses for m in a.missed_tools if m.is_near_miss)
    clear_misses = sum(1 for a in analyses for m in a.missed_tools if m.is_clear_miss)

    data = {
        "timestamp": timestamp,
        "total_queries": len(analyses),
        "aggregate_metrics": {
            "mean_recall": float(np.mean(all_recalls)),
            "min_recall": float(np.min(all_recalls)),
            "perfect_recall_count": sum(1 for r in all_recalls if r >= 1.0),
            "total_near_misses": near_misses,
            "total_clear_misses": clear_misses,
        },
        "archetype_stats": stats_data,
        "analyses": analyses_data,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Query-Type Miss Analysis")
    parser.add_argument(
        "--skip-decompose",
        action="store_true",
        help="Reuse cached decompositions (skip Claude API calls)",
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Query-Type Miss Analysis[/bold]\n"
        "Analyzing which query archetypes have the worst recall",
        border_style="blue",
    ))

    # Load cache and embedding model
    cache = load_cache()
    console.print(f"[dim]Loaded {len(cache)} cached decompositions[/dim]")

    console.print("[dim]Loading embedding model...[/dim]")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    console.print("[dim]Loading tool identity vectors...[/dim]")
    identity_vectors = load_identity_vectors(model=embedding_model)

    # Run analysis for each test query
    analyses: list[QueryAnalysis] = []
    skipped = 0

    for i, test in enumerate(TEST_QUERIES):
        result = get_or_decompose(test.query, cache, skip_decompose=args.skip_decompose)
        if result is None:
            console.print(f"[yellow]Skipping query {i}: not in cache and --skip-decompose[/yellow]")
            skipped += 1
            continue

        # Embed the decomposition
        result = embed_decomposition(result, test.query, model=embedding_model)

        analysis = analyze_query(
            test.query,
            test.archetype,
            test.expected_active,
            test.expected_inactive,
            result,
            identity_vectors,
        )
        analyses.append(analysis)
        console.print(f"[dim]  [{i+1}/{len(TEST_QUERIES)}] {test.query[:50]}...[/dim]")

    # Save cache
    save_cache(cache)

    if not analyses:
        console.print("[red]No analyses to display. Run without --skip-decompose first.[/red]")
        return

    console.print()

    # Compute archetype stats
    archetype_stats = compute_archetype_stats(analyses)

    # Display results
    display_archetype_summary(archetype_stats)
    console.print()

    display_near_miss_report(analyses)
    console.print()

    display_systematic_miss_report(analyses)
    console.print()

    display_global_tool_stats(analyses)
    console.print()

    display_query_details(analyses)

    # Save results
    output_path = save_results(analyses, archetype_stats)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
