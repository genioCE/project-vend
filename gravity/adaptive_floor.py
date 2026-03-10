#!/usr/bin/env python3
"""
Adaptive Floor Exploration

Simulates reliability-biased gravity fields with different floor values
to determine the optimal reliability bias floor.

Usage:
    python adaptive_floor.py              # Run full simulation
    python adaptive_floor.py --skip-decompose  # Reuse cached decompositions
    python adaptive_floor.py --no-plot    # Skip matplotlib plot
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cache import load_cache, save_cache, get_or_decompose
from decompose import embed_decomposition
from fragments import DecompositionResult, FragmentType
from gravity_field import (
    compute_pull_matrix,
    compute_composite_activation,
    find_activation_cutoff,
)
from test_queries import TEST_QUERIES
from tool_identities import TOOL_NAMES, META_TOOLS, ALWAYS_ACTIVE_TOOLS, load_identity_vectors

console = Console()
RESULTS_DIR = Path(__file__).parent / "results"

# Simulation parameters
N_SYNTHETIC_OUTCOMES = 200
DECAY_CONSTANT = 0.0139  # Same as TypeScript ledger


@dataclass
class ToolOutcome:
    """Simulated outcome for a tool activation."""
    tool: str
    composite_score: float
    errored: bool
    empty: bool
    result_size: int


@dataclass
class GravityOutcome:
    """Simulated gravity outcome record."""
    query: str
    fragments: list[tuple[str, str]]  # (type, text)
    activated_tools: list[str]
    tool_outcomes: list[ToolOutcome]


@dataclass
class ToolReliability:
    """Computed reliability profile for a tool."""
    tool: str
    total_activations: float
    error_rate: float
    empty_rate: float
    useful_rate: float
    by_fragment_type: dict[str, dict[str, float]]  # {fragment_type: {activations, useful_rate}}


def generate_synthetic_ledger(
    decompositions: list[tuple[str, DecompositionResult]],
    identity_vectors: np.ndarray,
) -> list[GravityOutcome]:
    """
    Generate synthetic ledger outcomes based on test query decompositions.

    Uses realistic distributions for tool outcomes:
    - Some tools have high useful rates (0.7-0.9)
    - Some tools have moderate useful rates (0.4-0.6)
    - Some tools have low useful rates (0.1-0.3)
    - Error rates are generally low (0.01-0.1)
    """
    outcomes = []

    # Assign baseline useful rates to tools (consistent across simulation)
    tool_base_rates: dict[str, float] = {}
    for name in TOOL_NAMES:
        # Distribution: 40% high (0.7-0.9), 40% moderate (0.4-0.6), 20% low (0.1-0.3)
        r = random.random()
        if r < 0.4:
            tool_base_rates[name] = random.uniform(0.7, 0.9)
        elif r < 0.8:
            tool_base_rates[name] = random.uniform(0.4, 0.6)
        else:
            tool_base_rates[name] = random.uniform(0.1, 0.3)

    # Generate outcomes by sampling from decompositions
    for i in range(N_SYNTHETIC_OUTCOMES):
        # Pick a random decomposition as template
        query, decomp = random.choice(decompositions)

        # Compute gravity field for this decomposition
        fragment_vectors = np.array(
            [f.embedding for f in decomp.fragments],
            dtype=np.float32,
        )
        query_vector = decomp.query_embedding
        all_vectors = np.vstack([fragment_vectors, query_vector.reshape(1, -1)])

        pull_matrix = compute_pull_matrix(identity_vectors, all_vectors)
        composite_scores = compute_composite_activation(pull_matrix)

        # Determine activated tools
        gated_scores = [
            (name, float(composite_scores[j]))
            for j, name in enumerate(TOOL_NAMES)
            if name not in META_TOOLS and name not in ALWAYS_ACTIVE_TOOLS
        ]
        gated_scores_only = [s for _, s in gated_scores]
        cutoff, _ = find_activation_cutoff(gated_scores_only)

        activated = []
        for name in ALWAYS_ACTIVE_TOOLS:
            activated.append(name)
        for name, score in gated_scores:
            if score >= cutoff:
                activated.append(name)

        # Generate outcomes for activated tools
        tool_outcomes = []
        fragment_types = [f.type.value for f in decomp.fragments]

        for name in activated:
            idx = TOOL_NAMES.index(name)
            composite = float(composite_scores[idx])

            # Determine outcome based on base rate + some noise
            base_useful = tool_base_rates[name]
            # Add noise: +/- 20% variation
            useful_chance = base_useful + random.uniform(-0.2, 0.2)
            useful_chance = max(0.05, min(0.95, useful_chance))

            error_chance = random.uniform(0.01, 0.08)

            roll = random.random()
            if roll < error_chance:
                errored = True
                empty = True
                result_size = 0
            elif roll < error_chance + (1 - useful_chance):
                errored = False
                empty = True
                result_size = random.randint(10, 100)
            else:
                errored = False
                empty = False
                result_size = random.randint(500, 5000)

            tool_outcomes.append(ToolOutcome(
                tool=name,
                composite_score=composite,
                errored=errored,
                empty=empty,
                result_size=result_size,
            ))

        outcomes.append(GravityOutcome(
            query=query,
            fragments=[(f.type.value, f.text) for f in decomp.fragments],
            activated_tools=activated,
            tool_outcomes=tool_outcomes,
        ))

    return outcomes


def decay_weight(queries_ago: int) -> float:
    """Compute exponential decay weight."""
    return math.exp(-DECAY_CONSTANT * queries_ago)


def compute_reliability(outcomes: list[GravityOutcome]) -> dict[str, ToolReliability]:
    """
    Compute reliability profiles from outcomes.
    Python port of the TypeScript computeReliability() function.
    """
    if not outcomes:
        return {}

    # Accumulate weighted stats per tool
    tool_stats: dict[str, dict] = defaultdict(lambda: {
        "total_weight": 0.0,
        "error_weight": 0.0,
        "empty_weight": 0.0,
        "useful_weight": 0.0,
        "by_fragment_type": defaultdict(lambda: {"total_weight": 0.0, "useful_weight": 0.0}),
    })

    total_outcomes = len(outcomes)

    for i, outcome in enumerate(reversed(outcomes)):
        queries_ago = i
        weight = decay_weight(queries_ago)
        fragment_types = set(ftype for ftype, _ in outcome.fragments)

        for tool_outcome in outcome.tool_outcomes:
            stats = tool_stats[tool_outcome.tool]
            stats["total_weight"] += weight

            if tool_outcome.errored:
                stats["error_weight"] += weight
            elif tool_outcome.empty:
                stats["empty_weight"] += weight
            else:
                stats["useful_weight"] += weight

            # Track per-fragment-type stats
            for frag_type in fragment_types:
                frag_stats = stats["by_fragment_type"][frag_type]
                frag_stats["total_weight"] += weight
                if not tool_outcome.errored and not tool_outcome.empty:
                    frag_stats["useful_weight"] += weight

    # Build reliability map
    reliability_map = {}
    for tool, stats in tool_stats.items():
        total = stats["total_weight"]
        error_rate = stats["error_weight"] / total if total > 0 else 0
        empty_rate = stats["empty_weight"] / total if total > 0 else 0
        useful_rate = stats["useful_weight"] / total if total > 0 else 0

        by_frag = {}
        for frag_type, frag_stats in stats["by_fragment_type"].items():
            ft = frag_stats["total_weight"]
            by_frag[frag_type] = {
                "activations": ft,
                "useful_rate": frag_stats["useful_weight"] / ft if ft > 0 else 0,
            }

        reliability_map[tool] = ToolReliability(
            tool=tool,
            total_activations=total,
            error_rate=error_rate,
            empty_rate=empty_rate,
            useful_rate=useful_rate,
            by_fragment_type=by_frag,
        )

    return reliability_map


def apply_reliability_bias(
    composite_scores: np.ndarray,
    reliability_map: dict[str, ToolReliability],
    floor: float,
) -> np.ndarray:
    """
    Apply reliability bias to composite scores.

    biased_score = composite * (floor + (1 - floor) * useful_rate)

    This means:
    - If floor=0.3 and useful_rate=0.0, tool gets 30% of its score
    - If floor=0.3 and useful_rate=1.0, tool gets 100% of its score
    - If floor=0.3 and useful_rate=0.5, tool gets 65% of its score
    """
    biased = np.copy(composite_scores)

    for i, name in enumerate(TOOL_NAMES):
        if name in reliability_map:
            useful_rate = reliability_map[name].useful_rate
            bias = floor + (1 - floor) * useful_rate
            biased[i] = composite_scores[i] * bias

    return biased


def evaluate_with_floor(
    decompositions: list[tuple[str, DecompositionResult, list[str]]],
    identity_vectors: np.ndarray,
    reliability_map: dict[str, ToolReliability],
    floor: float,
) -> list[float]:
    """
    Run all test queries with reliability bias and compute recalls.
    Returns list of per-query recall values.
    """
    recalls = []

    for query, decomp, expected_active in decompositions:
        # Compute gravity field
        fragment_vectors = np.array(
            [f.embedding for f in decomp.fragments],
            dtype=np.float32,
        )
        query_vector = decomp.query_embedding
        all_vectors = np.vstack([fragment_vectors, query_vector.reshape(1, -1)])

        pull_matrix = compute_pull_matrix(identity_vectors, all_vectors)
        composite_scores = compute_composite_activation(pull_matrix)

        # Apply reliability bias
        biased_scores = apply_reliability_bias(composite_scores, reliability_map, floor)

        # Find cutoff on biased scores
        gated_scores = [
            (name, float(biased_scores[j]))
            for j, name in enumerate(TOOL_NAMES)
            if name not in META_TOOLS and name not in ALWAYS_ACTIVE_TOOLS
        ]
        gated_scores_only = [s for _, s in gated_scores]
        cutoff, _ = find_activation_cutoff(gated_scores_only)

        # Determine activated tools
        activated = set(ALWAYS_ACTIVE_TOOLS)
        for name, score in gated_scores:
            if score >= cutoff:
                activated.add(name)

        # Meta tools: only if in top 3 of ALL biased scores
        all_biased_sorted = sorted(
            [(TOOL_NAMES[i], biased_scores[i]) for i in range(len(TOOL_NAMES))],
            key=lambda x: -x[1],
        )
        top_3_threshold = all_biased_sorted[2][1] if len(all_biased_sorted) >= 3 else 0
        for name in META_TOOLS:
            idx = TOOL_NAMES.index(name)
            if biased_scores[idx] >= top_3_threshold:
                activated.add(name)

        # Compute recall
        correct = sum(1 for name in expected_active if name in activated)
        recall = correct / len(expected_active) if expected_active else 1.0
        recalls.append(recall)

    return recalls


@dataclass
class FloorSweepResult:
    """Result of evaluating a single floor value."""
    floor: float
    mean_recall: float
    std_recall: float
    min_recall: float
    max_recall: float
    perfect_recall_count: int


def run_floor_sweep(
    decompositions: list[tuple[str, DecompositionResult, list[str]]],
    identity_vectors: np.ndarray,
    reliability_map: dict[str, ToolReliability],
    floor_values: list[float],
) -> list[FloorSweepResult]:
    """Sweep floor values and compute metrics for each."""
    results = []

    for floor in floor_values:
        recalls = evaluate_with_floor(
            decompositions, identity_vectors, reliability_map, floor
        )

        results.append(FloorSweepResult(
            floor=floor,
            mean_recall=float(np.mean(recalls)),
            std_recall=float(np.std(recalls)),
            min_recall=float(np.min(recalls)),
            max_recall=float(np.max(recalls)),
            perfect_recall_count=sum(1 for r in recalls if r >= 1.0),
        ))

    return results


def display_floor_sweep(results: list[FloorSweepResult]) -> None:
    """Display floor sweep results."""
    table = Table(title="Floor Sweep Results")
    table.add_column("Floor", width=8, justify="right")
    table.add_column("Mean Recall", width=12, justify="right")
    table.add_column("Std Dev", width=10, justify="right")
    table.add_column("Min", width=8, justify="right")
    table.add_column("Max", width=8, justify="right")
    table.add_column("Perfect", width=8, justify="right")

    best_floor = max(results, key=lambda r: r.mean_recall)

    for r in results:
        style = "bold green" if r == best_floor else ""
        recall_style = "green" if r.mean_recall >= 0.85 else "yellow" if r.mean_recall >= 0.7 else "red"

        table.add_row(
            f"[{style}]{r.floor:.2f}[/{style}]" if style else f"{r.floor:.2f}",
            f"[{recall_style}]{r.mean_recall:.1%}[/{recall_style}]",
            f"{r.std_recall:.3f}",
            f"{r.min_recall:.0%}",
            f"{r.max_recall:.0%}",
            str(r.perfect_recall_count),
        )

    console.print(table)
    console.print(f"\n[bold green]Optimal floor: {best_floor.floor:.2f} (mean recall: {best_floor.mean_recall:.1%})[/bold green]")


def display_reliability_summary(reliability_map: dict[str, ToolReliability]) -> None:
    """Display summary of computed reliability profiles."""
    table = Table(title="Simulated Tool Reliability Profiles")
    table.add_column("Tool", width=26)
    table.add_column("Activations", width=12, justify="right")
    table.add_column("Useful Rate", width=12, justify="right")
    table.add_column("Error Rate", width=10, justify="right")
    table.add_column("Empty Rate", width=10, justify="right")

    # Sort by useful rate descending
    sorted_tools = sorted(
        reliability_map.values(),
        key=lambda r: r.useful_rate,
        reverse=True,
    )

    for rel in sorted_tools[:15]:
        useful_style = "green" if rel.useful_rate >= 0.7 else "yellow" if rel.useful_rate >= 0.4 else "red"

        table.add_row(
            rel.tool,
            f"{rel.total_activations:.1f}",
            f"[{useful_style}]{rel.useful_rate:.0%}[/{useful_style}]",
            f"{rel.error_rate:.0%}",
            f"{rel.empty_rate:.0%}",
        )

    console.print(table)


def analyze_by_ledger_size(
    decompositions: list[tuple[str, DecompositionResult, list[str]]],
    identity_vectors: np.ndarray,
    all_outcomes: list[GravityOutcome],
) -> list[tuple[int, float, float]]:
    """
    Analyze optimal floor by ledger size.
    Returns list of (ledger_size, optimal_floor, recall_at_optimal).
    """
    results = []
    sizes = [25, 50, 100, 150, 200]

    for size in sizes:
        # Take first N outcomes
        subset = all_outcomes[:size]
        reliability_map = compute_reliability(subset)

        # Find optimal floor for this subset
        floor_values = [0.2, 0.3, 0.4, 0.5, 0.6]
        sweep = run_floor_sweep(decompositions, identity_vectors, reliability_map, floor_values)
        best = max(sweep, key=lambda r: r.mean_recall)

        results.append((size, best.floor, best.mean_recall))

    return results


def display_ledger_size_analysis(results: list[tuple[int, float, float]]) -> None:
    """Display how optimal floor changes with ledger size."""
    table = Table(title="Optimal Floor by Ledger Size")
    table.add_column("Ledger Size", width=12, justify="right")
    table.add_column("Optimal Floor", width=14, justify="right")
    table.add_column("Mean Recall", width=12, justify="right")

    for size, floor, recall in results:
        table.add_row(str(size), f"{floor:.2f}", f"{recall:.1%}")

    console.print(table)


def create_plot(results: list[FloorSweepResult], output_path: Path) -> None:
    """Create matplotlib plot of floor vs recall."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not available, skipping plot[/yellow]")
        return

    floors = [r.floor for r in results]
    means = [r.mean_recall for r in results]
    stds = [r.std_recall for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(floors, means, yerr=stds, fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax.axhline(y=max(means), color='green', linestyle='--', alpha=0.5, label=f'Best: {max(means):.1%}')
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Current floor (0.3)')

    best_idx = means.index(max(means))
    best_floor = floors[best_idx]
    ax.axvline(x=best_floor, color='green', linestyle=':', alpha=0.7, label=f'Optimal ({best_floor:.2f})')

    ax.set_xlabel('Reliability Bias Floor', fontsize=12)
    ax.set_ylabel('Mean Recall', fontsize=12)
    ax.set_title('Adaptive Floor Exploration: Floor Value vs. Recall', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    console.print(f"[dim]Plot saved to {output_path}[/dim]")


def save_results(
    sweep_results: list[FloorSweepResult],
    reliability_map: dict[str, ToolReliability],
    ledger_size_analysis: list[tuple[int, float, float]],
) -> Path:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"adaptive_floor_{timestamp}.json"

    sweep_data = [
        {
            "floor": r.floor,
            "mean_recall": r.mean_recall,
            "std_recall": r.std_recall,
            "min_recall": r.min_recall,
            "max_recall": r.max_recall,
            "perfect_recall_count": r.perfect_recall_count,
        }
        for r in sweep_results
    ]

    best = max(sweep_results, key=lambda r: r.mean_recall)

    reliability_data = {
        name: {
            "total_activations": rel.total_activations,
            "error_rate": rel.error_rate,
            "empty_rate": rel.empty_rate,
            "useful_rate": rel.useful_rate,
        }
        for name, rel in reliability_map.items()
    }

    data = {
        "timestamp": timestamp,
        "simulation_params": {
            "n_synthetic_outcomes": N_SYNTHETIC_OUTCOMES,
            "decay_constant": DECAY_CONSTANT,
        },
        "optimal_floor": best.floor,
        "optimal_recall": best.mean_recall,
        "floor_sweep": sweep_data,
        "ledger_size_analysis": [
            {"size": size, "optimal_floor": floor, "recall": recall}
            for size, floor, recall in ledger_size_analysis
        ],
        "reliability_profiles": reliability_data,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Adaptive Floor Exploration")
    parser.add_argument(
        "--skip-decompose",
        action="store_true",
        help="Reuse cached decompositions (skip Claude API calls)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plot generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic ledger generation",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    console.print(Panel(
        "[bold]Adaptive Floor Exploration[/bold]\n"
        "Simulating reliability-biased gravity fields to find optimal floor",
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

    # Get decompositions for all test queries
    decompositions: list[tuple[str, DecompositionResult, list[str]]] = []
    decomps_for_ledger: list[tuple[str, DecompositionResult]] = []

    for i, test in enumerate(TEST_QUERIES):
        result = get_or_decompose(test.query, cache, skip_decompose=args.skip_decompose)
        if result is None:
            console.print(f"[yellow]Skipping query {i}: not in cache and --skip-decompose[/yellow]")
            continue

        # Embed the decomposition
        result = embed_decomposition(result, test.query, model=embedding_model)

        decompositions.append((test.query, result, test.expected_active))
        decomps_for_ledger.append((test.query, result))
        console.print(f"[dim]  [{i+1}/{len(TEST_QUERIES)}] {test.query[:50]}...[/dim]")

    # Save cache
    save_cache(cache)

    if not decompositions:
        console.print("[red]No decompositions available. Run without --skip-decompose first.[/red]")
        return

    console.print()

    # Generate synthetic ledger
    console.print(f"[dim]Generating {N_SYNTHETIC_OUTCOMES} synthetic ledger outcomes...[/dim]")
    outcomes = generate_synthetic_ledger(decomps_for_ledger, identity_vectors)

    # Compute reliability from full ledger
    console.print("[dim]Computing reliability profiles...[/dim]")
    reliability_map = compute_reliability(outcomes)

    # Display reliability summary
    display_reliability_summary(reliability_map)
    console.print()

    # Run floor sweep
    console.print("[dim]Running floor sweep (0.1 to 0.7)...[/dim]")
    floor_values = [round(0.1 + i * 0.05, 2) for i in range(13)]  # 0.10 to 0.70
    sweep_results = run_floor_sweep(decompositions, identity_vectors, reliability_map, floor_values)

    # Display sweep results
    display_floor_sweep(sweep_results)
    console.print()

    # Analyze by ledger size
    console.print("[dim]Analyzing optimal floor by ledger size...[/dim]")
    ledger_size_analysis = analyze_by_ledger_size(decompositions, identity_vectors, outcomes)
    display_ledger_size_analysis(ledger_size_analysis)

    # Create plot
    if not args.no_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = RESULTS_DIR / f"adaptive_floor_{timestamp}.png"
        create_plot(sweep_results, plot_path)

    # Save results
    output_path = save_results(sweep_results, reliability_map, ledger_size_analysis)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
