#!/usr/bin/env python3
"""
Gravity Model Phase 1 — Experiment Runner

Decomposes test queries via Claude, computes gravity fields, validates
activation patterns against expected results, and displays rich terminal output.

Usage:
    python run_experiment.py              # Run all 20 test queries
    python run_experiment.py --query 0    # Run a single query by index
    python run_experiment.py --interactive  # Enter your own queries
    python run_experiment.py --skip-decompose  # Reuse cached decompositions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from decompose import decompose, embed_decomposition
from fragments import DecompositionResult, Fragment, FragmentType
from gravity_field import GravityField, compute_gravity_field
from test_queries import TEST_QUERIES, TestQuery
from tool_identities import TOOL_NAMES, TOOLS, load_identity_vectors

console = Console()
RESULTS_DIR = Path(__file__).parent / "results"


def display_decomposition(query: str, result: DecompositionResult) -> None:
    """Display the fragment decomposition."""
    table = Table(title="Fragment Decomposition", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Text", style="white")
    table.add_column("Primary", style="bold yellow", width=8)

    for i, f in enumerate(result.fragments):
        is_primary = "  *" if i == result.primary_mass_index else ""
        table.add_row(str(i), f.type.value, f.text, is_primary)

    console.print(table)
    console.print(f"  [dim]Claude reasoning: {result.claude_reasoning}[/dim]")


def display_gravity_field(field: GravityField, decomposition: DecompositionResult) -> None:
    """Display the full pull matrix and activation scores."""
    n_fragments = len(decomposition.fragments)

    # Pull matrix heatmap
    table = Table(title="Pull Matrix (tool × fragment)", show_lines=True)
    table.add_column("Tool", style="white", width=28)

    for i, f in enumerate(decomposition.fragments):
        marker = "*" if i == decomposition.primary_mass_index else ""
        table.add_column(
            f"{f.text[:12]}{marker}",
            style="white",
            width=10,
            justify="right",
        )
    table.add_column("Query", style="white", width=8, justify="right")
    table.add_column("Composite", style="bold", width=10, justify="right")
    table.add_column("Status", width=8)

    activated_names = {t.name for t in field.activated}

    for tool_info in field.all_tools:
        pulls_str = []
        for p in tool_info.pulls:
            if p >= 0.4:
                pulls_str.append(f"[bold green]{p:.3f}[/bold green]")
            elif p >= 0.25:
                pulls_str.append(f"[yellow]{p:.3f}[/yellow]")
            elif p >= 0.15:
                pulls_str.append(f"[dim]{p:.3f}[/dim]")
            else:
                pulls_str.append(f"[dim red]{p:.3f}[/dim red]")

        # Query pull column
        qp = tool_info.query_pull
        if qp >= 0.4:
            qp_str = f"[bold green]{qp:.3f}[/bold green]"
        elif qp >= 0.25:
            qp_str = f"[yellow]{qp:.3f}[/yellow]"
        elif qp >= 0.15:
            qp_str = f"[dim]{qp:.3f}[/dim]"
        else:
            qp_str = f"[dim red]{qp:.3f}[/dim red]"

        composite_str = f"[bold]{tool_info.composite_score:.3f}[/bold]"
        status = "[bold green]FIRE[/bold green]" if tool_info.name in activated_names else "[dim]---[/dim]"

        name = tool_info.name
        if tool_info.is_meta:
            name = f"[dim italic]{name}[/dim italic]"

        table.add_row(name, *pulls_str, qp_str, composite_str, status)

    console.print(table)


def display_primary_mass(field: GravityField, decomposition: DecompositionResult) -> None:
    """Display primary mass identification comparison."""
    claude_frag = decomposition.fragments[field.claude_primary_index]
    centroid_frag = decomposition.fragments[field.centroid_primary_index]

    agree_icon = "[bold green]AGREE[/bold green]" if field.primary_agreement else "[bold red]DISAGREE[/bold red]"

    table = Table(title="Primary Mass Identification")
    table.add_column("Method", width=16)
    table.add_column("Fragment", width=30)
    table.add_column("Score", width=10, justify="right")

    table.add_row(
        "Claude",
        f"{claude_frag.text} ({claude_frag.type.value})",
        "---",
    )
    table.add_row(
        "Centroid",
        f"{centroid_frag.text} ({centroid_frag.type.value})",
        f"{field.centroid_similarities[field.centroid_primary_index]:.3f}",
    )

    console.print(table)
    console.print(f"  Agreement: {agree_icon}")

    # Show all centroid similarities
    sims_text = "  Centroid similarities: "
    for i, f in enumerate(decomposition.fragments):
        marker = " *" if i == field.centroid_primary_index else ""
        sims_text += f"{f.text}={field.centroid_similarities[i]:.3f}{marker}  "
    console.print(f"[dim]{sims_text}[/dim]")


def validate_query(
    field: GravityField,
    test: TestQuery,
) -> dict:
    """Validate activation patterns against expected results."""
    activated_names = {t.name for t in field.activated}

    # Check expected active
    correct_active = []
    missed_active = []
    for name in test.expected_active:
        if name in activated_names:
            correct_active.append(name)
        else:
            missed_active.append(name)

    # Check expected inactive
    correct_inactive = []
    false_active = []
    for name in test.expected_inactive:
        if name not in activated_names:
            correct_inactive.append(name)
        else:
            false_active.append(name)

    # Precision/recall for expected_active
    precision = len(correct_active) / len(activated_names) if activated_names else 0
    recall = len(correct_active) / len(test.expected_active) if test.expected_active else 1.0

    return {
        "correct_active": correct_active,
        "missed_active": missed_active,
        "correct_inactive": correct_inactive,
        "false_active": false_active,
        "precision": precision,
        "recall": recall,
        "total_activated": len(activated_names),
    }


def display_validation(validation: dict) -> None:
    """Display validation results."""
    table = Table(title="Validation")
    table.add_column("Metric", width=20)
    table.add_column("Value", width=50)

    recall = validation["recall"]
    recall_style = "green" if recall >= 0.8 else "yellow" if recall >= 0.5 else "red"
    table.add_row("Recall (expected active)", f"[{recall_style}]{recall:.0%}[/{recall_style}]")
    table.add_row("Total activated", str(validation["total_activated"]))

    if validation["missed_active"]:
        table.add_row(
            "[red]Missed (should fire)[/red]",
            ", ".join(validation["missed_active"]),
        )
    if validation["false_active"]:
        table.add_row(
            "[yellow]False active (shouldn't fire)[/yellow]",
            ", ".join(validation["false_active"]),
        )
    if validation["correct_active"]:
        table.add_row(
            "[green]Correct active[/green]",
            ", ".join(validation["correct_active"]),
        )

    console.print(table)


def run_single_query(
    query: str,
    identity_vectors: np.ndarray,
    embedding_model,
    test: TestQuery | None = None,
    min_tools: int = 3,
    max_tools: int = 10,
) -> dict:
    """Run the full gravity pipeline for a single query."""
    console.print(Panel(f"[bold]{query}[/bold]", title="Query", border_style="blue"))

    # Step 1: Decompose
    console.print("[dim]Decomposing via Claude...[/dim]")
    t0 = time.time()
    result = decompose(query)
    decompose_ms = (time.time() - t0) * 1000
    console.print(f"[dim]  Decomposition: {decompose_ms:.0f}ms[/dim]")

    # Step 2: Embed
    t0 = time.time()
    result = embed_decomposition(result, query, model=embedding_model)
    embed_ms = (time.time() - t0) * 1000
    console.print(f"[dim]  Embedding: {embed_ms:.0f}ms[/dim]")

    display_decomposition(query, result)

    # Step 3: Gravity field
    t0 = time.time()
    gfield = compute_gravity_field(
        identity_vectors,
        result,
        min_tools=min_tools,
        max_tools=max_tools,
    )
    gravity_ms = (time.time() - t0) * 1000
    console.print(f"[dim]  Gravity field: {gravity_ms:.0f}ms[/dim]")

    display_gravity_field(gfield, result)
    console.print(f"  [dim]Adaptive cutoff: {gfield.adaptive_cutoff:.4f} (gap at position {gfield.gap_position})[/dim]")
    display_primary_mass(gfield, result)

    # Step 4: Validate (if test query provided)
    validation = None
    if test:
        validation = validate_query(gfield, test)
        display_validation(validation)

    console.print()

    return {
        "query": query,
        "fragments": [
            {"type": f.type.value, "text": f.text}
            for f in result.fragments
        ],
        "primary_mass": {
            "claude": result.fragments[result.primary_mass_index].text,
            "centroid": result.fragments[gfield.centroid_primary_index].text,
            "agreement": gfield.primary_agreement,
        },
        "activated_tools": [
            {
                "name": t.name,
                "composite": round(t.composite_score, 4),
                "primary_pull": round(t.primary_pull, 4),
            }
            for t in gfield.activated
        ],
        "all_tools": [
            {
                "name": t.name,
                "composite": round(t.composite_score, 4),
                "primary_pull": round(t.primary_pull, 4),
            }
            for t in gfield.all_tools
        ],
        "timing_ms": {
            "decompose": round(decompose_ms),
            "embed": round(embed_ms),
            "gravity": round(gravity_ms),
        },
        "validation": validation,
    }


def display_summary(results: list[dict]) -> None:
    """Display summary statistics across all test queries."""
    console.print(Panel("[bold]Experiment Summary[/bold]", border_style="green"))

    validated = [r for r in results if r.get("validation")]
    if not validated:
        console.print("[dim]No validation data (no test queries run)[/dim]")
        return

    recalls = [r["validation"]["recall"] for r in validated]
    agreements = [r["primary_mass"]["agreement"] for r in validated]
    tool_counts = [r["validation"]["total_activated"] for r in validated]

    table = Table(title="Aggregate Statistics")
    table.add_column("Metric", width=30)
    table.add_column("Value", width=20, justify="right")

    table.add_row("Queries run", str(len(validated)))
    table.add_row("Mean recall", f"{np.mean(recalls):.0%}")
    table.add_row("Min recall", f"{np.min(recalls):.0%}")
    table.add_row("Perfect recall (100%)", f"{sum(1 for r in recalls if r >= 1.0)}/{len(validated)}")
    table.add_row("Primary mass agreement", f"{sum(agreements)}/{len(validated)} ({np.mean(agreements):.0%})")
    table.add_row("Mean tools activated", f"{np.mean(tool_counts):.1f}")
    table.add_row("Max tools activated", str(max(tool_counts)))

    console.print(table)

    # Per-query recall
    table2 = Table(title="Per-Query Results")
    table2.add_column("#", width=3)
    table2.add_column("Query", width=50)
    table2.add_column("Recall", width=8, justify="right")
    table2.add_column("PM Agree", width=9)
    table2.add_column("# Tools", width=8, justify="right")
    table2.add_column("Missed", width=30)

    for i, r in enumerate(validated):
        v = r["validation"]
        recall = v["recall"]
        recall_style = "green" if recall >= 0.8 else "yellow" if recall >= 0.5 else "red"
        agree = "[green]Y[/green]" if r["primary_mass"]["agreement"] else "[red]N[/red]"
        missed = ", ".join(v["missed_active"][:3]) if v["missed_active"] else "[green]---[/green]"

        table2.add_row(
            str(i),
            r["query"][:50],
            f"[{recall_style}]{recall:.0%}[/{recall_style}]",
            agree,
            str(v["total_activated"]),
            missed,
        )

    console.print(table2)

    # Tool activation frequency
    tool_freq: dict[str, int] = {}
    for r in validated:
        for t in r["activated_tools"]:
            tool_freq[t["name"]] = tool_freq.get(t["name"], 0) + 1

    table3 = Table(title="Tool Activation Frequency")
    table3.add_column("Tool", width=28)
    table3.add_column("Activated", width=10, justify="right")
    table3.add_column("Rate", width=8, justify="right")

    for name in sorted(tool_freq, key=tool_freq.get, reverse=True):
        rate = tool_freq[name] / len(validated)
        table3.add_row(name, str(tool_freq[name]), f"{rate:.0%}")

    console.print(table3)

    # False activations (tools firing when they shouldn't)
    false_freq: dict[str, int] = {}
    for r in validated:
        for name in r["validation"]["false_active"]:
            false_freq[name] = false_freq.get(name, 0) + 1

    if false_freq:
        table4 = Table(title="False Activations (tools firing when shouldn't)")
        table4.add_column("Tool", width=28)
        table4.add_column("False fires", width=10, justify="right")
        for name in sorted(false_freq, key=false_freq.get, reverse=True):
            table4.add_row(name, str(false_freq[name]))
        console.print(table4)


def main():
    parser = argparse.ArgumentParser(description="Gravity Model Phase 1 Experiment")
    parser.add_argument("--query", type=int, help="Run a single test query by index (0-19)")
    parser.add_argument("--interactive", action="store_true", help="Enter custom queries")
    parser.add_argument("--min-tools", type=int, default=3, help="Minimum tools to activate")
    parser.add_argument("--max-tools", type=int, default=10, help="Maximum tools to activate")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Gravity Model — Phase 1 Experiment[/bold]\n"
        f"Min tools: {args.min_tools}  |  Max tools: {args.max_tools}  |  Adaptive gap detection",
        border_style="blue",
    ))

    # Load embedding model
    console.print("[dim]Loading embedding model (all-mpnet-base-v2)...[/dim]")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Load/compute identity vectors
    console.print("[dim]Loading tool identity vectors...[/dim]")
    identity_vectors = load_identity_vectors(model=embedding_model)
    console.print(f"[dim]  Identity vectors: {identity_vectors.shape}[/dim]")

    results = []

    if args.interactive:
        console.print("\n[bold]Interactive mode[/bold] — enter queries (empty line to quit)\n")
        while True:
            query = console.input("[bold blue]Query:[/bold blue] ").strip()
            if not query:
                break
            result = run_single_query(
                query, identity_vectors, embedding_model,
                min_tools=args.min_tools,
                max_tools=args.max_tools,
            )
            results.append(result)

    elif args.query is not None:
        idx = args.query
        if idx < 0 or idx >= len(TEST_QUERIES):
            console.print(f"[red]Query index must be 0-{len(TEST_QUERIES)-1}[/red]")
            sys.exit(1)
        test = TEST_QUERIES[idx]
        result = run_single_query(
            test.query, identity_vectors, embedding_model, test=test,
            composite_threshold=args.threshold,
            primary_floor=args.floor,
        )
        results.append(result)

    else:
        # Run all test queries
        for i, test in enumerate(TEST_QUERIES):
            console.print(f"\n[bold]═══ Query {i}/{len(TEST_QUERIES)-1} ═══[/bold]")
            result = run_single_query(
                test.query, identity_vectors, embedding_model, test=test,
                min_tools=args.min_tools,
                max_tools=args.max_tools,
            )
            results.append(result)

        display_summary(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"experiment_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    "min_tools": args.min_tools,
                    "max_tools": args.max_tools,
                    "model": "all-mpnet-base-v2",
                },
                "results": results,
            },
            f,
            indent=2,
        )
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
