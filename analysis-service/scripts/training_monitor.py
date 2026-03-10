#!/usr/bin/env python3
"""Training monitor - watch training progress from anywhere.

Reads the training_status.json file written by the training dashboard
and displays a live Rich UI. Can monitor local or remote training.

Usage:
  # Monitor local training
  python training_monitor.py

  # Monitor remote training (via SSH)
  python training_monitor.py --remote hewes@10.0.10.232 --path ~/training

  # One-shot status check (no live updates)
  python training_monitor.py --once
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

# Sparkline characters
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 30) -> str:
    """Generate a sparkline string from values."""
    if not values:
        return " " * width

    values = list(values)[-width:]
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val

    if val_range == 0:
        return SPARK_CHARS[3] * len(values)

    result = []
    for v in values:
        normalized = (v - min_val) / val_range
        idx = int(normalized * (len(SPARK_CHARS) - 1))
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))
        result.append(SPARK_CHARS[idx])

    spark = "".join(result)
    if len(spark) < width:
        spark = " " * (width - len(spark)) + spark

    return spark


def read_status_local(path: str) -> dict | None:
    """Read status from a local file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_status_remote(host: str, path: str) -> dict | None:
    """Read status from a remote file via SSH."""
    try:
        result = subprocess.run(
            ["ssh", host, f"cat {path}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m}m"


def build_display(status: dict | None, remote: str | None) -> Panel:
    """Build the monitor display."""
    if status is None:
        return Panel(
            "[yellow]No training status found.[/yellow]\n\n"
            "[dim]Waiting for training to start...[/dim]",
            title="[bold]Training Monitor",
            border_style="yellow",
        )

    # Extract data
    task_type = status.get("task_type", "unknown")
    run_name = status.get("run_name", "unknown")
    train_status = status.get("status", "unknown")
    timestamp = status.get("timestamp", 0)
    age = time.time() - timestamp

    phase = status.get("phase", {})
    phase_name = phase.get("name", "unknown")
    current_epoch = phase.get("current_epoch", 0)
    total_epochs = phase.get("total_epochs", 0)
    phase_start = phase.get("start_time", time.time())
    phase_elapsed = time.time() - phase_start

    batch = status.get("batch", {})
    current_batch = batch.get("current", 0)
    total_batches = batch.get("total", 0)
    batch_loss = batch.get("loss", 0)

    metrics = status.get("metrics", {})
    val_f1_history = metrics.get("val_f1", [])
    val_loss_history = metrics.get("val_loss", [])
    best_f1 = metrics.get("best_val_f1", 0)
    best_epoch = metrics.get("best_epoch", 0)
    best_thresh = metrics.get("best_threshold", 0)
    lr_history = metrics.get("lr", [])

    patience = status.get("patience", {})
    patience_current = patience.get("current", 0)
    patience_max = patience.get("max", 10)

    # Status color
    if train_status == "completed":
        status_color = "green"
        status_icon = "✓"
    elif train_status == "training":
        status_color = "cyan" if age < 30 else "yellow"
        status_icon = "●" if age < 30 else "?"
    else:
        status_color = "red"
        status_icon = "✗"

    # Build content
    lines = []

    # Header
    location = f"[dim]({remote})[/dim]" if remote else "[dim](local)[/dim]"
    header = Text()
    header.append(f"{status_icon} ", style=status_color)
    header.append(task_type.upper(), style="bold white")
    header.append(f" {location}\n", style="dim")
    header.append(f"Run: {run_name}\n", style="dim")
    header.append(f"Status: ", style="dim")
    header.append(train_status.upper(), style=f"bold {status_color}")
    if age > 5:
        header.append(f" (updated {format_duration(age)} ago)", style="dim yellow")
    lines.append(header)
    lines.append("")

    # Phase progress
    if total_epochs > 0:
        epoch_pct = (current_epoch + 1) / total_epochs
        epoch_bar = "━" * int(epoch_pct * 30) + "╺" + "─" * (29 - int(epoch_pct * 30))
        lines.append(f"[bold]{phase_name}[/bold]")
        lines.append(f"  Epoch {current_epoch + 1}/{total_epochs}  [{epoch_pct*100:.0f}%]  {format_duration(phase_elapsed)}")
        lines.append(f"  [cyan]{epoch_bar}[/cyan]")

    # Batch progress
    if total_batches > 0:
        batch_pct = (current_batch + 1) / total_batches
        batch_bar = "━" * int(batch_pct * 20) + "─" * (20 - int(batch_pct * 20))
        lines.append(f"  [dim]Batch {current_batch + 1}/{total_batches}  loss={batch_loss:.4f}[/dim]")
        lines.append(f"  [dim]{batch_bar}[/dim]")

    lines.append("")

    # Metrics table
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Metric")
    table.add_column("Current", justify="right")
    table.add_column("Best", justify="right", style="green")
    table.add_column("Trend", justify="left")

    # F1
    current_f1 = val_f1_history[-1] if val_f1_history else 0
    f1_spark = sparkline(val_f1_history, width=25)
    table.add_row("F1", f"{current_f1:.4f}", f"{best_f1:.4f} (ep {best_epoch + 1})", f1_spark)

    # Loss
    current_loss = val_loss_history[-1] if val_loss_history else 0
    loss_spark = sparkline(val_loss_history, width=25)
    table.add_row("Loss", f"{current_loss:.4f}", "-", loss_spark)

    lines.append(table)
    lines.append("")

    # Footer stats
    footer_parts = []
    if lr_history:
        footer_parts.append(f"LR: {lr_history[-1]:.2e}")
    footer_parts.append(f"Patience: {patience_current}/{patience_max}")
    if best_thresh:
        footer_parts.append(f"Thresh: {best_thresh:.3f}")

    lines.append("[dim]" + "  │  ".join(footer_parts) + "[/dim]")

    # Combine into panel
    from rich.console import Group
    content = Group(*[line if hasattr(line, '__rich_console__') else Text(str(line)) for line in lines])

    border_style = status_color
    return Panel(
        content,
        title="[bold]Training Monitor",
        border_style=border_style,
        padding=(1, 2),
    )


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument(
        "--remote", "-r",
        help="Remote host (e.g., hewes@10.0.10.232)",
    )
    parser.add_argument(
        "--path", "-p",
        default="~/training",
        help="Path to training directory (default: ~/training)",
    )
    parser.add_argument(
        "--file", "-f",
        default="training_status.json",
        help="Status file name (default: training_status.json)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Show status once and exit (no live updates)",
    )
    args = parser.parse_args()

    console = Console()
    status_path = f"{args.path}/{args.file}"

    def get_status():
        if args.remote:
            return read_status_remote(args.remote, status_path)
        else:
            return read_status_local(status_path)

    if args.once:
        # One-shot mode
        status = get_status()
        console.print(build_display(status, args.remote))
        return

    # Live mode
    console.print(f"[dim]Monitoring: {args.remote or 'local'}:{status_path}[/dim]")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")

    try:
        with Live(
            build_display(None, args.remote),
            console=console,
            refresh_per_second=1,
        ) as live:
            while True:
                status = get_status()
                live.update(build_display(status, args.remote))
                time.sleep(args.interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


if __name__ == "__main__":
    main()
