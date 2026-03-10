"""Training observability dashboard with Rich terminal UI + Weights & Biases.

Provides a beautiful live terminal display during training and persistent
cloud-based metrics tracking via W&B.

Usage:
    from training_dashboard import TrainingDashboard

    dashboard = TrainingDashboard(
        project="corpus-mcp",
        run_name="decision-classifier-v1",
        config={"epochs": 30, "batch_size": 16, ...},
        use_wandb=True,
    )

    with dashboard:
        for epoch in range(epochs):
            dashboard.start_epoch(epoch, total_epochs)
            for batch in dataloader:
                # train...
                dashboard.log_batch(batch_idx, len(dataloader), loss)
            dashboard.log_epoch(epoch, {"loss": 0.5, "f1": 0.8, ...})

    # Or without context manager:
    dashboard.finish()
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Rich imports
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Optional W&B
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─── Sparkline ────────────────────────────────────────────────────────────────


SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 40) -> str:
    """Generate a sparkline string from values."""
    if not values:
        return " " * width

    # Take last `width` values
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

    # Pad to width
    spark = "".join(result)
    if len(spark) < width:
        spark = " " * (width - len(spark)) + spark

    return spark


# ─── Mini ASCII Chart ─────────────────────────────────────────────────────────


def mini_chart(
    values: list[float],
    width: int = 50,
    height: int = 6,
    label: str = "",
) -> list[str]:
    """Generate a mini ASCII chart."""
    if not values:
        return [" " * width for _ in range(height)]

    # Take last width values
    data = list(values)[-width:]

    min_val = min(data)
    max_val = max(data)
    val_range = max_val - min_val if max_val > min_val else 1.0

    # Build the chart
    lines = []

    # Y-axis labels
    y_labels = [
        f"{max_val:.3f}",
        f"{(max_val + min_val) / 2:.3f}",
        f"{min_val:.3f}",
    ]
    label_width = max(len(l) for l in y_labels)

    for row in range(height):
        # Threshold for this row (top = max, bottom = min)
        threshold = max_val - (row / (height - 1)) * val_range

        line_chars = []
        for i, v in enumerate(data):
            if v >= threshold:
                # Use different chars for visual interest
                if row == 0 or (row > 0 and data[i] < max_val - ((row - 1) / (height - 1)) * val_range):
                    line_chars.append("█")
                else:
                    line_chars.append("█")
            else:
                line_chars.append(" ")

        # Add y-axis label on certain rows
        if row == 0:
            prefix = f"{max_val:>{label_width}.3f} │"
        elif row == height - 1:
            prefix = f"{min_val:>{label_width}.3f} │"
        elif row == height // 2:
            mid = (max_val + min_val) / 2
            prefix = f"{mid:>{label_width}.3f} │"
        else:
            prefix = " " * label_width + " │"

        lines.append(prefix + "".join(line_chars))

    # X-axis
    x_axis = " " * label_width + " └" + "─" * len(data)
    lines.append(x_axis)

    # Label
    if label:
        label_line = " " * label_width + "  " + label.center(len(data))
        lines.append(label_line)

    return lines


# ─── Dashboard State ──────────────────────────────────────────────────────────


@dataclass
class PhaseState:
    name: str
    total_epochs: int
    current_epoch: int = 0
    start_time: float = field(default_factory=time.time)


@dataclass
class MetricHistory:
    train_loss: deque = field(default_factory=lambda: deque(maxlen=100))
    val_loss: deque = field(default_factory=lambda: deque(maxlen=100))
    train_f1: deque = field(default_factory=lambda: deque(maxlen=100))
    val_f1: deque = field(default_factory=lambda: deque(maxlen=100))
    lr: deque = field(default_factory=lambda: deque(maxlen=100))

    # Best tracking
    best_val_f1: float = 0.0
    best_epoch: int = 0
    best_threshold: float = 0.5


# ─── Status File (for external monitoring) ────────────────────────────────────

STATUS_FILE = "training_status.json"


def write_status_file(
    status_path: str,
    task_type: str,
    run_name: str,
    phase: "PhaseState | None",
    history: MetricHistory,
    current_batch: int,
    total_batches: int,
    batch_loss: float,
    patience_counter: int,
    patience_max: int,
    config: dict,
    status: str = "training",
) -> None:
    """Write current training status to a JSON file for external monitoring."""
    import json

    status_data = {
        "status": status,
        "task_type": task_type,
        "run_name": run_name,
        "timestamp": time.time(),
        "phase": {
            "name": phase.name if phase else "initializing",
            "current_epoch": phase.current_epoch if phase else 0,
            "total_epochs": phase.total_epochs if phase else 0,
            "start_time": phase.start_time if phase else time.time(),
        },
        "batch": {
            "current": current_batch,
            "total": total_batches,
            "loss": batch_loss,
        },
        "metrics": {
            "train_loss": list(history.train_loss),
            "val_loss": list(history.val_loss),
            "val_f1": list(history.val_f1),
            "lr": list(history.lr),
            "best_val_f1": history.best_val_f1,
            "best_epoch": history.best_epoch,
            "best_threshold": history.best_threshold,
        },
        "patience": {
            "current": patience_counter,
            "max": patience_max,
        },
        "config": config,
    }

    try:
        with open(status_path, "w") as f:
            json.dump(status_data, f)
    except Exception:
        pass  # Don't crash training if status write fails


# ─── Dashboard ────────────────────────────────────────────────────────────────


class TrainingDashboard:
    """Rich terminal dashboard with optional W&B integration."""

    def __init__(
        self,
        project: str = "corpus-mcp",
        run_name: str | None = None,
        task_type: str = "classifier",
        config: dict[str, Any] | None = None,
        use_wandb: bool = True,
        wandb_mode: str = "online",  # "online", "offline", "disabled"
        status_file: str | None = STATUS_FILE,  # Write status for external monitoring
        headless: bool = False,  # If True, skip Rich UI but still write status
    ):
        self.project = project
        self.run_name = run_name or f"{task_type}-{int(time.time())}"
        self.task_type = task_type
        self.config = config or {}
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.status_file = status_file
        self.headless = headless

        # State
        self.phase: PhaseState | None = None
        self.history = MetricHistory()
        self.current_batch = 0
        self.total_batches = 0
        self.batch_loss = 0.0
        self.gpu_memory: str = "N/A"
        self.patience_counter = 0
        self.patience_max = config.get("patience", 10) if config else 10

        # Rich console and progress
        self.console = Console()
        self.live: Live | None = None

        # Epoch progress
        self.epoch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.epoch_task_id = None

        # Batch progress
        self.batch_progress = Progress(
            TextColumn("  [dim]Batch"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[loss]:.4f}"),
            console=self.console,
        )
        self.batch_task_id = None

        # W&B
        self.wandb_run = None
        if self.use_wandb:
            self._init_wandb(wandb_mode)

    def _init_wandb(self, mode: str) -> None:
        """Initialize Weights & Biases run."""
        try:
            self.wandb_run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
                mode=mode,
                reinit=True,
            )
            self.console.print(
                f"[green]✓[/green] W&B run initialized: [link={self.wandb_run.url}]{self.wandb_run.url}[/link]"
            )
        except Exception as e:
            self.console.print(f"[yellow]⚠[/yellow] W&B init failed: {e}")
            self.use_wandb = False

    def _get_gpu_memory(self) -> str:
        """Get GPU memory usage string."""
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                pct = (allocated / total) * 100
                return f"{allocated:.1f}GB/{total:.1f}GB ({pct:.0f}%)"
        except Exception:
            pass
        return "N/A"

    def _build_header(self) -> Panel:
        """Build the header panel."""
        phase_name = self.phase.name if self.phase else "Initializing"
        phase_color = "cyan" if "FROZEN" in phase_name.upper() else "magenta"

        title = Text()
        title.append(f"🚀 {self.task_type.upper()} TRAINING", style="bold white")
        title.append(" │ ", style="dim")
        title.append(self.run_name, style="italic")

        phase_text = Text()
        phase_text.append("Phase: ", style="dim")
        phase_text.append(phase_name, style=f"bold {phase_color}")

        if self.phase:
            epoch_info = f"  Epoch {self.phase.current_epoch + 1}/{self.phase.total_epochs}"
            phase_text.append(epoch_info, style="white")

        return Panel(
            Group(title, phase_text),
            border_style="blue",
            padding=(0, 2),
        )

    def _build_progress_panel(self) -> Panel:
        """Build the progress bars panel."""
        return Panel(
            Group(self.epoch_progress, self.batch_progress),
            title="[bold]Progress",
            border_style="green",
            padding=(0, 1),
        )

    def _build_metrics_table(self) -> Table:
        """Build the metrics table."""
        table = Table(
            title="Metrics",
            show_header=True,
            header_style="bold",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Train", justify="right")
        table.add_column("Val", justify="right")
        table.add_column("Best", justify="right", style="green")
        table.add_column("Trend", justify="center")

        # Loss
        train_loss = self.history.train_loss[-1] if self.history.train_loss else 0
        val_loss = self.history.val_loss[-1] if self.history.val_loss else 0
        loss_spark = sparkline(list(self.history.val_loss), width=20)
        table.add_row(
            "Loss",
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            "-",
            loss_spark,
        )

        # F1
        train_f1 = self.history.train_f1[-1] if self.history.train_f1 else 0
        val_f1 = self.history.val_f1[-1] if self.history.val_f1 else 0
        best_info = f"{self.history.best_val_f1:.4f} (ep {self.history.best_epoch + 1})"
        f1_spark = sparkline(list(self.history.val_f1), width=20)
        table.add_row(
            "F1",
            f"{train_f1:.4f}",
            f"{val_f1:.4f}",
            best_info,
            f1_spark,
        )

        return table

    def _build_charts_panel(self) -> Panel:
        """Build the loss/F1 charts panel."""
        # Loss chart
        loss_values = list(self.history.val_loss)
        loss_chart = mini_chart(loss_values, width=45, height=5, label="Loss")

        # F1 chart
        f1_values = list(self.history.val_f1)
        f1_chart = mini_chart(f1_values, width=45, height=5, label="F1")

        # Side by side
        lines = []
        max_lines = max(len(loss_chart), len(f1_chart))
        for i in range(max_lines):
            left = loss_chart[i] if i < len(loss_chart) else " " * 45
            right = f1_chart[i] if i < len(f1_chart) else " " * 45
            lines.append(f"{left}   {right}")

        return Panel(
            "\n".join(lines),
            title="[bold]Training Curves",
            border_style="yellow",
        )

    def _build_status_bar(self) -> Panel:
        """Build the bottom status bar."""
        parts = []

        # GPU
        gpu_mem = self._get_gpu_memory()
        if gpu_mem != "N/A":
            parts.append(f"🔥 GPU: {gpu_mem}")

        # Learning rate
        if self.history.lr:
            lr = self.history.lr[-1]
            parts.append(f"LR: {lr:.2e}")

        # Patience
        parts.append(f"Patience: {self.patience_counter}/{self.patience_max}")

        # Threshold (for binary classifiers)
        if self.history.best_threshold:
            parts.append(f"Thresh: {self.history.best_threshold:.3f}")

        # W&B status
        if self.use_wandb and self.wandb_run:
            parts.append(f"[dim]W&B: {self.wandb_run.name}[/dim]")

        status_text = "    ".join(parts)
        return Panel(
            Text(status_text, justify="center"),
            border_style="dim",
            padding=(0, 1),
        )

    def _build_display(self) -> Group:
        """Build the full dashboard display."""
        return Group(
            self._build_header(),
            self._build_progress_panel(),
            self._build_metrics_table(),
            self._build_charts_panel(),
            self._build_status_bar(),
        )

    def __enter__(self) -> "TrainingDashboard":
        """Start the live display."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the live display."""
        self.finish()

    def start(self) -> None:
        """Start the live dashboard."""
        if self.headless:
            # Headless mode: no Rich UI, just status file updates
            return

        self.console.print()
        self.console.print("[bold green]Starting training dashboard...[/bold green]")
        self.console.print()

        self.live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self.live.start()

    def start_phase(self, name: str, total_epochs: int) -> None:
        """Start a new training phase (e.g., 'Frozen Encoder', 'Fine-tuning')."""
        self.phase = PhaseState(name=name, total_epochs=total_epochs)

        # Reset epoch progress
        if self.epoch_task_id is not None:
            self.epoch_progress.remove_task(self.epoch_task_id)

        self.epoch_task_id = self.epoch_progress.add_task(
            f"[{name}]",
            total=total_epochs,
        )

        if self.live:
            self.live.update(self._build_display())

    def start_epoch(self, epoch: int, total_batches: int) -> None:
        """Start a new epoch."""
        if self.phase:
            self.phase.current_epoch = epoch

        self.total_batches = total_batches
        self.current_batch = 0

        # Update epoch progress
        if self.epoch_task_id is not None:
            self.epoch_progress.update(self.epoch_task_id, completed=epoch)

        # Reset batch progress
        if self.batch_task_id is not None:
            self.batch_progress.remove_task(self.batch_task_id)

        self.batch_task_id = self.batch_progress.add_task(
            "batch",
            total=total_batches,
            loss=0.0,
        )

        if self.live:
            self.live.update(self._build_display())

    def log_batch(self, batch_idx: int, loss: float) -> None:
        """Log batch-level metrics."""
        self.current_batch = batch_idx
        self.batch_loss = loss

        if self.batch_task_id is not None:
            self.batch_progress.update(
                self.batch_task_id,
                completed=batch_idx + 1,
                loss=loss,
            )

        # Don't update full display every batch - too slow
        # The progress bar updates itself

    def log_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Log epoch-level metrics."""
        # Extract metrics
        train_loss = metrics.get("train_loss", 0)
        val_loss = metrics.get("val_loss", metrics.get("loss", 0))
        train_f1 = metrics.get("train_f1", 0)
        val_f1 = metrics.get("val_f1", metrics.get("f1", 0))
        lr = metrics.get("lr", 0)
        threshold = metrics.get("threshold", None)

        # Update history
        self.history.train_loss.append(train_loss)
        self.history.val_loss.append(val_loss)
        self.history.train_f1.append(train_f1)
        self.history.val_f1.append(val_f1)
        if lr:
            self.history.lr.append(lr)

        # Track best
        if is_best:
            self.history.best_val_f1 = val_f1
            self.history.best_epoch = epoch
            if threshold:
                self.history.best_threshold = threshold
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Update epoch progress
        if self.epoch_task_id is not None:
            self.epoch_progress.update(self.epoch_task_id, completed=epoch + 1)

        # Log to W&B
        if self.use_wandb and self.wandb_run:
            wandb_metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/f1": train_f1,
                "val/f1": val_f1,
                "best/f1": self.history.best_val_f1,
                "patience": self.patience_counter,
            }
            if lr:
                wandb_metrics["lr"] = lr
            if threshold:
                wandb_metrics["threshold"] = threshold

            # Add any extra metrics
            for k, v in metrics.items():
                if k not in ["train_loss", "val_loss", "train_f1", "val_f1", "lr", "threshold", "loss", "f1"]:
                    wandb_metrics[k] = v

            wandb.log(wandb_metrics)

        # Update display
        if self.live:
            self.live.update(self._build_display())

        # Write status file for external monitoring
        if self.status_file:
            write_status_file(
                self.status_file,
                self.task_type,
                self.run_name,
                self.phase,
                self.history,
                self.current_batch,
                self.total_batches,
                self.batch_loss,
                self.patience_counter,
                self.patience_max,
                self.config,
                status="training",
            )

    def log_cv_fold(self, fold_idx: int, metrics: dict[str, float]) -> None:
        """Log cross-validation fold results."""
        if self.use_wandb and self.wandb_run:
            wandb.log({f"cv/fold_{fold_idx}/{k}": v for k, v in metrics.items()})

    def log_test_results(self, metrics: dict[str, float]) -> None:
        """Log final test set results."""
        if self.use_wandb and self.wandb_run:
            wandb.log({f"test/{k}": v for k, v in metrics.items()})

            # Also log as summary
            for k, v in metrics.items():
                wandb.run.summary[f"test_{k}"] = v

    def finish(self, final_message: str | None = None) -> None:
        """Finish training and clean up."""
        # Stop live display
        if self.live:
            self.live.stop()
            self.live = None

        # Write final status
        if self.status_file:
            write_status_file(
                self.status_file,
                self.task_type,
                self.run_name,
                self.phase,
                self.history,
                self.current_batch,
                self.total_batches,
                self.batch_loss,
                self.patience_counter,
                self.patience_max,
                self.config,
                status="completed",
            )

        # Print final summary
        self.console.print()
        self.console.print("[bold green]═" * 60)
        self.console.print("[bold green]Training Complete!")
        self.console.print(f"  Best F1: {self.history.best_val_f1:.4f} (epoch {self.history.best_epoch + 1})")
        if self.history.best_threshold:
            self.console.print(f"  Best Threshold: {self.history.best_threshold:.3f}")
        self.console.print("[bold green]═" * 60)

        if final_message:
            self.console.print(final_message)

        # Finish W&B
        if self.use_wandb and self.wandb_run:
            wandb.finish()
            self.console.print(f"\n[dim]W&B run saved: {self.wandb_run.url}[/dim]")

        self.console.print()


# ─── Simplified API ───────────────────────────────────────────────────────────


def create_dashboard(
    task: str,
    config: dict[str, Any],
    use_wandb: bool = True,
) -> TrainingDashboard:
    """Create a training dashboard with sensible defaults.

    Args:
        task: Task name like "decision-classifier" or "entity-classifier"
        config: Training configuration dict
        use_wandb: Whether to use Weights & Biases

    Returns:
        TrainingDashboard instance
    """
    # Generate run name from config
    base_model = config.get("base_model", "unknown")
    model_short = base_model.split("/")[-1].split("-")[0]
    timestamp = int(time.time())
    run_name = f"{task}-{model_short}-{timestamp}"

    return TrainingDashboard(
        project="corpus-mcp-training",
        run_name=run_name,
        task_type=task,
        config=config,
        use_wandb=use_wandb,
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────


def demo():
    """Demo the dashboard with fake training data."""
    import random

    config = {
        "base_model": "all-mpnet-base-v2",
        "epochs_frozen": 5,
        "epochs_unfrozen": 10,
        "batch_size": 16,
        "patience": 10,
    }

    dashboard = create_dashboard("demo-classifier", config, use_wandb=False)

    with dashboard:
        # Phase 1: Frozen
        dashboard.start_phase("Frozen Encoder", total_epochs=5)
        for epoch in range(5):
            dashboard.start_epoch(epoch, total_batches=50)

            for batch in range(50):
                loss = 0.8 - epoch * 0.1 + random.uniform(-0.05, 0.05)
                dashboard.log_batch(batch, loss)
                time.sleep(0.02)

            f1 = 0.5 + epoch * 0.08 + random.uniform(-0.02, 0.02)
            dashboard.log_epoch(
                epoch,
                {
                    "train_loss": loss,
                    "val_loss": loss + 0.05,
                    "val_f1": f1,
                    "lr": 1e-3,
                },
                is_best=(f1 > dashboard.history.best_val_f1),
            )

        # Phase 2: Unfrozen
        dashboard.start_phase("Fine-tuning", total_epochs=10)
        for epoch in range(10):
            dashboard.start_epoch(epoch, total_batches=50)

            for batch in range(50):
                loss = 0.4 - epoch * 0.02 + random.uniform(-0.02, 0.02)
                dashboard.log_batch(batch, loss)
                time.sleep(0.02)

            f1 = 0.75 + epoch * 0.015 + random.uniform(-0.01, 0.01)
            dashboard.log_epoch(
                epoch,
                {
                    "train_loss": loss,
                    "val_loss": loss + 0.03,
                    "val_f1": f1,
                    "lr": 5e-5 * (0.9 ** epoch),
                    "threshold": 0.45 + epoch * 0.01,
                },
                is_best=(f1 > dashboard.history.best_val_f1),
            )


if __name__ == "__main__":
    demo()
