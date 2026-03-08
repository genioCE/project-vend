"""Evaluate fine-tuned state classifier against Claude baseline and local rules.

Runs the trained model on hold-out test entries and produces per-dimension
metrics (MAE, Pearson r, Spearman ρ). Optionally compares against the local
rule engine on the same test set.

Usage:
  python evaluate_classifier.py \
    --model ./state-classifier \
    --data training_data.jsonl \
    --splits splits.json \
    [--compare-local] \
    [--output eval_comparison.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

ALL_DIMENSIONS = (
    "valence",
    "activation",
    "agency",
    "certainty",
    "relational_openness",
    "self_trust",
    "time_orientation",
    "integration",
)


# ─── Model (must match train_state_classifier.py) ────────────────────────────


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_finetuned_model(model_dir: str):
    """Load the fine-tuned encoder + regression head."""
    model_path = Path(model_dir)
    config_path = model_path / "training_config.json"

    hidden_dim = 256
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
            hidden_dim = config.get("hidden_dim", 256)

    encoder = SentenceTransformer(str(model_path / "model"))
    head = RegressionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=8)
    head.load_state_dict(
        torch.load(str(model_path / "regression_head.pt"), map_location="cpu", weights_only=True)
    )
    head.eval()
    return encoder, head


def predict_finetuned(encoder, head, texts: list[str]) -> np.ndarray:
    """Run finetuned model inference on a list of texts. Returns (N, 8) array."""
    with torch.no_grad():
        embeddings = encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=False,
            normalize_embeddings=True,
        )
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        logits = head(embeddings)
        scores = torch.tanh(logits).numpy()
    return scores


# ─── Local Rule Engine ────────────────────────────────────────────────────────


def predict_local(texts: list[str]) -> np.ndarray:
    """Run local state engine on a list of texts. Returns (N, 8) array."""
    # Import from parent package
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.state_engine import analyze_state

    all_scores = []
    for text in texts:
        result = analyze_state(text)
        scores = [result["dimensions"][dim]["score"] for dim in ALL_DIMENSIONS]
        all_scores.append(scores)
    return np.array(all_scores)


# ─── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute per-dimension metrics between predictions and Claude labels."""
    metrics = {}
    mae_list = []

    for i, dim in enumerate(ALL_DIMENSIONS):
        dim_preds = preds[:, i]
        dim_labels = labels[:, i]

        mae = float(np.mean(np.abs(dim_preds - dim_labels)))
        mse = float(np.mean((dim_preds - dim_labels) ** 2))
        bias = float(np.mean(dim_preds - dim_labels))  # positive = over-scoring

        # Pearson correlation
        if np.std(dim_preds) > 1e-8 and np.std(dim_labels) > 1e-8:
            pearson_r, pearson_p = stats.pearsonr(dim_preds, dim_labels)
        else:
            pearson_r, pearson_p = 0.0, 1.0

        # Spearman rank correlation
        if np.std(dim_preds) > 1e-8 and np.std(dim_labels) > 1e-8:
            spearman_r, spearman_p = stats.spearmanr(dim_preds, dim_labels)
        else:
            spearman_r, spearman_p = 0.0, 1.0

        metrics[dim] = {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "bias": round(bias, 4),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 6),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 6),
            "pred_mean": round(float(np.mean(dim_preds)), 4),
            "pred_std": round(float(np.std(dim_preds)), 4),
            "label_mean": round(float(np.mean(dim_labels)), 4),
            "label_std": round(float(np.std(dim_labels)), 4),
        }
        mae_list.append(mae)

    metrics["_summary"] = {
        "mean_mae": round(float(np.mean(mae_list)), 4),
        "max_mae": round(float(np.max(mae_list)), 4),
        "worst_dimension": ALL_DIMENSIONS[int(np.argmax(mae_list))],
    }
    return metrics


# ─── Quality Gate Check ──────────────────────────────────────────────────────


def check_quality_gates(metrics: dict) -> dict:
    """Check if the model passes all quality gates for deployment."""
    gates = {
        "mean_mae_threshold": 0.18,
        "max_dim_mae_threshold": 0.25,
        "min_pearson_threshold": 0.70,
    }

    mean_mae = metrics["_summary"]["mean_mae"]
    gates["mean_mae_value"] = mean_mae
    gates["mean_mae_passed"] = mean_mae < 0.18

    dim_maes = {dim: metrics[dim]["mae"] for dim in ALL_DIMENSIONS}
    max_dim_mae = max(dim_maes.values())
    gates["max_dim_mae_value"] = max_dim_mae
    gates["max_dim_mae_passed"] = max_dim_mae < 0.25
    gates["failing_mae_dims"] = [d for d, m in dim_maes.items() if m >= 0.25]

    pearsons = {dim: metrics[dim]["pearson_r"] for dim in ALL_DIMENSIONS}
    min_pearson = min(pearsons.values())
    gates["min_pearson_value"] = min_pearson
    gates["min_pearson_passed"] = min_pearson > 0.70
    gates["failing_pearson_dims"] = [d for d, p in pearsons.items() if p <= 0.70]

    gates["all_passed"] = all([
        gates["mean_mae_passed"],
        gates["max_dim_mae_passed"],
        gates["min_pearson_passed"],
    ])
    return gates


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned state classifier")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--data", required=True, help="Path to training_data.jsonl")
    parser.add_argument("--splits", required=True, help="Path to splits.json")
    parser.add_argument("--compare-local", action="store_true", help="Also evaluate local rule engine")
    parser.add_argument("--output", default="eval_comparison.json", help="Output JSON file")
    parser.add_argument("--sample", type=int, default=0, help="Show N sample comparisons")
    args = parser.parse_args()

    # Load test data
    with open(args.splits) as f:
        splits = json.load(f)
    test_ids = set(splits["test_ids"])

    examples = {}
    with open(args.data) as f:
        for line in f:
            rec = json.loads(line)
            if rec["entry_id"] in test_ids:
                examples[rec["entry_id"]] = rec

    if not examples:
        logger.error("No test examples found!")
        sys.exit(1)

    # Sort for reproducibility
    sorted_ids = sorted(examples.keys())
    texts = [examples[eid]["text"] for eid in sorted_ids]
    labels = np.array([
        [examples[eid]["labels"][dim] for dim in ALL_DIMENSIONS]
        for eid in sorted_ids
    ])

    logger.info(f"Evaluating on {len(sorted_ids)} hold-out test entries")

    # ─── Fine-tuned model evaluation ──────────────────────────────────────

    logger.info("Loading fine-tuned model...")
    encoder, head = load_finetuned_model(args.model)

    logger.info("Running fine-tuned model inference...")
    ft_preds = predict_finetuned(encoder, head, texts)
    ft_metrics = compute_metrics(ft_preds, labels)
    ft_gates = check_quality_gates(ft_metrics)

    logger.info("")
    logger.info("=" * 72)
    logger.info("FINE-TUNED MODEL vs CLAUDE (hold-out test set)")
    logger.info("=" * 72)
    logger.info(
        f"{'Dimension':25s} {'MAE':>7s} {'Bias':>7s} {'Pearson':>8s} "
        f"{'Spearman':>9s} {'Pred μ':>7s} {'Claude μ':>9s}"
    )
    logger.info("-" * 72)
    for dim in ALL_DIMENSIONS:
        m = ft_metrics[dim]
        logger.info(
            f"{dim:25s} {m['mae']:7.4f} {m['bias']:+7.4f} {m['pearson_r']:8.4f} "
            f"{m['spearman_r']:9.4f} {m['pred_mean']:+7.4f} {m['label_mean']:+7.4f}"
        )
    s = ft_metrics["_summary"]
    logger.info("-" * 72)
    logger.info(f"{'MEAN':25s} {s['mean_mae']:7.4f}")
    logger.info(f"  Worst dimension: {s['worst_dimension']} (MAE={s['max_mae']:.4f})")

    logger.info("")
    if ft_gates["all_passed"]:
        logger.info("QUALITY GATES: ALL PASSED")
    else:
        logger.warning("QUALITY GATES: FAILED")
        if not ft_gates["mean_mae_passed"]:
            logger.warning(f"  Mean MAE {ft_gates['mean_mae_value']:.4f} >= 0.18")
        if not ft_gates["max_dim_mae_passed"]:
            logger.warning(f"  Dims with MAE >= 0.25: {ft_gates['failing_mae_dims']}")
        if not ft_gates["min_pearson_passed"]:
            logger.warning(f"  Dims with Pearson <= 0.70: {ft_gates['failing_pearson_dims']}")

    result = {
        "test_count": len(sorted_ids),
        "finetuned": {
            "metrics": ft_metrics,
            "quality_gates": ft_gates,
        },
    }

    # ─── Local rule engine comparison ─────────────────────────────────────

    if args.compare_local:
        logger.info("")
        logger.info("Running local rule engine on same test entries...")
        local_preds = predict_local(texts)
        local_metrics = compute_metrics(local_preds, labels)

        logger.info("")
        logger.info("=" * 72)
        logger.info("LOCAL RULE ENGINE vs CLAUDE (same test set)")
        logger.info("=" * 72)
        logger.info(
            f"{'Dimension':25s} {'MAE':>7s} {'Bias':>7s} {'Pearson':>8s} "
            f"{'Spearman':>9s} {'Pred μ':>7s} {'Claude μ':>9s}"
        )
        logger.info("-" * 72)
        for dim in ALL_DIMENSIONS:
            m = local_metrics[dim]
            logger.info(
                f"{dim:25s} {m['mae']:7.4f} {m['bias']:+7.4f} {m['pearson_r']:8.4f} "
                f"{m['spearman_r']:9.4f} {m['pred_mean']:+7.4f} {m['label_mean']:+7.4f}"
            )
        ls = local_metrics["_summary"]
        logger.info("-" * 72)
        logger.info(f"{'MEAN':25s} {ls['mean_mae']:7.4f}")

        # ─── Improvement table ────────────────────────────────────────────

        logger.info("")
        logger.info("=" * 72)
        logger.info("IMPROVEMENT: Fine-tuned over Local Rules")
        logger.info("=" * 72)
        logger.info(f"{'Dimension':25s} {'Local MAE':>10s} {'FT MAE':>10s} {'Δ MAE':>10s} {'Improvement':>12s}")
        logger.info("-" * 72)
        for dim in ALL_DIMENSIONS:
            local_mae = local_metrics[dim]["mae"]
            ft_mae = ft_metrics[dim]["mae"]
            delta = local_mae - ft_mae
            pct = (delta / local_mae * 100) if local_mae > 0 else 0
            logger.info(
                f"{dim:25s} {local_mae:10.4f} {ft_mae:10.4f} {delta:+10.4f} {pct:+11.1f}%"
            )
        local_mean = local_metrics["_summary"]["mean_mae"]
        ft_mean = ft_metrics["_summary"]["mean_mae"]
        delta_mean = local_mean - ft_mean
        pct_mean = (delta_mean / local_mean * 100) if local_mean > 0 else 0
        logger.info("-" * 72)
        logger.info(
            f"{'MEAN':25s} {local_mean:10.4f} {ft_mean:10.4f} {delta_mean:+10.4f} {pct_mean:+11.1f}%"
        )

        result["local"] = {"metrics": local_metrics}

    # ─── Sample comparisons ───────────────────────────────────────────────

    if args.sample > 0:
        n_show = min(args.sample, len(sorted_ids))
        logger.info("")
        logger.info("=" * 72)
        logger.info(f"SAMPLE COMPARISONS (first {n_show} test entries)")
        logger.info("=" * 72)

        for idx in range(n_show):
            eid = sorted_ids[idx]
            logger.info(f"\n  Entry: {eid}")
            logger.info(f"  {'Dimension':25s} {'Claude':>8s} {'FT':>8s} {'Δ':>8s}", )
            logger.info(f"  {'-' * 50}")
            for i, dim in enumerate(ALL_DIMENSIONS):
                claude_score = labels[idx, i]
                ft_score = ft_preds[idx, i]
                delta = ft_score - claude_score
                logger.info(f"  {dim:25s} {claude_score:+8.4f} {ft_score:+8.4f} {delta:+8.4f}")

    # ─── Save results ─────────────────────────────────────────────────────

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nSaved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
