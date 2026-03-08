"""Evaluate a trained theme classifier on the hold-out test set.

Loads the trained model + thresholds, runs inference on test entries,
and produces per-label and aggregate metrics with quality gate checks.

Usage:
  python evaluate_theme_classifier.py \
    --model-dir ./theme-classifier-mpnet-v1 \
    --data theme_training_data.jsonl \
    --splits theme_splits.json \
    --label-index theme_label_index.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from train_theme_classifier import (
    ClassificationHead,
    ThemeClassifier,
    ThemeDataset,
    ThemeExample,
    collate_fn,
    compute_metrics,
    evaluate,
    load_examples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval-theme")


def main():
    parser = argparse.ArgumentParser(description="Evaluate theme classifier on test set")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--data", required=True, help="Path to theme_training_data.jsonl")
    parser.add_argument("--splits", required=True, help="Path to theme_splits.json")
    parser.add_argument("--label-index", required=True, help="Path to theme_label_index.json")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load label index
    with open(args.label_index) as f:
        label_index = json.load(f)
    n_labels = len(label_index)
    label_names = sorted(label_index.keys(), key=lambda k: label_index[k])
    logger.info(f"Labels: {n_labels}")

    # Load training config
    config_path = model_dir / "training_config.json"
    with config_path.open() as f:
        config = json.load(f)
    input_dim = config.get("input_dim", 768)
    hidden_dim = config.get("hidden_dim", 256)
    logger.info(f"Model: {config.get('base_model', 'unknown')} "
                f"dim={input_dim} hidden={hidden_dim}")

    # Load thresholds
    threshold_path = model_dir / "theme_thresholds.json"
    if threshold_path.exists():
        with threshold_path.open() as f:
            threshold_dict = json.load(f)
        thresholds = np.array([threshold_dict.get(label, 0.5) for label in label_names])
        logger.info(f"Loaded per-label thresholds (range: {thresholds.min():.2f}-{thresholds.max():.2f})")
    else:
        thresholds = np.full(n_labels, 0.5)
        logger.info("No thresholds found, using default 0.5")

    # Load model
    encoder = SentenceTransformer(str(model_dir / "model"))
    head = ClassificationHead(
        input_dim=input_dim, hidden_dim=hidden_dim,
        output_dim=n_labels, dropout=0.1,
    )
    head.load_state_dict(
        torch.load(str(model_dir / "classification_head.pt"), map_location="cpu", weights_only=True)
    )
    model = ThemeClassifier(encoder, head)
    model.head.eval()

    # Load data
    all_examples = load_examples(args.data, label_index)
    with open(args.splits) as f:
        splits = json.load(f)

    test_ids = set(splits["test_ids"])
    test_examples = [all_examples[eid] for eid in sorted(test_ids) if eid in all_examples]
    logger.info(f"Test set: {len(test_examples)} entries")

    # Evaluate
    test_loader = DataLoader(
        ThemeDataset(test_examples), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    test_metrics, test_preds, test_labels = evaluate(
        model, test_loader, loss_fn, "cpu",
    )

    # Metrics with default threshold (0.5)
    metrics_default = compute_metrics(test_preds, test_labels, None)
    # Metrics with optimized thresholds
    metrics_optimized = compute_metrics(test_preds, test_labels, thresholds)

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE METRICS")
    logger.info("=" * 60)

    logger.info(f"\n{'Metric':25s} {'Default (0.5)':>15s} {'Optimized':>15s}")
    logger.info("-" * 58)
    for key in ["macro_f1", "micro_f1", "weighted_f1", "hamming_loss", "mAP", "exact_match"]:
        logger.info(f"  {key:25s} {metrics_default[key]:15.4f} {metrics_optimized[key]:15.4f}")

    # Per-label metrics
    binary_preds = (test_preds >= thresholds).astype(int)
    per_label_f1 = f1_score(test_labels, binary_preds, average=None, zero_division=0)
    per_label_prec = precision_score(test_labels, binary_preds, average=None, zero_division=0)
    per_label_rec = recall_score(test_labels, binary_preds, average=None, zero_division=0)

    logger.info("\n" + "=" * 60)
    logger.info("PER-LABEL METRICS (all labels, optimized thresholds)")
    logger.info("=" * 60)
    logger.info(f"\n{'Label':40s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Sup':>5s} {'Thr':>5s}")
    logger.info("-" * 72)

    labels_with_support = []
    for i, label in enumerate(label_names):
        support = int(test_labels[:, i].sum())
        logger.info(
            f"  {label:40s} {per_label_prec[i]:6.3f} {per_label_rec[i]:6.3f} "
            f"{per_label_f1[i]:6.3f} {support:5d} {thresholds[i]:5.2f}"
        )
        if support >= 5:
            labels_with_support.append(i)

    # Summary for labels with support >= 5
    if labels_with_support:
        logger.info(f"\nLabels with support >= 5: {len(labels_with_support)}")
        subset_f1 = [per_label_f1[i] for i in labels_with_support]
        logger.info(f"  Mean F1: {np.mean(subset_f1):.4f}")
        logger.info(f"  Min F1:  {np.min(subset_f1):.4f}")
        zero_f1_labels = [label_names[i] for i in labels_with_support if per_label_f1[i] == 0.0]
        if zero_f1_labels:
            logger.warning(f"  Labels with F1=0: {zero_f1_labels}")

    # Quality gates
    logger.info("\n" + "=" * 60)
    logger.info("QUALITY GATES")
    logger.info("=" * 60)

    gates = {
        "macro_f1_threshold": 0.30,
        "micro_f1_threshold": 0.45,
        "macro_f1_passed": metrics_optimized["macro_f1"] > 0.30,
        "micro_f1_passed": metrics_optimized["micro_f1"] > 0.45,
    }

    # Check no label with support >= 10 has F1 = 0.0
    labels_sup10_zero_f1 = [
        label_names[i] for i in range(n_labels)
        if int(test_labels[:, i].sum()) >= 10 and per_label_f1[i] == 0.0
    ]
    gates["no_sup10_zero_f1"] = len(labels_sup10_zero_f1) == 0

    all_passed = all([gates["macro_f1_passed"], gates["micro_f1_passed"], gates["no_sup10_zero_f1"]])

    logger.info(f"  Macro F1 > 0.30: {'PASS' if gates['macro_f1_passed'] else 'FAIL'} "
                f"({metrics_optimized['macro_f1']:.4f})")
    logger.info(f"  Micro F1 > 0.45: {'PASS' if gates['micro_f1_passed'] else 'FAIL'} "
                f"({metrics_optimized['micro_f1']:.4f})")
    logger.info(f"  No sup>=10 F1=0: {'PASS' if gates['no_sup10_zero_f1'] else 'FAIL'}")

    if labels_sup10_zero_f1:
        logger.warning(f"  Labels with support>=10 and F1=0: {labels_sup10_zero_f1}")

    if all_passed:
        logger.info("\nALL QUALITY GATES PASSED")
    else:
        logger.warning("\nQUALITY GATES FAILED")

    # Save report
    eval_report = {
        "test_count": len(test_examples),
        "n_labels": n_labels,
        "test_metrics_default_threshold": metrics_default,
        "test_metrics_optimized_threshold": metrics_optimized,
        "quality_gates": gates,
        "per_label": {
            label_names[i]: {
                "precision": float(per_label_prec[i]),
                "recall": float(per_label_rec[i]),
                "f1": float(per_label_f1[i]),
                "support": int(test_labels[:, i].sum()),
                "threshold": float(thresholds[i]),
            }
            for i in range(n_labels)
        },
    }

    eval_path = model_dir / "eval_report.json"
    with eval_path.open("w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"\nSaved evaluation report to {eval_path}")

    # Print sample predictions for qualitative review
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE PREDICTIONS (first 5 test entries)")
    logger.info("=" * 60)

    for idx in range(min(5, len(test_examples))):
        ex = test_examples[idx]
        true_labels = [label_names[i] for i in range(n_labels) if ex.labels[i] > 0.5]
        pred_labels = [label_names[i] for i in range(n_labels)
                       if test_preds[idx, i] >= thresholds[i]]
        # Also show top-5 probabilities
        top5_idx = np.argsort(-test_preds[idx])[:5]
        top5 = [(label_names[i], f"{test_preds[idx, i]:.3f}") for i in top5_idx]

        logger.info(f"\n  Entry: {ex.entry_id}")
        logger.info(f"  True:  {true_labels}")
        logger.info(f"  Pred:  {pred_labels}")
        logger.info(f"  Top-5: {top5}")


if __name__ == "__main__":
    main()
