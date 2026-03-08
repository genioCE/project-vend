"""Train a fine-tuned state label classifier on Claude-labeled journal data.

Architecture: transformer encoder → shared MLP → 8 regression heads (tanh)
Training: two-phase (frozen encoder, then unfrozen with discriminative LR)

Supports multiple base models:
  - all-mpnet-base-v2 (default, 768-dim, similarity-optimized)
  - microsoft/deberta-v3-base (768-dim, NLU-optimized — better for affect)
  - Any HuggingFace model compatible with sentence-transformers

Usage:
  python train_state_classifier.py \
    --data training_data.jsonl \
    --splits splits.json \
    --output ./state-classifier \
    [--base-model microsoft/deberta-v3-base] \
    [--epochs-frozen 8] [--epochs-unfrozen 30] [--batch-size 8] [--device cuda]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

# Known sentence-transformer models (can be loaded directly)
_SENTENCE_TRANSFORMER_MODELS = {
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-mpnet-base-v2",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

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


def build_encoder(model_name: str, device: str) -> tuple[SentenceTransformer, int]:
    """Build a SentenceTransformer encoder from a model name.

    For known sentence-transformer models, loads directly.
    For HuggingFace models (e.g. microsoft/deberta-v3-base), wraps
    in a Transformer + Pooling pipeline.

    Returns (encoder, embedding_dim).
    """
    if model_name in _SENTENCE_TRANSFORMER_MODELS:
        encoder = SentenceTransformer(model_name, device=device)
        embed_dim = encoder.get_sentence_embedding_dimension()
        logger.info(f"Loaded SentenceTransformer '{model_name}' (dim={embed_dim})")
        return encoder, embed_dim

    # Generic HuggingFace model — wrap as SentenceTransformer
    from sentence_transformers import models as st_models

    word_model = st_models.Transformer(model_name, max_seq_length=512)
    pooling = st_models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    encoder = SentenceTransformer(modules=[word_model, pooling], device=device)
    # Force float32 — some HF models (DeBERTa) default to fp16 which causes
    # NaN gradients during fine-tuning
    encoder = encoder.float()
    embed_dim = encoder.get_sentence_embedding_dimension()
    logger.info(f"Built SentenceTransformer from '{model_name}' (dim={embed_dim})")
    return encoder, embed_dim


# ─── Dataset ─────────────────────────────────────────────────────────────────


@dataclass
class TrainingExample:
    entry_id: str
    text: str
    labels: list[float]  # 8 scores in [-1, 1]


class StateDataset(Dataset):
    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.text, torch.tensor(ex.labels, dtype=torch.float32)


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.stack(labels)


# ─── Model ───────────────────────────────────────────────────────────────────


class RegressionHead(nn.Module):
    """Shared MLP head: Linear(768→H) → LayerNorm → GELU → Dropout → Linear(H→8)."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateClassifier(nn.Module):
    """Full model: sentence-transformer encoder + regression head."""

    def __init__(self, encoder: SentenceTransformer, head: RegressionHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def encode_texts_no_grad(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using sentence-transformer .encode() — no gradients.

        Used during frozen-encoder training and evaluation.
        """
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return torch.tensor(embeddings, dtype=torch.float32)

    def encode_texts_with_grad(self, texts: list[str], device: str) -> torch.Tensor:
        """Encode texts through the encoder's forward pass WITH gradient tracking.

        Tokenizes → transformer forward → mean pooling → L2 normalize.
        This is the critical path for fine-tuning the encoder weights.
        """
        features = self.encoder.tokenize(texts)
        features = {k: v.to(device) for k, v in features.items()}

        # Forward through transformer module (encoder[0]) — keeps gradients
        trans_output = self.encoder[0](features)

        # Forward through pooling module (encoder[1])
        pooled = self.encoder[1](trans_output)

        # Get sentence embeddings and normalize
        embeddings = pooled["sentence_embedding"]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # Ensure float32 — some models (DeBERTa) compute in fp16
        return embeddings.float()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through regression head with tanh activation."""
        logits = self.head(embeddings)
        return torch.tanh(logits)


# ─── Training ────────────────────────────────────────────────────────────────


def train_epoch(
    model: StateClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    encoder_frozen: bool = True,
    grad_accum_steps: int = 1,
    scheduler=None,
) -> float:
    model.head.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (texts, labels) in enumerate(dataloader):
        labels = labels.to(device)

        if encoder_frozen:
            # Fast path: use .encode() which is optimized but has no gradients
            with torch.no_grad():
                embeddings = model.encode_texts_no_grad(texts).to(device)
        else:
            # Grad path: tokenize → forward → pool with gradient tracking
            embeddings = model.encode_texts_with_grad(texts, device)

        predictions = model.forward(embeddings)
        loss = loss_fn(predictions, labels)

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        # Step optimizer every grad_accum_steps batches
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Gradient clipping to prevent exploding gradients during fine-tuning
            if not encoder_frozen:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: StateClassifier,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> dict:
    model.head.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for texts, labels in dataloader:
            labels = labels.to(device)
            embeddings = model.encode_texts_no_grad(texts).to(device)
            predictions = model.forward(embeddings)

            loss = loss_fn(predictions, labels)
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Per-dimension metrics
    metrics = {"loss": total_loss / max(n_batches, 1)}
    mae_per_dim = {}
    for i, dim in enumerate(ALL_DIMENSIONS):
        dim_preds = preds[:, i]
        dim_labels = labels[:, i]
        mae = float(np.mean(np.abs(dim_preds - dim_labels)))
        mse = float(np.mean((dim_preds - dim_labels) ** 2))

        # Pearson correlation
        if np.std(dim_preds) > 0 and np.std(dim_labels) > 0:
            pearson = float(np.corrcoef(dim_preds, dim_labels)[0, 1])
        else:
            pearson = 0.0

        mae_per_dim[dim] = mae
        metrics[f"{dim}_mae"] = mae
        metrics[f"{dim}_mse"] = mse
        metrics[f"{dim}_pearson"] = pearson

    metrics["mean_mae"] = float(np.mean(list(mae_per_dim.values())))
    metrics["mean_pearson"] = float(np.mean([
        metrics[f"{dim}_pearson"] for dim in ALL_DIMENSIONS
    ]))
    return metrics


# ─── LR Scheduler ────────────────────────────────────────────────────────────


def build_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup for warmup_steps, then cosine decay to 0."""
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ─── Cross-Validation ────────────────────────────────────────────────────────


def run_cv_fold(
    fold_idx: int,
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample],
    config: TrainConfig,
) -> dict:
    logger.info(f"  Fold {fold_idx}: {len(train_examples)} train, {len(val_examples)} val")

    encoder, embed_dim = build_encoder(config.base_model, config.device)
    head = RegressionHead(input_dim=embed_dim, hidden_dim=config.hidden_dim, output_dim=8, dropout=config.dropout)
    model = StateClassifier(encoder, head)

    if config.device == "cuda":
        head = head.to("cuda")

    train_loader = DataLoader(
        StateDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        StateDataset(val_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    loss_fn = nn.SmoothL1Loss()

    # Phase A: Frozen encoder — only train the MLP head
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr_head_frozen, weight_decay=0.01)

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs_frozen):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, config.device,
            encoder_frozen=True, grad_accum_steps=config.grad_accum_steps,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, config.device)

        if val_metrics["mean_mae"] < best_val_mae:
            best_val_mae = val_metrics["mean_mae"]
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == config.epochs_frozen - 1:
            logger.info(
                f"    [Frozen] Epoch {epoch+1}/{config.epochs_frozen}: "
                f"train_loss={train_loss:.4f} val_mae={val_metrics['mean_mae']:.4f}"
            )

        if patience_counter >= config.patience:
            logger.info(f"    Early stopping at epoch {epoch+1} (frozen phase)")
            break

    # Phase B: Unfrozen encoder — fine-tune everything with discriminative LR
    for param in encoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": config.lr_encoder},
            {"params": head.parameters(), "lr": config.lr_head_unfrozen},
        ],
        weight_decay=0.01,
    )

    scheduler = None
    if config.use_scheduler:
        steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
        total_steps = steps_per_epoch * config.epochs_unfrozen
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = build_cosine_scheduler(optimizer, warmup_steps, total_steps)

    best_val_mae = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(config.epochs_unfrozen):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, config.device,
            encoder_frozen=False, grad_accum_steps=config.grad_accum_steps,
            scheduler=scheduler,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, config.device)

        if val_metrics["mean_mae"] < best_val_mae:
            best_val_mae = val_metrics["mean_mae"]
            patience_counter = 0
            best_state = {
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
                "metrics": val_metrics,
            }
        else:
            patience_counter += 1

        if epoch % 3 == 0 or epoch == config.epochs_unfrozen - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"    [Unfrozen] Epoch {epoch+1}/{config.epochs_unfrozen}: "
                f"train_loss={train_loss:.4f} val_mae={val_metrics['mean_mae']:.4f} "
                f"val_r={val_metrics['mean_pearson']:.4f} lr={cur_lr:.2e}"
            )

        if patience_counter >= config.patience:
            logger.info(f"    Early stopping at epoch {epoch+1} (unfrozen phase)")
            break

    return best_state["metrics"] if best_state else val_metrics


# ─── Full Training ───────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    base_model: str = "all-mpnet-base-v2"
    epochs_frozen: int = 8
    epochs_unfrozen: int = 30
    batch_size: int = 8
    grad_accum_steps: int = 4  # effective batch = 8 * 4 = 32
    lr_head_frozen: float = 1e-3
    lr_encoder: float = 5e-5
    lr_head_unfrozen: float = 5e-4
    hidden_dim: int = 256
    dropout: float = 0.1
    patience: int = 10
    use_scheduler: bool = True
    device: str = "cpu"


def load_examples(data_path: str) -> dict[str, TrainingExample]:
    """Load all examples from JSONL file, keyed by entry_id."""
    examples = {}
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            labels = [rec["labels"][dim] for dim in ALL_DIMENSIONS]
            examples[rec["entry_id"]] = TrainingExample(
                entry_id=rec["entry_id"],
                text=rec["text"],
                labels=labels,
            )
    return examples


def train_final_model(
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample],
    config: TrainConfig,
    output_dir: Path,
) -> dict:
    """Train the final model with validation-based early stopping and save best weights."""
    logger.info(f"Training final model on {len(train_examples)} examples "
                f"(val: {len(val_examples)}) with base model '{config.base_model}'...")

    encoder, embed_dim = build_encoder(config.base_model, config.device)
    head = RegressionHead(input_dim=embed_dim, hidden_dim=config.hidden_dim, output_dim=8, dropout=config.dropout)
    model = StateClassifier(encoder, head)

    if config.device == "cuda":
        head = head.to("cuda")

    train_loader = DataLoader(
        StateDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        StateDataset(val_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    loss_fn = nn.SmoothL1Loss()

    # Phase A: Frozen encoder — train MLP head only
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr_head_frozen, weight_decay=0.01)

    best_val_mae = float("inf")
    for epoch in range(config.epochs_frozen):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, config.device,
            encoder_frozen=True, grad_accum_steps=config.grad_accum_steps,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, config.device)
        if val_metrics["mean_mae"] < best_val_mae:
            best_val_mae = val_metrics["mean_mae"]

        if epoch % 2 == 0 or epoch == config.epochs_frozen - 1:
            logger.info(
                f"  [Final/Frozen] Epoch {epoch+1}: loss={train_loss:.4f} "
                f"val_mae={val_metrics['mean_mae']:.4f}"
            )

    # Phase B: Unfrozen encoder — fine-tune with gradients, cosine LR, early stopping
    for param in encoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": config.lr_encoder},
            {"params": head.parameters(), "lr": config.lr_head_unfrozen},
        ],
        weight_decay=0.01,
    )

    scheduler = None
    if config.use_scheduler:
        steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
        total_steps = steps_per_epoch * config.epochs_unfrozen
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = build_cosine_scheduler(optimizer, warmup_steps, total_steps)

    best_val_mae = float("inf")
    best_encoder_state = None
    best_head_state = None
    patience_counter = 0

    for epoch in range(config.epochs_unfrozen):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, config.device,
            encoder_frozen=False, grad_accum_steps=config.grad_accum_steps,
            scheduler=scheduler,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, config.device)

        if val_metrics["mean_mae"] < best_val_mae:
            best_val_mae = val_metrics["mean_mae"]
            patience_counter = 0
            # Save best checkpoint in memory
            best_head_state = copy.deepcopy(head.state_dict())
            # For encoder, we need to save its internal model state
            best_encoder_state = copy.deepcopy(encoder[0].auto_model.state_dict())
            logger.info(
                f"  [Final/Unfrozen] Epoch {epoch+1}: NEW BEST "
                f"val_mae={val_metrics['mean_mae']:.4f} "
                f"val_r={val_metrics['mean_pearson']:.4f}"
            )
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == config.epochs_unfrozen - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  [Final/Unfrozen] Epoch {epoch+1}: loss={train_loss:.4f} "
                f"val_mae={val_metrics['mean_mae']:.4f} "
                f"val_r={val_metrics['mean_pearson']:.4f} lr={cur_lr:.2e}"
            )

        if patience_counter >= config.patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (best val_mae={best_val_mae:.4f})")
            break

    # Restore best checkpoint
    if best_encoder_state is not None:
        encoder[0].auto_model.load_state_dict(best_encoder_state)
        head.load_state_dict(best_head_state)
        logger.info(f"  Restored best checkpoint (val_mae={best_val_mae:.4f})")

    # Save model
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the fine-tuned encoder
    encoder.save(str(model_dir))
    logger.info(f"  Saved encoder to {model_dir}")

    # Save regression head
    head_path = output_dir / "regression_head.pt"
    torch.save(head.cpu().state_dict(), str(head_path))
    logger.info(f"  Saved regression head to {head_path}")

    # Save training config
    config_path = output_dir / "training_config.json"
    # Determine short model name for versioning
    model_short = config.base_model.split("/")[-1].lower()
    config_dict = {
        "dimensions": list(ALL_DIMENSIONS),
        "input_dim": embed_dim,
        "hidden_dim": config.hidden_dim,
        "output_dim": 8,
        "base_model": config.base_model,
        "prompt_version": "state-label-finetuned-v1",
        "model_version": f"finetuned-{model_short}-v1",
        "epochs_frozen": config.epochs_frozen,
        "epochs_unfrozen": config.epochs_unfrozen,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "effective_batch_size": config.batch_size * config.grad_accum_steps,
        "lr_encoder": config.lr_encoder,
        "lr_head_frozen": config.lr_head_frozen,
        "lr_head_unfrozen": config.lr_head_unfrozen,
        "dropout": config.dropout,
        "loss": "SmoothL1Loss",
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "best_val_mae": best_val_mae,
    }
    with config_path.open("w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"  Saved config to {config_path}")

    return config_dict


def main():
    parser = argparse.ArgumentParser(description="Train state label classifier")
    parser.add_argument("--data", required=True, help="Path to training_data.jsonl")
    parser.add_argument("--splits", required=True, help="Path to splits.json")
    parser.add_argument("--output", default="./state-classifier", help="Output directory")
    parser.add_argument(
        "--base-model",
        default="all-mpnet-base-v2",
        help="Base encoder model (sentence-transformer or HuggingFace model name, "
             "e.g. 'microsoft/deberta-v3-base', 'all-mpnet-base-v2')",
    )
    parser.add_argument("--epochs-frozen", type=int, default=8)
    parser.add_argument("--epochs-unfrozen", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-encoder", type=float, default=5e-5, help="Encoder learning rate")
    parser.add_argument("--lr-head-unfrozen", type=float, default=5e-4, help="Head LR during unfrozen phase")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation, train final only")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping — run all epochs")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable cosine LR scheduler")
    args = parser.parse_args()

    config = TrainConfig(
        base_model=args.base_model,
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr_encoder=args.lr_encoder,
        lr_head_unfrozen=args.lr_head_unfrozen,
        device=args.device,
        patience=999 if args.no_early_stop else 10,
        use_scheduler=not args.no_scheduler,
    )

    eff_batch = config.batch_size * config.grad_accum_steps
    logger.info(f"Device: {config.device}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Config: frozen={config.epochs_frozen}, unfrozen={config.epochs_unfrozen}, "
                f"batch={config.batch_size}x{config.grad_accum_steps}={eff_batch}, "
                f"hidden={config.hidden_dim}, dropout={config.dropout}, "
                f"lr_enc={config.lr_encoder}")

    # Load data
    all_examples = load_examples(args.data)
    logger.info(f"Loaded {len(all_examples)} examples")

    with open(args.splits) as f:
        splits = json.load(f)

    test_ids = set(splits["test_ids"])
    train_ids = splits["train_ids"]

    test_examples = [all_examples[eid] for eid in test_ids if eid in all_examples]
    train_examples = [all_examples[eid] for eid in train_ids if eid in all_examples]

    logger.info(f"Train: {len(train_examples)}, Test: {len(test_examples)}")

    # Cross-validation
    if not args.skip_cv:
        logger.info("=" * 60)
        logger.info("Running 5-fold cross-validation...")
        cv_results = []

        for fold_idx in range(splits["n_folds"]):
            fold_val_ids = set(splits["folds"][str(fold_idx)])
            fold_train = [ex for ex in train_examples if ex.entry_id not in fold_val_ids]
            fold_val = [ex for ex in train_examples if ex.entry_id in fold_val_ids]

            fold_metrics = run_cv_fold(fold_idx, fold_train, fold_val, config)
            cv_results.append(fold_metrics)

        # Aggregate CV results
        logger.info("=" * 60)
        logger.info("Cross-validation results:")
        logger.info(f"{'Dimension':25s} {'Mean MAE':>10s} {'Std MAE':>10s} {'Mean Pearson':>12s}")
        logger.info("-" * 60)

        for dim in ALL_DIMENSIONS:
            maes = [r[f"{dim}_mae"] for r in cv_results]
            pearsons = [r[f"{dim}_pearson"] for r in cv_results]
            logger.info(
                f"{dim:25s} {np.mean(maes):10.4f} {np.std(maes):10.4f} {np.mean(pearsons):12.4f}"
            )

        mean_maes = [r["mean_mae"] for r in cv_results]
        logger.info(f"{'OVERALL':25s} {np.mean(mean_maes):10.4f} {np.std(mean_maes):10.4f}")

    # Train final model — use small holdout (5%) for early stopping, rest for training
    logger.info("=" * 60)
    # Deterministic 5% val split: take every 20th example
    final_val = [ex for i, ex in enumerate(train_examples) if i % 20 == 0]
    final_train = [ex for i, ex in enumerate(train_examples) if i % 20 != 0]
    logger.info(f"Final model split: {len(final_train)} train, {len(final_val)} val (5%)")

    output_dir = Path(args.output)
    start_time = time.time()
    train_final_model(final_train, final_val, config, output_dir)
    elapsed = time.time() - start_time
    logger.info(f"Final model training completed in {elapsed:.1f}s")

    # Evaluate on hold-out test set
    logger.info("=" * 60)
    logger.info("Evaluating on hold-out test set...")

    encoder = SentenceTransformer(str(output_dir / "model"))
    saved_embed_dim = encoder.get_sentence_embedding_dimension()
    head = RegressionHead(input_dim=saved_embed_dim, hidden_dim=config.hidden_dim, output_dim=8, dropout=config.dropout)
    head.load_state_dict(torch.load(str(output_dir / "regression_head.pt"), map_location="cpu", weights_only=True))
    model = StateClassifier(encoder, head)
    model.head.eval()

    test_loader = DataLoader(
        StateDataset(test_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_metrics = evaluate(model, test_loader, nn.SmoothL1Loss(), "cpu")

    logger.info(f"{'Dimension':25s} {'MAE':>8s} {'MSE':>8s} {'Pearson':>8s}")
    logger.info("-" * 52)
    for dim in ALL_DIMENSIONS:
        logger.info(
            f"{dim:25s} {test_metrics[f'{dim}_mae']:8.4f} "
            f"{test_metrics[f'{dim}_mse']:8.4f} {test_metrics[f'{dim}_pearson']:8.4f}"
        )
    logger.info(f"{'MEAN MAE':25s} {test_metrics['mean_mae']:8.4f}")
    logger.info(f"{'MEAN PEARSON':25s} {test_metrics['mean_pearson']:8.4f}")

    # Save eval report
    eval_report = {
        "test_count": len(test_examples),
        "test_metrics": test_metrics,
        "quality_gates": {
            "mean_mae_threshold": 0.18,
            "max_dim_mae_threshold": 0.25,
            "min_pearson_threshold": 0.60,
            "mean_mae_passed": test_metrics["mean_mae"] < 0.18,
            "max_dim_mae_passed": all(
                test_metrics[f"{d}_mae"] < 0.25 for d in ALL_DIMENSIONS
            ),
            "min_pearson_passed": all(
                test_metrics[f"{d}_pearson"] > 0.60 for d in ALL_DIMENSIONS
            ),
        },
    }

    eval_path = output_dir / "eval_report.json"
    with eval_path.open("w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Saved evaluation report to {eval_path}")

    # Quality gate check
    gates = eval_report["quality_gates"]
    all_passed = all([
        gates["mean_mae_passed"],
        gates["max_dim_mae_passed"],
        gates["min_pearson_passed"],
    ])

    if all_passed:
        logger.info("ALL QUALITY GATES PASSED")
    else:
        logger.warning("QUALITY GATES FAILED:")
        if not gates["mean_mae_passed"]:
            logger.warning(f"  Mean MAE {test_metrics['mean_mae']:.4f} > 0.18")
        if not gates["max_dim_mae_passed"]:
            for d in ALL_DIMENSIONS:
                if test_metrics[f"{d}_mae"] >= 0.25:
                    logger.warning(f"  {d} MAE {test_metrics[f'{d}_mae']:.4f} > 0.25")
        if not gates["min_pearson_passed"]:
            for d in ALL_DIMENSIONS:
                if test_metrics[f"{d}_pearson"] <= 0.60:
                    logger.warning(f"  {d} Pearson {test_metrics[f'{d}_pearson']:.4f} < 0.60")


if __name__ == "__main__":
    main()
