"""Train a 5-class entity type classifier on Claude-labeled data.

Architecture: sentence-transformer encoder → MLP → 5-class softmax
Training: two-phase (frozen encoder, then unfrozen with discriminative LR)

Uses CrossEntropyLoss with class weights for imbalance (concept dominates).
Input format: "[ENTITY] {name} [CONTEXT] {window}" — short inputs, no chunking.

Features rich terminal dashboard + optional W&B integration for observability.

Usage:
  python train_entity_classifier.py \
    --data entity_training_data.jsonl \
    --splits entity_splits.json \
    --output ./entity-classifier-mpnet-v1 \
    [--base-model all-mpnet-base-v2] \
    [--epochs-frozen 8] [--epochs-unfrozen 30] [--batch-size 16] [--device cuda] \
    [--wandb]  # Enable W&B logging
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

# Dashboard for observability
try:
    from training_dashboard import TrainingDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    TrainingDashboard = None

# Known sentence-transformer models
_SENTENCE_TRANSFORMER_MODELS = {
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-mpnet-base-v2",
}

ENTITY_TYPES = ["concept", "organization", "person", "place", "spiritual"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train-entity")


def build_encoder(model_name: str, device: str) -> tuple[SentenceTransformer, int]:
    """Build a SentenceTransformer encoder from a model name."""
    if model_name in _SENTENCE_TRANSFORMER_MODELS:
        encoder = SentenceTransformer(model_name, device=device)
        embed_dim = encoder.get_sentence_embedding_dimension()
        logger.info(f"Loaded SentenceTransformer '{model_name}' (dim={embed_dim})")
        return encoder, embed_dim

    from sentence_transformers import models as st_models

    word_model = st_models.Transformer(model_name, max_seq_length=512)
    pooling = st_models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    encoder = SentenceTransformer(modules=[word_model, pooling], device=device)
    encoder = encoder.float()
    embed_dim = encoder.get_sentence_embedding_dimension()
    logger.info(f"Built SentenceTransformer from '{model_name}' (dim={embed_dim})")
    return encoder, embed_dim


# ─── Dataset ─────────────────────────────────────────────────────────────────


@dataclass
class EntityExample:
    entry_id: str
    entity_name: str
    text: str  # "[ENTITY] name [CONTEXT] window"
    label: int  # class index


class EntityDataset(Dataset):
    def __init__(self, examples: list[EntityExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.text, ex.label


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


# ─── Model ───────────────────────────────────────────────────────────────────


class ClassificationHead(nn.Module):
    """MLP classification head: Linear→LayerNorm→GELU→Dropout→Linear."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits


class EntityClassifier(nn.Module):
    """Full model: sentence-transformer encoder + classification head."""

    def __init__(self, encoder: SentenceTransformer, head: ClassificationHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def encode_texts_no_grad(self, texts: list[str]) -> torch.Tensor:
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return torch.tensor(embeddings, dtype=torch.float32)

    def encode_texts_with_grad(self, texts: list[str], device: str) -> torch.Tensor:
        features = self.encoder.tokenize(texts)
        features = {k: v.to(device) for k, v in features.items()}
        trans_output = self.encoder[0](features)
        pooled = self.encoder[1](trans_output)
        embeddings = pooled["sentence_embedding"]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.float()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)


# ─── Training ────────────────────────────────────────────────────────────────


def train_epoch(
    model: EntityClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    encoder_frozen: bool = True,
    grad_accum_steps: int = 1,
    scheduler=None,
    batch_callback=None,  # Optional callback: fn(batch_idx, loss)
) -> float:
    model.head.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (texts, labels) in enumerate(dataloader):
        labels = labels.to(device)

        if encoder_frozen:
            with torch.no_grad():
                embeddings = model.encode_texts_no_grad(texts).to(device)
        else:
            embeddings = model.encode_texts_with_grad(texts, device)

        logits = model.forward(embeddings)
        loss = loss_fn(logits, labels)

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if not encoder_frozen:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

        # Call batch callback for live progress updates
        if batch_callback is not None:
            batch_callback(batch_idx, loss.item())

    return total_loss / max(n_batches, 1)


def evaluate(
    model: EntityClassifier,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (metrics_dict, all_preds, all_labels)."""
    model.head.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for texts, labels in dataloader:
            labels = labels.to(device)
            embeddings = model.encode_texts_no_grad(texts).to(device)
            logits = model.forward(embeddings)

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    accuracy = float(np.mean(preds == labels))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    return metrics, preds, labels


def build_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    base_model: str = "all-mpnet-base-v2"
    epochs_frozen: int = 8
    epochs_unfrozen: int = 30
    batch_size: int = 16  # larger since inputs are short
    grad_accum_steps: int = 2  # effective batch = 32
    lr_head_frozen: float = 1e-3
    lr_encoder: float = 5e-5
    lr_head_unfrozen: float = 5e-4
    hidden_dim: int = 256
    dropout: float = 0.1
    patience: int = 10
    use_scheduler: bool = True
    device: str = "cpu"


# ─── Data Loading ────────────────────────────────────────────────────────────


def format_input(entity_name: str, context: str) -> str:
    """Format entity + context into model input string."""
    return f"[ENTITY] {entity_name} [CONTEXT] {context}"


def load_examples(
    data_path: str,
    type_index: dict[str, int],
) -> list[EntityExample]:
    """Load all examples from JSONL file."""
    examples = []
    skipped = 0
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            etype = rec["entity_type"]
            if etype not in type_index:
                skipped += 1
                continue
            examples.append(
                EntityExample(
                    entry_id=rec["entry_id"],
                    entity_name=rec["entity_name"],
                    text=format_input(rec["entity_name"], rec["context"]),
                    label=type_index[etype],
                )
            )
    if skipped:
        logger.warning(f"Skipped {skipped} examples with unknown entity types")
    return examples


def compute_class_weights(
    examples: list[EntityExample],
    n_classes: int,
    clamp_max: float = 20.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.zeros(n_classes)
    for ex in examples:
        counts[ex.label] += 1

    # Inverse frequency: total / (n_classes * count_i)
    total = len(examples)
    weights = total / (n_classes * np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.5, clamp_max)

    logger.info("Class weights:")
    for i, etype in enumerate(ENTITY_TYPES):
        logger.info(f"  {etype:15s}: count={int(counts[i]):5d}, weight={weights[i]:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


# ─── Cross-Validation ────────────────────────────────────────────────────────


def run_cv_fold(
    fold_idx: int,
    train_examples: list[EntityExample],
    val_examples: list[EntityExample],
    config: TrainConfig,
    n_classes: int,
    class_weights: torch.Tensor,
) -> dict:
    logger.info(f"  Fold {fold_idx}: {len(train_examples)} train, {len(val_examples)} val")

    encoder, embed_dim = build_encoder(config.base_model, config.device)
    head = ClassificationHead(
        input_dim=embed_dim, hidden_dim=config.hidden_dim,
        output_dim=n_classes, dropout=config.dropout,
    )
    model = EntityClassifier(encoder, head)

    if config.device == "cuda":
        head = head.to("cuda")
        cw = class_weights.to("cuda")
    else:
        cw = class_weights

    train_loader = DataLoader(
        EntityDataset(train_examples), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        EntityDataset(val_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    loss_fn = nn.CrossEntropyLoss(weight=cw)

    # Phase A: Frozen encoder
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr_head_frozen, weight_decay=0.01)

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(config.epochs_frozen):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, config.device,
            encoder_frozen=True, grad_accum_steps=config.grad_accum_steps,
        )
        val_metrics, _, _ = evaluate(model, val_loader, loss_fn, config.device)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == config.epochs_frozen - 1:
            logger.info(
                f"    [Frozen] Epoch {epoch+1}/{config.epochs_frozen}: "
                f"loss={train_loss:.4f} acc={val_metrics['accuracy']:.4f} "
                f"macro_f1={val_metrics['macro_f1']:.4f}"
            )

        if patience_counter >= config.patience:
            logger.info(f"    Early stopping at epoch {epoch+1} (frozen phase)")
            break

    # Phase B: Unfrozen encoder
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

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(config.epochs_unfrozen):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, config.device,
            encoder_frozen=False, grad_accum_steps=config.grad_accum_steps,
            scheduler=scheduler,
        )
        val_metrics, _, _ = evaluate(model, val_loader, loss_fn, config.device)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 3 == 0 or epoch == config.epochs_unfrozen - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"    [Unfrozen] Epoch {epoch+1}/{config.epochs_unfrozen}: "
                f"loss={train_loss:.4f} acc={val_metrics['accuracy']:.4f} "
                f"macro_f1={val_metrics['macro_f1']:.4f} lr={cur_lr:.2e}"
            )

        if patience_counter >= config.patience:
            logger.info(f"    Early stopping at epoch {epoch+1} (unfrozen phase)")
            break

    return val_metrics


# ─── Full Training ───────────────────────────────────────────────────────────


def train_final_model(
    train_examples: list[EntityExample],
    val_examples: list[EntityExample],
    config: TrainConfig,
    n_classes: int,
    class_weights: torch.Tensor,
    output_dir: Path,
    dashboard: "TrainingDashboard | None" = None,
) -> dict:
    """Train final model with early stopping. Returns config_dict."""
    logger.info(
        f"Training final model on {len(train_examples)} examples "
        f"(val: {len(val_examples)}) with base model '{config.base_model}'..."
    )

    encoder, embed_dim = build_encoder(config.base_model, config.device)
    head = ClassificationHead(
        input_dim=embed_dim, hidden_dim=config.hidden_dim,
        output_dim=n_classes, dropout=config.dropout,
    )
    model = EntityClassifier(encoder, head)

    if config.device == "cuda":
        head = head.to("cuda")
        cw = class_weights.to("cuda")
    else:
        cw = class_weights

    train_loader = DataLoader(
        EntityDataset(train_examples), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        EntityDataset(val_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    loss_fn = nn.CrossEntropyLoss(weight=cw)

    # Phase A: Frozen encoder
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr_head_frozen, weight_decay=0.01)

    best_frozen_f1 = -1.0
    best_frozen_head_state = None
    frozen_patience_counter = 0
    frozen_patience = min(config.patience, 8)

    if dashboard:
        dashboard.start_phase("Frozen Encoder", config.epochs_frozen)

    for epoch in range(config.epochs_frozen):
        if dashboard:
            dashboard.start_epoch(epoch, len(train_loader))

        batch_cb = (lambda idx, loss: dashboard.log_batch(idx, loss)) if dashboard else None
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, config.device,
            encoder_frozen=True, grad_accum_steps=config.grad_accum_steps,
            batch_callback=batch_cb,
        )
        val_metrics, _, _ = evaluate(model, val_loader, loss_fn, config.device)
        is_best = val_metrics["macro_f1"] > best_frozen_f1

        if is_best:
            best_frozen_f1 = val_metrics["macro_f1"]
            best_frozen_head_state = copy.deepcopy(head.state_dict())
            frozen_patience_counter = 0
            logger.info(
                f"  [Final/Frozen] Epoch {epoch+1}: NEW BEST loss={train_loss:.4f} "
                f"acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
            )
        else:
            frozen_patience_counter += 1
            if epoch % 2 == 0 or epoch == config.epochs_frozen - 1:
                logger.info(
                    f"  [Final/Frozen] Epoch {epoch+1}: loss={train_loss:.4f} "
                    f"acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
                )

        if dashboard:
            dashboard.log_epoch(
                epoch,
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["macro_f1"],
                    "accuracy": val_metrics["accuracy"],
                    "weighted_f1": val_metrics["weighted_f1"],
                    "lr": config.lr_head_frozen,
                },
                is_best=is_best,
            )

        if frozen_patience_counter >= frozen_patience:
            logger.info(f"  Frozen early stopping at epoch {epoch+1}")
            break

    # Restore best frozen head state
    if best_frozen_head_state is not None:
        head.load_state_dict(best_frozen_head_state)
        logger.info(f"  Restored best frozen head state (macro_f1={best_frozen_f1:.4f})")

    # Initialize best state from frozen phase
    best_val_f1 = best_frozen_f1
    best_encoder_state = None
    best_head_state = copy.deepcopy(head.state_dict())

    if config.epochs_unfrozen > 0:
        # Phase B: Unfrozen encoder
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

        patience_counter = 0

        if dashboard:
            dashboard.start_phase("Fine-tuning Encoder", config.epochs_unfrozen)

        for epoch in range(config.epochs_unfrozen):
            if dashboard:
                dashboard.start_epoch(epoch, len(train_loader))

            batch_cb = (lambda idx, loss: dashboard.log_batch(idx, loss)) if dashboard else None
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, config.device,
                encoder_frozen=False, grad_accum_steps=config.grad_accum_steps,
                scheduler=scheduler, batch_callback=batch_cb,
            )
            val_metrics, _, _ = evaluate(model, val_loader, loss_fn, config.device)
            cur_lr = optimizer.param_groups[0]["lr"]
            is_best = val_metrics["macro_f1"] > best_val_f1

            if is_best:
                best_val_f1 = val_metrics["macro_f1"]
                patience_counter = 0
                best_head_state = copy.deepcopy(head.state_dict())
                best_encoder_state = copy.deepcopy(encoder[0].auto_model.state_dict())
                logger.info(
                    f"  [Final/Unfrozen] Epoch {epoch+1}: NEW BEST "
                    f"acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
                )
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == config.epochs_unfrozen - 1:
                logger.info(
                    f"  [Final/Unfrozen] Epoch {epoch+1}: loss={train_loss:.4f} "
                    f"acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f} "
                    f"lr={cur_lr:.2e}"
                )

            if dashboard:
                dashboard.log_epoch(
                    epoch,
                    {
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_f1": val_metrics["macro_f1"],
                        "accuracy": val_metrics["accuracy"],
                        "weighted_f1": val_metrics["weighted_f1"],
                        "lr": cur_lr,
                    },
                    is_best=is_best,
                )

            if patience_counter >= config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best checkpoint
        if best_encoder_state is not None:
            encoder[0].auto_model.load_state_dict(best_encoder_state)
        if best_head_state is not None:
            head.load_state_dict(best_head_state)
        logger.info(f"  Restored best checkpoint (macro_f1={best_val_f1:.4f})")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(str(model_dir))
    logger.info(f"  Saved encoder to {model_dir}")

    head_path = output_dir / "classification_head.pt"
    torch.save(head.cpu().state_dict(), str(head_path))
    logger.info(f"  Saved classification head to {head_path}")

    # Save entity type index
    type_index = {etype: i for i, etype in enumerate(ENTITY_TYPES)}
    type_index_path = output_dir / "entity_type_index.json"
    with type_index_path.open("w") as f:
        json.dump(type_index, f, indent=2)

    # Save training config
    model_short = config.base_model.split("/")[-1].lower()
    config_dict = {
        "task": "multi-class-classification",
        "n_classes": n_classes,
        "class_names": ENTITY_TYPES,
        "input_dim": embed_dim,
        "hidden_dim": config.hidden_dim,
        "output_dim": n_classes,
        "base_model": config.base_model,
        "prompt_version": "entity-finetuned-v1",
        "model_version": f"finetuned-entity-{model_short}-v1",
        "epochs_frozen": config.epochs_frozen,
        "epochs_unfrozen": config.epochs_unfrozen,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "effective_batch_size": config.batch_size * config.grad_accum_steps,
        "lr_encoder": config.lr_encoder,
        "lr_head_frozen": config.lr_head_frozen,
        "lr_head_unfrozen": config.lr_head_unfrozen,
        "dropout": config.dropout,
        "loss": "CrossEntropyLoss",
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "best_val_macro_f1": best_val_f1,
    }
    config_path = output_dir / "training_config.json"
    with config_path.open("w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"  Saved config to {config_path}")

    return config_dict


def main():
    parser = argparse.ArgumentParser(description="Train entity type classifier")
    parser.add_argument("--data", required=True, help="Path to entity_training_data.jsonl")
    parser.add_argument("--splits", required=True, help="Path to entity_splits.json")
    parser.add_argument("--output", default="./entity-classifier-mpnet-v1", help="Output directory")
    parser.add_argument("--base-model", default="all-mpnet-base-v2")
    parser.add_argument("--epochs-frozen", type=int, default=8)
    parser.add_argument("--epochs-unfrozen", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-encoder", type=float, default=5e-5)
    parser.add_argument("--lr-head-frozen", type=float, default=1e-3)
    parser.add_argument("--lr-head-unfrozen", type=float, default=5e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight-clamp", type=float, default=20.0)
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging for cloud-based tracking",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable rich terminal dashboard (use plain logging)",
    )
    args = parser.parse_args()

    config = TrainConfig(
        base_model=args.base_model,
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr_head_frozen=args.lr_head_frozen,
        lr_encoder=args.lr_encoder,
        lr_head_unfrozen=args.lr_head_unfrozen,
        device=args.device,
        patience=args.patience,
    )

    eff_batch = config.batch_size * config.grad_accum_steps
    logger.info(f"Device: {config.device}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(
        f"Config: frozen={config.epochs_frozen}, unfrozen={config.epochs_unfrozen}, "
        f"batch={config.batch_size}x{config.grad_accum_steps}={eff_batch}, "
        f"hidden={config.hidden_dim}, dropout={config.dropout}"
    )

    # Entity type index
    type_index = {etype: i for i, etype in enumerate(ENTITY_TYPES)}
    n_classes = len(ENTITY_TYPES)
    logger.info(f"Entity types ({n_classes}): {ENTITY_TYPES}")

    # Load data
    all_examples = load_examples(args.data, type_index)
    logger.info(f"Loaded {len(all_examples)} examples")

    with open(args.splits) as f:
        splits = json.load(f)

    test_ids = set(splits["test_ids"])
    train_ids = set(splits["train_ids"])

    test_examples = [ex for ex in all_examples if ex.entry_id in test_ids]
    train_examples = [ex for ex in all_examples if ex.entry_id in train_ids]

    logger.info(f"Train: {len(train_examples)}, Test: {len(test_examples)}")

    # Compute class weights
    class_weights = compute_class_weights(train_examples, n_classes, clamp_max=args.weight_clamp)

    # Cross-validation
    if not args.skip_cv:
        logger.info("=" * 60)
        logger.info("Running 5-fold cross-validation...")
        cv_results = []

        for fold_idx in range(splits["n_folds"]):
            fold_val_ids = set(splits["folds"][str(fold_idx)])
            fold_train = [ex for ex in train_examples if ex.entry_id not in fold_val_ids]
            fold_val = [ex for ex in train_examples if ex.entry_id in fold_val_ids]

            fold_metrics = run_cv_fold(
                fold_idx, fold_train, fold_val, config, n_classes, class_weights,
            )
            cv_results.append(fold_metrics)

        logger.info("=" * 60)
        logger.info("Cross-validation results:")
        for key in ["accuracy", "macro_f1", "weighted_f1"]:
            values = [r[key] for r in cv_results]
            logger.info(f"  {key:20s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    # Train final model
    logger.info("=" * 60)
    val_frac = args.val_fraction
    val_every_n = max(2, int(round(1.0 / val_frac)))
    final_val = [ex for i, ex in enumerate(train_examples) if i % val_every_n == 0]
    final_train = [ex for i, ex in enumerate(train_examples) if i % val_every_n != 0]
    actual_pct = 100 * len(final_val) / len(train_examples)
    logger.info(f"Final model split: {len(final_train)} train, {len(final_val)} val ({actual_pct:.0f}%)")

    output_dir = Path(args.output)

    # Create dashboard for observability
    dashboard = None
    use_dashboard = DASHBOARD_AVAILABLE and not args.no_dashboard
    if use_dashboard:
        model_short = config.base_model.split("/")[-1].split("-")[0]
        run_name = f"entity-{model_short}-{int(time.time())}"
        dashboard_config = {
            "base_model": config.base_model,
            "epochs_frozen": config.epochs_frozen,
            "epochs_unfrozen": config.epochs_unfrozen,
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "lr_encoder": config.lr_encoder,
            "lr_head_frozen": config.lr_head_frozen,
            "lr_head_unfrozen": config.lr_head_unfrozen,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "patience": config.patience,
            "n_classes": n_classes,
            "class_names": ENTITY_TYPES,
            "train_count": len(final_train),
            "val_count": len(final_val),
            "device": config.device,
        }
        dashboard = TrainingDashboard(
            project="corpus-mcp-training",
            run_name=run_name,
            task_type="entity-classifier",
            config=dashboard_config,
            use_wandb=args.wandb,
        )
        dashboard.start()

    start_time = time.time()
    try:
        config_dict = train_final_model(
            final_train, final_val, config, n_classes, class_weights, output_dir,
            dashboard=dashboard,
        )
    finally:
        if dashboard:
            dashboard.finish()

    elapsed = time.time() - start_time
    logger.info(f"Final model training completed in {elapsed:.1f}s")

    # Evaluate on test set
    logger.info("=" * 60)
    logger.info("Evaluating on hold-out test set...")

    encoder = SentenceTransformer(str(output_dir / "model"))
    saved_embed_dim = encoder.get_sentence_embedding_dimension()
    head = ClassificationHead(
        input_dim=saved_embed_dim, hidden_dim=config.hidden_dim,
        output_dim=n_classes, dropout=config.dropout,
    )
    head.load_state_dict(
        torch.load(str(output_dir / "classification_head.pt"), map_location="cpu", weights_only=True)
    )
    model = EntityClassifier(encoder, head)
    model.head.eval()

    test_loader = DataLoader(
        EntityDataset(test_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, loss_fn, "cpu")

    logger.info("Test results:")
    for key in ["accuracy", "macro_f1", "weighted_f1"]:
        logger.info(f"  {key:20s}: {test_metrics[key]:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(
        test_labels, test_preds, target_names=ENTITY_TYPES, zero_division=0,
    )
    logger.info("\n" + report)

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    logger.info("Confusion Matrix:")
    logger.info(f"{'':15s} " + " ".join(f"{t[:6]:>6s}" for t in ENTITY_TYPES))
    for i, row in enumerate(cm):
        logger.info(f"  {ENTITY_TYPES[i]:13s} " + " ".join(f"{v:6d}" for v in row))

    # Save eval report
    per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
    eval_report = {
        "test_count": len(test_examples),
        "n_classes": n_classes,
        "test_metrics": test_metrics,
        "per_class_f1": {ENTITY_TYPES[i]: float(per_class_f1[i]) for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "quality_gates": {
            "macro_f1_threshold": 0.60,
            "accuracy_threshold": 0.70,
            "macro_f1_passed": test_metrics["macro_f1"] > 0.60,
            "accuracy_passed": test_metrics["accuracy"] > 0.70,
        },
    }

    eval_path = output_dir / "eval_report.json"
    with eval_path.open("w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"\nSaved evaluation report to {eval_path}")

    # Quality gate check
    gates = eval_report["quality_gates"]
    all_passed = gates["macro_f1_passed"] and gates["accuracy_passed"]

    if all_passed:
        logger.info("ALL QUALITY GATES PASSED")
    else:
        logger.warning("QUALITY GATES FAILED:")
        if not gates["macro_f1_passed"]:
            logger.warning(f"  Macro F1 {test_metrics['macro_f1']:.4f} <= 0.60")
        if not gates["accuracy_passed"]:
            logger.warning(f"  Accuracy {test_metrics['accuracy']:.4f} <= 0.70")


if __name__ == "__main__":
    main()
