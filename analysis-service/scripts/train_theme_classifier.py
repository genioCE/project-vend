"""Train a multi-label theme classifier on Claude-labeled journal data.

Architecture: transformer encoder → shared MLP → N-label classification head (sigmoid)
Training: two-phase (frozen encoder, then unfrozen with discriminative LR)

Uses BCEWithLogitsLoss with per-label pos_weight for class imbalance.
After training, optimizes per-label classification thresholds on validation set.

Usage:
  python train_theme_classifier.py \
    --data theme_training_data.jsonl \
    --splits theme_splits.json \
    --label-index theme_label_index.json \
    --output ./theme-classifier-mpnet-v1 \
    [--base-model all-mpnet-base-v2] \
    [--epochs-frozen 8] [--epochs-unfrozen 30] [--batch-size 8] [--device cuda]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
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
logger = logging.getLogger("train-theme")


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
class ThemeExample:
    entry_id: str
    text: str
    labels: list[float]  # binary vector: 0.0 or 1.0 for each theme


class ThemeDataset(Dataset):
    def __init__(self, examples: list[ThemeExample]):
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


class ClassificationHead(nn.Module):
    """Classification head. MLP when hidden_dim>0, linear otherwise.

    MLP: Linear(768→H) → LayerNorm → GELU → Dropout → Linear(H→N_labels)
    Linear: Dropout → Linear(768→N_labels)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 150, dropout: float = 0.1):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            # Linear-only: much fewer parameters, less overfitting
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits — BCEWithLogitsLoss applies sigmoid


class ThemeClassifier(nn.Module):
    """Full model: sentence-transformer encoder + classification head."""

    def __init__(self, encoder: SentenceTransformer, head: ClassificationHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def encode_texts_no_grad(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using .encode() — no gradients. For frozen phase + eval."""
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return torch.tensor(embeddings, dtype=torch.float32)

    def encode_texts_with_grad(self, texts: list[str], device: str) -> torch.Tensor:
        """Encode texts WITH gradient tracking. For fine-tuning encoder."""
        features = self.encoder.tokenize(texts)
        features = {k: v.to(device) for k, v in features.items()}
        trans_output = self.encoder[0](features)
        pooled = self.encoder[1](trans_output)
        embeddings = pooled["sentence_embedding"]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.float()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head. Returns raw logits."""
        return self.head(embeddings)


# ─── Metrics ─────────────────────────────────────────────────────────────────


def compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Compute multi-label classification metrics.

    Args:
        all_preds: (N, L) array of sigmoid probabilities
        all_labels: (N, L) array of binary labels
        thresholds: (L,) array of per-label thresholds, or None for 0.5
    """
    n_labels = all_labels.shape[1]
    if thresholds is None:
        thresholds = np.full(n_labels, 0.5)

    binary_preds = (all_preds >= thresholds).astype(int)

    # Micro and macro F1
    micro_f1 = float(f1_score(all_labels, binary_preds, average="micro", zero_division=0))
    macro_f1 = float(f1_score(all_labels, binary_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_labels, binary_preds, average="weighted", zero_division=0))

    micro_prec = float(precision_score(all_labels, binary_preds, average="micro", zero_division=0))
    micro_recall = float(recall_score(all_labels, binary_preds, average="micro", zero_division=0))

    # Hamming loss: fraction of wrong labels
    hamming = float(np.mean(all_labels != binary_preds))

    # Exact match ratio (subset accuracy)
    exact_match = float(np.mean(np.all(all_labels == binary_preds, axis=1)))

    # Mean average precision (mAP) — computed per-label
    per_label_ap = []
    for i in range(n_labels):
        support = int(all_labels[:, i].sum())
        if support == 0:
            continue
        # Sort by predicted probability descending
        sorted_idx = np.argsort(-all_preds[:, i])
        sorted_labels = all_labels[sorted_idx, i]
        # Compute AP
        cum_tp = np.cumsum(sorted_labels)
        precision_at_k = cum_tp / np.arange(1, len(sorted_labels) + 1)
        ap = float(np.sum(precision_at_k * sorted_labels) / support)
        per_label_ap.append(ap)
    mAP = float(np.mean(per_label_ap)) if per_label_ap else 0.0

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_prec,
        "micro_recall": micro_recall,
        "hamming_loss": hamming,
        "exact_match": exact_match,
        "mAP": mAP,
    }


# ─── Training ────────────────────────────────────────────────────────────────


def train_epoch(
    model: ThemeClassifier,
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

    return total_loss / max(n_batches, 1)


def evaluate(
    model: ThemeClassifier,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    thresholds: np.ndarray | None = None,
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

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(preds, labels, thresholds)
    metrics["loss"] = total_loss / max(n_batches, 1)

    return metrics, preds, labels


def optimize_thresholds(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    sweep_range: tuple[float, float, float] = (0.1, 0.9, 0.05),
) -> np.ndarray:
    """Find optimal per-label threshold by sweeping to maximize per-label F1."""
    n_labels = all_labels.shape[1]
    thresholds = np.full(n_labels, 0.5)
    low, high, step = sweep_range

    for i in range(n_labels):
        support = int(all_labels[:, i].sum())
        if support == 0:
            thresholds[i] = 0.5
            continue

        best_f1 = -1.0
        best_t = 0.5

        for t in np.arange(low, high + step / 2, step):
            preds_binary = (all_preds[:, i] >= t).astype(int)
            f1 = float(f1_score(all_labels[:, i], preds_binary, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[i] = best_t

    return thresholds


# ─── LR Scheduler ────────────────────────────────────────────────────────────


def build_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ─── Cross-Validation ────────────────────────────────────────────────────────


def run_cv_fold(
    fold_idx: int,
    train_examples: list[ThemeExample],
    val_examples: list[ThemeExample],
    config: TrainConfig,
    n_labels: int,
    pos_weight: torch.Tensor,
) -> dict:
    logger.info(f"  Fold {fold_idx}: {len(train_examples)} train, {len(val_examples)} val")

    encoder, embed_dim = build_encoder(config.base_model, config.device)
    head = ClassificationHead(
        input_dim=embed_dim, hidden_dim=config.hidden_dim,
        output_dim=n_labels, dropout=config.dropout,
    )
    model = ThemeClassifier(encoder, head)

    if config.device == "cuda":
        head = head.to("cuda")
        pw = pos_weight.to("cuda")
    else:
        pw = pos_weight

    train_loader = DataLoader(
        ThemeDataset(train_examples), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ThemeDataset(val_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

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
        val_metrics, val_preds, val_labels = evaluate(model, val_loader, loss_fn, config.device)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == config.epochs_frozen - 1:
            logger.info(
                f"    [Frozen] Epoch {epoch+1}/{config.epochs_frozen}: "
                f"loss={train_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_micro_f1={val_metrics['micro_f1']:.4f}"
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
        val_metrics, val_preds, val_labels = evaluate(model, val_loader, loss_fn, config.device)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 3 == 0 or epoch == config.epochs_unfrozen - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"    [Unfrozen] Epoch {epoch+1}/{config.epochs_unfrozen}: "
                f"loss={train_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_micro_f1={val_metrics['micro_f1']:.4f} lr={cur_lr:.2e}"
            )

        if patience_counter >= config.patience:
            logger.info(f"    Early stopping at epoch {epoch+1} (unfrozen phase)")
            break

    return val_metrics


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


def load_examples(data_path: str, label_index: dict[str, int]) -> dict[str, ThemeExample]:
    """Load all examples from JSONL file, keyed by entry_id."""
    n_labels = len(label_index)
    index_order = sorted(label_index.keys(), key=lambda k: label_index[k])

    examples = {}
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            labels = [float(rec["labels"].get(label, 0.0)) for label in index_order]
            examples[rec["entry_id"]] = ThemeExample(
                entry_id=rec["entry_id"],
                text=rec["text"],
                labels=labels,
            )
    return examples


def compute_pos_weight(examples: list[ThemeExample], n_labels: int, clamp_max: float = 20.0) -> torch.Tensor:
    """Compute per-label pos_weight = num_negatives / num_positives, clamped."""
    label_matrix = np.array([ex.labels for ex in examples])
    pos_counts = label_matrix.sum(axis=0)  # (n_labels,)
    neg_counts = len(examples) - pos_counts

    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = neg_counts / pos_counts
    weights = np.clip(weights, 1.0, clamp_max)

    return torch.tensor(weights, dtype=torch.float32)


def train_final_model(
    train_examples: list[ThemeExample],
    val_examples: list[ThemeExample],
    config: TrainConfig,
    n_labels: int,
    pos_weight: torch.Tensor,
    label_names: list[str],
    output_dir: Path,
) -> tuple[dict, np.ndarray]:
    """Train the final model with validation-based early stopping. Returns (config_dict, thresholds)."""
    logger.info(f"Training final model on {len(train_examples)} examples "
                f"(val: {len(val_examples)}) with base model '{config.base_model}'...")

    encoder, embed_dim = build_encoder(config.base_model, config.device)
    head = ClassificationHead(
        input_dim=embed_dim, hidden_dim=config.hidden_dim,
        output_dim=n_labels, dropout=config.dropout,
    )
    model = ThemeClassifier(encoder, head)

    if config.device == "cuda":
        head = head.to("cuda")
        pw = pos_weight.to("cuda")
    else:
        pw = pos_weight

    train_loader = DataLoader(
        ThemeDataset(train_examples), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ThemeDataset(val_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Phase A: Frozen encoder — with early stopping + best state tracking
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr_head_frozen, weight_decay=0.01)

    best_frozen_f1 = -1.0
    best_frozen_head_state = None
    frozen_patience_counter = 0
    frozen_patience = min(config.patience, 8)  # shorter patience for frozen phase

    for epoch in range(config.epochs_frozen):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, config.device,
            encoder_frozen=True, grad_accum_steps=config.grad_accum_steps,
        )
        val_metrics, _, _ = evaluate(model, val_loader, loss_fn, config.device)

        if val_metrics["macro_f1"] > best_frozen_f1:
            best_frozen_f1 = val_metrics["macro_f1"]
            best_frozen_head_state = copy.deepcopy(head.state_dict())
            frozen_patience_counter = 0
            logger.info(
                f"  [Final/Frozen] Epoch {epoch+1}: NEW BEST loss={train_loss:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )
        else:
            frozen_patience_counter += 1
            if epoch % 2 == 0 or epoch == config.epochs_frozen - 1:
                logger.info(
                    f"  [Final/Frozen] Epoch {epoch+1}: loss={train_loss:.4f} "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f} (no improvement {frozen_patience_counter}/{frozen_patience})"
                )

        if frozen_patience_counter >= frozen_patience:
            logger.info(f"  Frozen early stopping at epoch {epoch+1} (best macro_f1={best_frozen_f1:.4f})")
            break

    # Restore best frozen head state before unfreezing
    if best_frozen_head_state is not None:
        head.load_state_dict(best_frozen_head_state)
        logger.info(f"  Restored best frozen head state (macro_f1={best_frozen_f1:.4f})")

    # Evaluate frozen model for threshold optimization (used if frozen-only or as fallback)
    frozen_metrics, frozen_val_preds, frozen_val_labels = evaluate(model, val_loader, loss_fn, config.device)
    logger.info(f"  Frozen model val: macro_f1={frozen_metrics['macro_f1']:.4f} "
                f"micro_f1={frozen_metrics['micro_f1']:.4f}")

    # Initialize best state from frozen phase
    best_val_f1 = frozen_metrics["macro_f1"]
    best_val_preds = frozen_val_preds.copy()
    best_val_labels = frozen_val_labels.copy()
    best_encoder_state = None
    best_head_state = copy.deepcopy(head.state_dict())

    if config.epochs_unfrozen == 0:
        logger.info("  Frozen-only mode: skipping encoder fine-tuning")
    else:
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

        for epoch in range(config.epochs_unfrozen):
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, config.device,
                encoder_frozen=False, grad_accum_steps=config.grad_accum_steps,
                scheduler=scheduler,
            )
            val_metrics, val_preds, val_labels = evaluate(model, val_loader, loss_fn, config.device)

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                patience_counter = 0
                best_head_state = copy.deepcopy(head.state_dict())
                best_encoder_state = copy.deepcopy(encoder[0].auto_model.state_dict())
                best_val_preds = val_preds.copy()
                best_val_labels = val_labels.copy()
                logger.info(
                    f"  [Final/Unfrozen] Epoch {epoch+1}: NEW BEST "
                    f"macro_f1={val_metrics['macro_f1']:.4f} "
                    f"micro_f1={val_metrics['micro_f1']:.4f}"
                )
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == config.epochs_unfrozen - 1:
                cur_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  [Final/Unfrozen] Epoch {epoch+1}: loss={train_loss:.4f} "
                    f"macro_f1={val_metrics['macro_f1']:.4f} "
                    f"micro_f1={val_metrics['micro_f1']:.4f} lr={cur_lr:.2e}"
                )

            if patience_counter >= config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1} (best macro_f1={best_val_f1:.4f})")
                break

        # Restore best checkpoint from unfrozen phase
        if best_encoder_state is not None:
            encoder[0].auto_model.load_state_dict(best_encoder_state)
        if best_head_state is not None:
            head.load_state_dict(best_head_state)
        logger.info(f"  Restored best unfrozen checkpoint (macro_f1={best_val_f1:.4f})")

    # Optimize per-label thresholds on validation predictions from best epoch
    logger.info("  Optimizing per-label thresholds on validation set...")
    thresholds = optimize_thresholds(best_val_preds, best_val_labels)

    # Evaluate with optimized thresholds
    optimized_metrics = compute_metrics(best_val_preds, best_val_labels, thresholds)
    logger.info(
        f"  After threshold tuning: macro_f1={optimized_metrics['macro_f1']:.4f} "
        f"micro_f1={optimized_metrics['micro_f1']:.4f} "
        f"(was macro_f1={best_val_f1:.4f} with default 0.5)"
    )

    # Save model
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(str(model_dir))
    logger.info(f"  Saved encoder to {model_dir}")

    # Save classification head
    head_path = output_dir / "classification_head.pt"
    torch.save(head.cpu().state_dict(), str(head_path))
    logger.info(f"  Saved classification head to {head_path}")

    # Save thresholds
    threshold_dict = {label: float(thresholds[i]) for i, label in enumerate(label_names)}
    threshold_path = output_dir / "theme_thresholds.json"
    with threshold_path.open("w") as f:
        json.dump(threshold_dict, f, indent=2)
    logger.info(f"  Saved thresholds to {threshold_path}")

    # Save training config
    model_short = config.base_model.split("/")[-1].lower()
    config_dict = {
        "task": "multi-label-classification",
        "n_labels": n_labels,
        "label_names": label_names,
        "input_dim": embed_dim,
        "hidden_dim": config.hidden_dim,
        "output_dim": n_labels,
        "base_model": config.base_model,
        "prompt_version": "theme-finetuned-v1",
        "model_version": f"finetuned-theme-{model_short}-v1",
        "epochs_frozen": config.epochs_frozen,
        "epochs_unfrozen": config.epochs_unfrozen,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "effective_batch_size": config.batch_size * config.grad_accum_steps,
        "lr_encoder": config.lr_encoder,
        "lr_head_frozen": config.lr_head_frozen,
        "lr_head_unfrozen": config.lr_head_unfrozen,
        "dropout": config.dropout,
        "loss": "BCEWithLogitsLoss",
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "best_val_macro_f1": best_val_f1,
    }
    config_path = output_dir / "training_config.json"
    with config_path.open("w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"  Saved config to {config_path}")

    return config_dict, thresholds


def main():
    parser = argparse.ArgumentParser(description="Train multi-label theme classifier")
    parser.add_argument("--data", required=True, help="Path to theme_training_data.jsonl")
    parser.add_argument("--splits", required=True, help="Path to theme_splits.json")
    parser.add_argument("--label-index", required=True, help="Path to theme_label_index.json")
    parser.add_argument("--output", default="./theme-classifier-mpnet-v1", help="Output directory")
    parser.add_argument(
        "--base-model", default="all-mpnet-base-v2",
        help="Base encoder model name",
    )
    parser.add_argument("--epochs-frozen", type=int, default=8)
    parser.add_argument("--epochs-unfrozen", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-encoder", type=float, default=5e-5)
    parser.add_argument("--lr-head-frozen", type=float, default=1e-3,
                        help="Head LR during frozen encoder phase")
    parser.add_argument("--lr-head-unfrozen", type=float, default=5e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of training data for validation during final training (default 0.15)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation, train final only")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable cosine LR scheduler")
    parser.add_argument("--pos-weight-clamp", type=float, default=20.0,
                        help="Max pos_weight for BCEWithLogitsLoss (handles class imbalance)")
    parser.add_argument("--max-labels", type=int, default=0,
                        help="Only train on the top N most frequent labels (0 = all)")
    parser.add_argument("--frozen-only", action="store_true",
                        help="Only train the head with frozen encoder (no fine-tuning)")
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
        patience=999 if args.no_early_stop else args.patience,
        use_scheduler=not args.no_scheduler,
    )

    eff_batch = config.batch_size * config.grad_accum_steps
    logger.info(f"Device: {config.device}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Config: frozen={config.epochs_frozen}, unfrozen={config.epochs_unfrozen}, "
                f"batch={config.batch_size}x{config.grad_accum_steps}={eff_batch}, "
                f"hidden={config.hidden_dim}, dropout={config.dropout}, "
                f"lr_enc={config.lr_encoder}, lr_head_frozen={config.lr_head_frozen}, "
                f"lr_head_unfrozen={config.lr_head_unfrozen}, val_frac={args.val_fraction}")

    # Load label index
    with open(args.label_index) as f:
        label_index = json.load(f)

    # --max-labels: only keep the top N most frequent labels
    if args.max_labels > 0 and args.max_labels < len(label_index):
        # label_index is sorted by frequency (index 0 = most frequent)
        all_labels_sorted = sorted(label_index.keys(), key=lambda k: label_index[k])
        kept_labels = all_labels_sorted[:args.max_labels]
        label_index = {label: idx for idx, label in enumerate(kept_labels)}
        logger.info(f"--max-labels={args.max_labels}: keeping top labels: {kept_labels}")

    n_labels = len(label_index)
    label_names = sorted(label_index.keys(), key=lambda k: label_index[k])
    logger.info(f"Label count: {n_labels}")

    if args.frozen_only:
        config.epochs_unfrozen = 0
        logger.info("Frozen-only mode: encoder will NOT be fine-tuned")

    # Load data
    all_examples = load_examples(args.data, label_index)
    logger.info(f"Loaded {len(all_examples)} examples")

    with open(args.splits) as f:
        splits = json.load(f)

    test_ids = set(splits["test_ids"])
    train_ids = splits["train_ids"]

    test_examples = [all_examples[eid] for eid in test_ids if eid in all_examples]
    train_examples = [all_examples[eid] for eid in train_ids if eid in all_examples]

    logger.info(f"Train: {len(train_examples)}, Test: {len(test_examples)}")

    # Compute pos_weight from training set
    pos_weight = compute_pos_weight(train_examples, n_labels, clamp_max=args.pos_weight_clamp)
    logger.info(f"pos_weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

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
                fold_idx, fold_train, fold_val, config, n_labels, pos_weight,
            )
            cv_results.append(fold_metrics)

        logger.info("=" * 60)
        logger.info("Cross-validation results:")
        for key in ["macro_f1", "micro_f1", "weighted_f1", "hamming_loss", "mAP"]:
            values = [r[key] for r in cv_results]
            logger.info(f"  {key:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Train final model — holdout for early stopping + threshold tuning
    logger.info("=" * 60)
    val_frac = args.val_fraction
    val_every_n = max(2, int(round(1.0 / val_frac)))
    final_val = [ex for i, ex in enumerate(train_examples) if i % val_every_n == 0]
    final_train = [ex for i, ex in enumerate(train_examples) if i % val_every_n != 0]
    actual_pct = 100 * len(final_val) / len(train_examples)
    logger.info(f"Final model split: {len(final_train)} train, {len(final_val)} val ({actual_pct:.0f}%)")

    output_dir = Path(args.output)
    start_time = time.time()
    config_dict, thresholds = train_final_model(
        final_train, final_val, config, n_labels, pos_weight, label_names, output_dir,
    )
    elapsed = time.time() - start_time
    logger.info(f"Final model training completed in {elapsed:.1f}s")

    # Save the label index used for training (may be filtered by --max-labels)
    label_index_out = output_dir / "theme_label_index.json"
    with label_index_out.open("w") as f:
        json.dump(label_index, f, indent=2)

    clusters_path = Path(args.label_index).parent / "theme_clusters.json"
    if clusters_path.exists():
        shutil.copy2(str(clusters_path), str(output_dir / "theme_clusters.json"))

    # Evaluate on hold-out test set
    logger.info("=" * 60)
    logger.info("Evaluating on hold-out test set...")

    encoder = SentenceTransformer(str(output_dir / "model"))
    saved_embed_dim = encoder.get_sentence_embedding_dimension()
    head = ClassificationHead(
        input_dim=saved_embed_dim, hidden_dim=config.hidden_dim,
        output_dim=n_labels, dropout=config.dropout,
    )
    head.load_state_dict(
        torch.load(str(output_dir / "classification_head.pt"), map_location="cpu", weights_only=True)
    )
    model = ThemeClassifier(encoder, head)
    model.head.eval()

    test_loader = DataLoader(
        ThemeDataset(test_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    # Evaluate with default thresholds (0.5) and optimized thresholds
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_metrics_default, test_preds, test_labels = evaluate(
        model, test_loader, loss_fn, "cpu",
    )
    test_metrics_optimized = compute_metrics(test_preds, test_labels, thresholds)

    logger.info("Test results (threshold=0.5):")
    for key in ["macro_f1", "micro_f1", "weighted_f1", "hamming_loss", "mAP", "exact_match"]:
        logger.info(f"  {key:20s}: {test_metrics_default[key]:.4f}")

    logger.info("Test results (optimized thresholds):")
    for key in ["macro_f1", "micro_f1", "weighted_f1", "hamming_loss", "mAP", "exact_match"]:
        logger.info(f"  {key:20s}: {test_metrics_optimized[key]:.4f}")

    # Per-label metrics for labels with support >= 5
    logger.info("\nPer-label metrics (support >= 5, optimized thresholds):")
    logger.info(f"{'Label':40s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Sup':>5s} {'Thr':>5s}")
    logger.info("-" * 72)

    binary_preds = (test_preds >= thresholds).astype(int)
    per_label_f1 = f1_score(test_labels, binary_preds, average=None, zero_division=0)
    per_label_prec = precision_score(test_labels, binary_preds, average=None, zero_division=0)
    per_label_rec = recall_score(test_labels, binary_preds, average=None, zero_division=0)

    for i, label in enumerate(label_names):
        support = int(test_labels[:, i].sum())
        if support >= 5:
            logger.info(
                f"  {label:40s} {per_label_prec[i]:6.3f} {per_label_rec[i]:6.3f} "
                f"{per_label_f1[i]:6.3f} {support:5d} {thresholds[i]:5.2f}"
            )

    # Save eval report
    eval_report = {
        "test_count": len(test_examples),
        "n_labels": n_labels,
        "test_metrics_default_threshold": test_metrics_default,
        "test_metrics_optimized_threshold": test_metrics_optimized,
        "quality_gates": {
            "macro_f1_threshold": 0.30,
            "micro_f1_threshold": 0.45,
            "macro_f1_passed": test_metrics_optimized["macro_f1"] > 0.30,
            "micro_f1_passed": test_metrics_optimized["micro_f1"] > 0.45,
        },
    }

    eval_path = output_dir / "eval_report.json"
    with eval_path.open("w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"\nSaved evaluation report to {eval_path}")

    # Quality gate check
    gates = eval_report["quality_gates"]
    all_passed = gates["macro_f1_passed"] and gates["micro_f1_passed"]

    if all_passed:
        logger.info("ALL QUALITY GATES PASSED")
    else:
        logger.warning("QUALITY GATES FAILED:")
        if not gates["macro_f1_passed"]:
            logger.warning(f"  Macro F1 {test_metrics_optimized['macro_f1']:.4f} <= 0.30")
        if not gates["micro_f1_passed"]:
            logger.warning(f"  Micro F1 {test_metrics_optimized['micro_f1']:.4f} <= 0.45")


if __name__ == "__main__":
    main()
