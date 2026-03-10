"""Train a binary sentence-level decision detector on Claude-labeled data.

Architecture: sentence-transformer encoder → MLP → 1 (binary logit)
Training: two-phase (frozen encoder, then unfrozen with discriminative LR)

Uses BCEWithLogitsLoss with pos_weight for class imbalance (~5-10% positive rate).
Threshold optimized on validation set (sweep 0.1-0.9).

Features rich terminal dashboard + optional W&B integration for observability.

Usage:
  python train_decision_classifier.py \
    --data decision_training_data.jsonl \
    --splits decision_splits.json \
    --output ./decision-classifier-mpnet-v1 \
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
    precision_score,
    recall_score,
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
class FocalLoss(nn.Module):
    """Focal loss for binary classification with extreme class imbalance.

    Focal loss down-weights easy (well-classified) examples so the model
    focuses on hard negatives/positives. With gamma=2, an example classified
    with p=0.9 gets 100x less weight than one at p=0.5.

    Args:
        alpha: Weight for the positive class (like pos_weight in BCE).
            If None, no class weighting is applied.
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
            gamma=0 recovers standard BCE. Typical values: 1.0 - 5.0.
    """

    def __init__(self, alpha: float | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE component (numerically stable via logsigmoid)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = probability of correct class
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting (per-sample, based on class)
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * 1.0
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce
        return loss.mean()


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
logger = logging.getLogger("train-decision")


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
class DecisionExample:
    entry_id: str
    sentence: str
    label: int  # 0 or 1


class DecisionDataset(Dataset):
    def __init__(self, examples: list[DecisionExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.sentence, ex.label


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.float32)


# ─── Model ───────────────────────────────────────────────────────────────────


class BinaryHead(nn.Module):
    """MLP binary classification head: Linear→LayerNorm→GELU→Dropout→Linear(1)."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logits, shape (batch,)


class DecisionClassifier(nn.Module):
    """Full model: sentence-transformer encoder + binary head."""

    def __init__(self, encoder: SentenceTransformer, head: BinaryHead):
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
    model: DecisionClassifier,
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
    model: DecisionClassifier,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (metrics_dict, all_probs, all_labels)."""
    model.head.eval()
    all_logits = []
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

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits_arr = np.concatenate(all_logits)
    labels_arr = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-logits_arr))  # sigmoid

    preds = (probs >= threshold).astype(int)

    accuracy = float(np.mean(preds == labels_arr))
    f1 = float(f1_score(labels_arr, preds, zero_division=0))
    precision = float(precision_score(labels_arr, preds, zero_division=0))
    recall = float(recall_score(labels_arr, preds, zero_division=0))

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
    }

    return metrics, probs, labels_arr


def optimize_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    min_thresh: float = 0.1,
    max_thresh: float = 0.9,
    steps: int = 81,
) -> tuple[float, float]:
    """Find the threshold that maximizes F1 score."""
    best_f1 = -1.0
    best_thresh = 0.5

    for thresh in np.linspace(min_thresh, max_thresh, steps):
        preds = (probs >= thresh).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    return best_thresh, best_f1


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
    batch_size: int = 16
    grad_accum_steps: int = 2  # effective batch = 32
    lr_head_frozen: float = 1e-3
    lr_encoder: float = 5e-5
    lr_head_unfrozen: float = 5e-4
    hidden_dim: int = 256
    dropout: float = 0.1
    patience: int = 10
    use_scheduler: bool = True
    device: str = "cpu"
    # Two-phase unfrozen: drop encoder LR after N epochs
    lr_encoder_phase2: float | None = None  # None = no drop
    lr_drop_epoch: int = 3  # switch LR after this many unfrozen epochs


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_examples(data_path: str, use_context: bool = False) -> list[DecisionExample]:
    """Load all examples from JSONL file.

    If use_context=True, uses 'text_with_context' field (prev [SEP] target [SEP] next)
    instead of bare 'sentence'. Falls back to 'sentence' if context field missing.
    """
    examples = []
    n_with_context = 0
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            if use_context and "text_with_context" in rec:
                text = rec["text_with_context"]
                n_with_context += 1
            else:
                text = rec["sentence"]
            examples.append(
                DecisionExample(
                    entry_id=rec["entry_id"],
                    sentence=text,
                    label=rec["is_decision"],
                )
            )
    if use_context:
        logger.info(f"Using context windows: {n_with_context}/{len(examples)} examples have context")
    return examples


def compute_pos_weight(
    examples: list[DecisionExample], clamp: float = 30.0,
) -> float:
    """Compute positive class weight from label distribution."""
    n_pos = sum(1 for ex in examples if ex.label == 1)
    n_neg = len(examples) - n_pos
    if n_pos == 0:
        logger.warning("No positive examples! Using pos_weight=1.0")
        return 1.0

    weight = n_neg / n_pos
    weight = min(weight, clamp)
    logger.info(
        f"Class balance: {n_pos} positive ({100*n_pos/len(examples):.1f}%), "
        f"{n_neg} negative ({100*n_neg/len(examples):.1f}%)"
    )
    logger.info(f"pos_weight: {weight:.2f}")
    return weight


def build_loss_fn(
    loss_type: str,
    pos_weight: float,
    device: str,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> nn.Module:
    """Build loss function based on type."""
    if loss_type == "focal":
        alpha = focal_alpha if focal_alpha is not None else pos_weight
        loss_fn = FocalLoss(alpha=alpha, gamma=focal_gamma)
        logger.info(f"Loss: FocalLoss(alpha={alpha:.2f}, gamma={focal_gamma:.1f})")
    else:
        pw = torch.tensor([pos_weight])
        if device == "cuda":
            pw = pw.to("cuda")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        logger.info(f"Loss: BCEWithLogitsLoss(pos_weight={pos_weight:.2f})")
    return loss_fn


# ─── Cross-Validation ────────────────────────────────────────────────────────


def run_cv_fold(
    fold_idx: int,
    train_examples: list[DecisionExample],
    val_examples: list[DecisionExample],
    config: TrainConfig,
    loss_fn: nn.Module,
) -> dict:
    logger.info(f"  Fold {fold_idx}: {len(train_examples)} train, {len(val_examples)} val")

    encoder, embed_dim = build_encoder(config.base_model, config.device)
    head = BinaryHead(
        input_dim=embed_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
    )
    model = DecisionClassifier(encoder, head)

    if config.device == "cuda":
        head = head.to("cuda")

    train_loader = DataLoader(
        DecisionDataset(train_examples), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        DecisionDataset(val_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

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
        val_metrics, val_probs, val_labels = evaluate(
            model, val_loader, loss_fn, config.device,
        )

        # Optimize threshold on validation set
        opt_thresh, opt_f1 = optimize_threshold(val_probs, val_labels)

        if opt_f1 > best_val_f1:
            best_val_f1 = opt_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == config.epochs_frozen - 1:
            logger.info(
                f"    [Frozen] Epoch {epoch+1}/{config.epochs_frozen}: "
                f"loss={train_loss:.4f} f1={opt_f1:.4f} "
                f"thresh={opt_thresh:.2f}"
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
        val_metrics, val_probs, val_labels = evaluate(
            model, val_loader, loss_fn, config.device,
        )

        opt_thresh, opt_f1 = optimize_threshold(val_probs, val_labels)

        if opt_f1 > best_val_f1:
            best_val_f1 = opt_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 3 == 0 or epoch == config.epochs_unfrozen - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"    [Unfrozen] Epoch {epoch+1}/{config.epochs_unfrozen}: "
                f"loss={train_loss:.4f} f1={opt_f1:.4f} "
                f"thresh={opt_thresh:.2f} lr={cur_lr:.2e}"
            )

        if patience_counter >= config.patience:
            logger.info(f"    Early stopping at epoch {epoch+1} (unfrozen phase)")
            break

    return {"f1": best_val_f1, "threshold": opt_thresh}


# ─── Full Training ───────────────────────────────────────────────────────────


def _save_checkpoint(
    output_dir: Path,
    phase: str,
    epoch: int,
    head_state: dict,
    encoder_state: dict | None,
    optimizer_state: dict,
    scheduler_state: dict | None,
    best_val_f1: float,
    best_thresh: float,
    best_head_state: dict,
    best_encoder_state: dict | None,
    patience_counter: int,
    frozen_complete: bool = False,
) -> None:
    """Save a training checkpoint that can be resumed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "phase": phase,
        "epoch": epoch,
        "head_state": head_state,
        "encoder_state": encoder_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "best_val_f1": best_val_f1,
        "best_thresh": best_thresh,
        "best_head_state": best_head_state,
        "best_encoder_state": best_encoder_state,
        "patience_counter": patience_counter,
        "frozen_complete": frozen_complete,
    }
    ckpt_path = output_dir / "checkpoint.pt"
    tmp_path = output_dir / "checkpoint.pt.tmp"
    torch.save(ckpt, str(tmp_path))
    tmp_path.rename(ckpt_path)
    logger.info(f"  Checkpoint saved: phase={phase} epoch={epoch+1}")


def _load_checkpoint(output_dir: Path) -> dict | None:
    """Load a training checkpoint if it exists."""
    ckpt_path = output_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    logger.info(
        f"  Resuming from checkpoint: phase={ckpt['phase']} "
        f"epoch={ckpt['epoch']+1} best_f1={ckpt['best_val_f1']:.4f}"
    )
    return ckpt


def train_final_model(
    train_examples: list[DecisionExample],
    val_examples: list[DecisionExample],
    config: TrainConfig,
    loss_fn: nn.Module,
    output_dir: Path,
    loss_name: str = "BCEWithLogitsLoss",
    dashboard: "TrainingDashboard | None" = None,
    resume: bool = False,
) -> dict:
    """Train final model with early stopping and checkpointing. Returns config_dict."""
    logger.info(
        f"Training final model on {len(train_examples)} examples "
        f"(val: {len(val_examples)}) with base model '{config.base_model}'..."
    )

    encoder, embed_dim = build_encoder(config.base_model, config.device)
    head = BinaryHead(
        input_dim=embed_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
    )
    model = DecisionClassifier(encoder, head)

    if config.device == "cuda":
        head = head.to("cuda")

    train_loader = DataLoader(
        DecisionDataset(train_examples), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        DecisionDataset(val_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    # Check for checkpoint to resume from
    ckpt = _load_checkpoint(output_dir) if resume else None
    skip_frozen = False
    start_unfrozen_epoch = 0

    if ckpt is not None:
        # Restore best states from checkpoint
        best_val_f1_from_ckpt = ckpt["best_val_f1"]
        best_thresh_from_ckpt = ckpt["best_thresh"]
        best_head_state_from_ckpt = ckpt["best_head_state"]
        best_encoder_state_from_ckpt = ckpt["best_encoder_state"]
        patience_from_ckpt = ckpt["patience_counter"]

        if ckpt["frozen_complete"] or ckpt["phase"] == "unfrozen":
            skip_frozen = True
            # Restore head state from checkpoint
            head.load_state_dict(ckpt["head_state"])
            if ckpt["phase"] == "unfrozen" and ckpt["encoder_state"] is not None:
                encoder[0].auto_model.load_state_dict(ckpt["encoder_state"])
                start_unfrozen_epoch = ckpt["epoch"] + 1
            logger.info(f"  Skipping frozen phase (already complete)")
        else:
            # Resuming mid-frozen phase — just restart frozen (it's fast)
            logger.info(f"  Checkpoint was mid-frozen phase, restarting frozen (fast)")
            ckpt = None  # treat as fresh start

    # Phase A: Frozen encoder
    best_frozen_f1 = -1.0
    best_frozen_head_state = None
    best_frozen_thresh = 0.5
    frozen_patience_counter = 0
    frozen_patience = min(config.patience, 8)

    if not skip_frozen:
        for param in encoder.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr_head_frozen, weight_decay=0.01)

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
            val_metrics, val_probs, val_labels = evaluate(
                model, val_loader, loss_fn, config.device,
            )

            opt_thresh, opt_f1 = optimize_threshold(val_probs, val_labels)
            is_best = opt_f1 > best_frozen_f1

            if is_best:
                best_frozen_f1 = opt_f1
                best_frozen_head_state = copy.deepcopy(head.state_dict())
                best_frozen_thresh = opt_thresh
                frozen_patience_counter = 0
                logger.info(
                    f"  [Final/Frozen] Epoch {epoch+1}: NEW BEST loss={train_loss:.4f} "
                    f"f1={opt_f1:.4f} thresh={opt_thresh:.2f}"
                )
            else:
                frozen_patience_counter += 1
                if epoch % 2 == 0 or epoch == config.epochs_frozen - 1:
                    logger.info(
                        f"  [Final/Frozen] Epoch {epoch+1}: loss={train_loss:.4f} "
                        f"f1={opt_f1:.4f} thresh={opt_thresh:.2f}"
                    )

            if dashboard:
                dashboard.log_epoch(
                    epoch,
                    {
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_f1": opt_f1,
                        "threshold": opt_thresh,
                        "lr": config.lr_head_frozen,
                        "precision": val_metrics["precision"],
                        "recall": val_metrics["recall"],
                    },
                    is_best=is_best,
                )

            if frozen_patience_counter >= frozen_patience:
                logger.info(f"  Frozen early stopping at epoch {epoch+1}")
                break

        # Restore best frozen head state
        if best_frozen_head_state is not None:
            head.load_state_dict(best_frozen_head_state)
            logger.info(f"  Restored best frozen head state (f1={best_frozen_f1:.4f})")

    # Initialize best state
    if skip_frozen and ckpt is not None:
        best_val_f1 = best_val_f1_from_ckpt
        best_encoder_state = best_encoder_state_from_ckpt
        best_head_state = best_head_state_from_ckpt
        best_thresh = best_thresh_from_ckpt
    else:
        best_val_f1 = best_frozen_f1
        best_encoder_state = None
        best_head_state = copy.deepcopy(head.state_dict())
        best_thresh = best_frozen_thresh

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

        # If resuming unfrozen phase, restore optimizer/scheduler and fast-forward
        if ckpt is not None and start_unfrozen_epoch > 0:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            # Move optimizer tensors to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(config.device)
            if scheduler is not None and ckpt["scheduler_state"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            patience_counter = patience_from_ckpt
            logger.info(f"  Resuming unfrozen phase at epoch {start_unfrozen_epoch + 1}")
        else:
            patience_counter = 0
            start_unfrozen_epoch = 0

        if dashboard:
            dashboard.start_phase("Fine-tuning Encoder", config.epochs_unfrozen)

        for epoch in range(start_unfrozen_epoch, config.epochs_unfrozen):
            if dashboard:
                dashboard.start_epoch(epoch, len(train_loader))

            batch_cb = (lambda idx, loss: dashboard.log_batch(idx, loss)) if dashboard else None
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, config.device,
                encoder_frozen=False, grad_accum_steps=config.grad_accum_steps,
                scheduler=scheduler, batch_callback=batch_cb,
            )
            val_metrics, val_probs, val_labels = evaluate(
                model, val_loader, loss_fn, config.device,
            )

            opt_thresh, opt_f1 = optimize_threshold(val_probs, val_labels)
            cur_lr = optimizer.param_groups[0]["lr"]
            is_best = opt_f1 > best_val_f1

            if is_best:
                best_val_f1 = opt_f1
                best_thresh = opt_thresh
                patience_counter = 0
                best_head_state = copy.deepcopy(head.state_dict())
                best_encoder_state = copy.deepcopy(encoder[0].auto_model.state_dict())
                logger.info(
                    f"  [Final/Unfrozen] Epoch {epoch+1}: NEW BEST "
                    f"f1={opt_f1:.4f} thresh={opt_thresh:.2f}"
                )
            else:
                patience_counter += 1

            # Two-phase LR drop: reduce encoder LR after lr_drop_epoch epochs
            if (
                config.lr_encoder_phase2 is not None
                and epoch + 1 == config.lr_drop_epoch
            ):
                old_lr = optimizer.param_groups[0]["lr"]
                optimizer.param_groups[0]["lr"] = config.lr_encoder_phase2
                logger.info(
                    f"  [LR Drop] Encoder LR: {old_lr:.2e} → {config.lr_encoder_phase2:.2e} "
                    f"at epoch {epoch+1}"
                )
                # Reset patience so the model gets a fair chance at the new LR
                patience_counter = 0

            if epoch % 5 == 0 or epoch == config.epochs_unfrozen - 1:
                logger.info(
                    f"  [Final/Unfrozen] Epoch {epoch+1}: loss={train_loss:.4f} "
                    f"f1={opt_f1:.4f} thresh={opt_thresh:.2f} "
                    f"lr={cur_lr:.2e}"
                )

            if dashboard:
                dashboard.log_epoch(
                    epoch,
                    {
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_f1": opt_f1,
                        "threshold": opt_thresh,
                        "lr": cur_lr,
                        "precision": val_metrics["precision"],
                        "recall": val_metrics["recall"],
                    },
                    is_best=is_best,
                )

            # Save checkpoint after each epoch
            _save_checkpoint(
                output_dir,
                phase="unfrozen",
                epoch=epoch,
                head_state=copy.deepcopy(head.state_dict()),
                encoder_state=copy.deepcopy(encoder[0].auto_model.state_dict()),
                optimizer_state=copy.deepcopy(optimizer.state_dict()),
                scheduler_state=copy.deepcopy(scheduler.state_dict()) if scheduler else None,
                best_val_f1=best_val_f1,
                best_thresh=best_thresh,
                best_head_state=best_head_state,
                best_encoder_state=best_encoder_state,
                patience_counter=patience_counter,
                frozen_complete=True,
            )

            if patience_counter >= config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best checkpoint
        if best_encoder_state is not None:
            encoder[0].auto_model.load_state_dict(best_encoder_state)
        if best_head_state is not None:
            head.load_state_dict(best_head_state)
        logger.info(f"  Restored best checkpoint (f1={best_val_f1:.4f})")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(str(model_dir))
    logger.info(f"  Saved encoder to {model_dir}")

    head_path = output_dir / "classification_head.pt"
    torch.save(head.cpu().state_dict(), str(head_path))
    logger.info(f"  Saved classification head to {head_path}")

    # Save decision threshold
    thresh_path = output_dir / "decision_threshold.json"
    with thresh_path.open("w") as f:
        json.dump({"threshold": best_thresh, "val_f1": best_val_f1}, f, indent=2)
    logger.info(f"  Saved threshold ({best_thresh:.3f}) to {thresh_path}")

    # Save training config
    model_short = config.base_model.split("/")[-1].lower()
    config_dict = {
        "task": "binary-classification",
        "input_dim": embed_dim,
        "hidden_dim": config.hidden_dim,
        "output_dim": 1,
        "base_model": config.base_model,
        "prompt_version": "decision-finetuned-v1",
        "model_version": f"finetuned-decision-{model_short}-v1",
        "epochs_frozen": config.epochs_frozen,
        "epochs_unfrozen": config.epochs_unfrozen,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "effective_batch_size": config.batch_size * config.grad_accum_steps,
        "lr_encoder": config.lr_encoder,
        "lr_head_frozen": config.lr_head_frozen,
        "lr_head_unfrozen": config.lr_head_unfrozen,
        "dropout": config.dropout,
        "loss": loss_name,
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "best_val_f1": best_val_f1,
        "best_threshold": best_thresh,
    }
    config_path = output_dir / "training_config.json"
    with config_path.open("w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"  Saved config to {config_path}")

    return config_dict


def main():
    parser = argparse.ArgumentParser(description="Train decision detector")
    parser.add_argument("--data", required=True, help="Path to decision_training_data.jsonl")
    parser.add_argument("--splits", required=True, help="Path to decision_splits.json")
    parser.add_argument("--output", default="./decision-classifier-mpnet-v1", help="Output directory")
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
    parser.add_argument("--pos-weight-clamp", type=float, default=30.0)
    parser.add_argument(
        "--loss", choices=["bce", "focal"], default="bce",
        help="Loss function: 'bce' (BCEWithLogitsLoss) or 'focal' (FocalLoss)",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss gamma (focusing parameter). Only used with --loss focal.",
    )
    parser.add_argument(
        "--focal-alpha", type=float, default=None,
        help="Focal loss alpha (positive class weight). Defaults to computed pos_weight. Only used with --loss focal.",
    )
    parser.add_argument(
        "--neg-ratio", type=float, default=0,
        help="Downsample negatives to N:1 neg:pos ratio (0 = no downsampling). "
             "E.g., --neg-ratio 10 keeps all positives + 10x negatives.",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging for cloud-based tracking",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable rich terminal dashboard (use plain logging)",
    )
    parser.add_argument(
        "--no-scheduler", action="store_true",
        help="Disable cosine warmup LR scheduler (use flat LR for encoder)",
    )
    parser.add_argument(
        "--lr-encoder-phase2", type=float, default=None,
        help="Drop encoder LR to this value after --lr-drop-epoch unfrozen epochs. "
             "E.g., --lr-encoder 2e-5 --lr-encoder-phase2 5e-6 --lr-drop-epoch 3",
    )
    parser.add_argument(
        "--lr-drop-epoch", type=int, default=3,
        help="Unfrozen epoch at which to drop encoder LR to --lr-encoder-phase2 (default: 3)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from checkpoint if one exists in --output dir",
    )
    parser.add_argument(
        "--use-context", action="store_true",
        help="Use context windows (prev [SEP] target [SEP] next) instead of bare sentences",
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
        use_scheduler=not args.no_scheduler,
        lr_encoder_phase2=args.lr_encoder_phase2,
        lr_drop_epoch=args.lr_drop_epoch,
    )

    eff_batch = config.batch_size * config.grad_accum_steps
    logger.info(f"Device: {config.device}")
    logger.info(f"Base model: {config.base_model}")
    lr_phase2_str = ""
    if config.lr_encoder_phase2 is not None:
        lr_phase2_str = f", LR drop: {config.lr_encoder:.0e}→{config.lr_encoder_phase2:.0e} @ epoch {config.lr_drop_epoch}"
    logger.info(
        f"Config: frozen={config.epochs_frozen}, unfrozen={config.epochs_unfrozen}, "
        f"batch={config.batch_size}x{config.grad_accum_steps}={eff_batch}, "
        f"hidden={config.hidden_dim}, dropout={config.dropout}, "
        f"scheduler={'cosine_warmup' if config.use_scheduler else 'none (flat LR)'}"
        f"{lr_phase2_str}"
    )

    # Load data
    all_examples = load_examples(args.data, use_context=args.use_context)
    logger.info(f"Loaded {len(all_examples)} examples")

    with open(args.splits) as f:
        splits = json.load(f)

    test_ids = set(splits["test_ids"])
    train_ids = set(splits["train_ids"])

    test_examples = [ex for ex in all_examples if ex.entry_id in test_ids]
    train_examples = [ex for ex in all_examples if ex.entry_id in train_ids]

    # Downsample negatives if requested
    if args.neg_ratio > 0:
        import random
        pos = [ex for ex in train_examples if ex.label == 1]
        neg = [ex for ex in train_examples if ex.label == 0]
        max_neg = int(len(pos) * args.neg_ratio)
        if max_neg < len(neg):
            random.seed(42)
            neg = random.sample(neg, max_neg)
            logger.info(
                f"Downsampled negatives: {len(pos)} pos + {len(neg)} neg "
                f"({args.neg_ratio:.0f}:1 ratio, {len(pos)+len(neg)} total)"
            )
        train_examples = pos + neg
        random.shuffle(train_examples)

    n_train_pos = sum(1 for ex in train_examples if ex.label == 1)
    n_test_pos = sum(1 for ex in test_examples if ex.label == 1)
    logger.info(
        f"Train: {len(train_examples)} ({n_train_pos} positive, "
        f"{100*n_train_pos/len(train_examples):.1f}%)"
    )
    logger.info(
        f"Test: {len(test_examples)} ({n_test_pos} positive, "
        f"{100*n_test_pos/len(test_examples):.1f}%)"
    )

    # Compute pos_weight and build loss function
    pos_weight = compute_pos_weight(train_examples, clamp=args.pos_weight_clamp)
    loss_fn = build_loss_fn(
        args.loss, pos_weight, config.device,
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
    )
    loss_name = type(loss_fn).__name__

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
                fold_idx, fold_train, fold_val, config, loss_fn,
            )
            cv_results.append(fold_metrics)

        logger.info("=" * 60)
        logger.info("Cross-validation results:")
        f1_values = [r["f1"] for r in cv_results]
        logger.info(f"  {'f1':20s}: {np.mean(f1_values):.4f} +/- {np.std(f1_values):.4f}")

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
        run_name = f"decision-{model_short}-{int(time.time())}"
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
            "train_count": len(final_train),
            "val_count": len(final_val),
            "device": config.device,
        }
        dashboard = TrainingDashboard(
            project="corpus-mcp-training",
            run_name=run_name,
            task_type="decision-classifier",
            config=dashboard_config,
            use_wandb=args.wandb,
        )
        dashboard.start()

    start_time = time.time()
    try:
        config_dict = train_final_model(
            final_train, final_val, config, loss_fn, output_dir,
            loss_name=loss_name, dashboard=dashboard, resume=args.resume,
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
    head = BinaryHead(
        input_dim=saved_embed_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
    )
    head.load_state_dict(
        torch.load(str(output_dir / "classification_head.pt"), map_location="cpu", weights_only=True)
    )
    model = DecisionClassifier(encoder, head)
    model.head.eval()

    # Load saved threshold
    with (output_dir / "decision_threshold.json").open() as f:
        thresh_data = json.load(f)
    saved_thresh = thresh_data["threshold"]
    logger.info(f"Using saved threshold: {saved_thresh:.3f}")

    test_loader = DataLoader(
        DecisionDataset(test_examples), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    test_loss_fn = build_loss_fn(
        args.loss, pos_weight, config.device,
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
    )
    test_metrics, test_probs, test_labels = evaluate(
        model, test_loader, test_loss_fn, "cpu", threshold=saved_thresh,
    )

    # Also optimize threshold on test set for comparison
    test_opt_thresh, test_opt_f1 = optimize_threshold(test_probs, test_labels)

    logger.info("Test results (saved threshold):")
    for key in ["f1", "precision", "recall", "accuracy"]:
        logger.info(f"  {key:20s}: {test_metrics[key]:.4f}")
    logger.info(f"  {'threshold':20s}: {saved_thresh:.3f}")
    logger.info(f"  Test-optimal threshold: {test_opt_thresh:.3f} (F1={test_opt_f1:.4f})")

    # Classification report
    test_preds = (test_probs >= saved_thresh).astype(int)
    logger.info("\nClassification Report:")
    report = classification_report(
        test_labels, test_preds,
        target_names=["not_decision", "is_decision"],
        zero_division=0,
    )
    logger.info("\n" + report)

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    logger.info("Confusion Matrix:")
    logger.info(f"{'':15s} {'pred_0':>8s} {'pred_1':>8s}")
    logger.info(f"  {'actual_0':13s} {cm[0][0]:8d} {cm[0][1]:8d}")
    logger.info(f"  {'actual_1':13s} {cm[1][0]:8d} {cm[1][1]:8d}")

    # Save eval report
    eval_report = {
        "test_count": len(test_examples),
        "test_positive": int(n_test_pos),
        "test_positive_rate": round(100 * n_test_pos / len(test_examples), 2),
        "test_metrics": test_metrics,
        "saved_threshold": saved_thresh,
        "test_optimal_threshold": test_opt_thresh,
        "test_optimal_f1": test_opt_f1,
        "confusion_matrix": cm.tolist(),
        "quality_gates": {
            "f1_threshold": 0.50,
            "precision_threshold": 0.40,
            "recall_threshold": 0.40,
            "f1_passed": test_metrics["f1"] > 0.50,
            "precision_passed": test_metrics["precision"] > 0.40,
            "recall_passed": test_metrics["recall"] > 0.40,
        },
    }

    eval_path = output_dir / "eval_report.json"
    with eval_path.open("w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"\nSaved evaluation report to {eval_path}")

    # Quality gate check
    gates = eval_report["quality_gates"]
    all_passed = gates["f1_passed"] and gates["precision_passed"] and gates["recall_passed"]

    if all_passed:
        logger.info("ALL QUALITY GATES PASSED")
    else:
        logger.warning("QUALITY GATES FAILED:")
        if not gates["f1_passed"]:
            logger.warning(f"  F1 {test_metrics['f1']:.4f} <= 0.50")
        if not gates["precision_passed"]:
            logger.warning(f"  Precision {test_metrics['precision']:.4f} <= 0.40")
        if not gates["recall_passed"]:
            logger.warning(f"  Recall {test_metrics['recall']:.4f} <= 0.40")


if __name__ == "__main__":
    main()
