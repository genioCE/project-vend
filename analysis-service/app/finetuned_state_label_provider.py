"""Fine-tuned transformer-based state label provider.

Uses a sentence-transformer encoder fine-tuned on Claude-labeled journal data
with a regression head producing 8-dimension psychological state scores.
CPU inference, no API calls, ~100ms per entry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .corpus_utils.text_processing import chunk_text
from .state_label_provider import (
    ALL_DIMENSIONS,
    DIMENSION_ANCHORS,
    StateLabelGeneration,
)

logger = logging.getLogger("state-label-provider.finetuned")


class _RegressionHead(nn.Module):
    """Shared MLP: Linear(768→256) → LayerNorm → GELU → Dropout → Linear(256→8)."""

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


class FinetunedStateLabelProvider:
    """Fine-tuned state label provider. Zero API calls, CPU inference."""

    prompt_version = "state-label-finetuned-v1"

    def __init__(self, model_dir: str) -> None:
        from sentence_transformers import SentenceTransformer

        model_path = Path(model_dir)
        encoder_path = model_path / "model"
        head_path = model_path / "regression_head.pt"
        config_path = model_path / "training_config.json"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        if not head_path.exists():
            raise FileNotFoundError(f"Regression head not found at {head_path}")

        # Load training config for architecture params
        hidden_dim = 256
        if config_path.exists():
            with config_path.open() as f:
                config = json.load(f)
                hidden_dim = config.get("hidden_dim", 256)
                self._model_version = config.get("model_version", "finetuned-mpnet-v1")
        else:
            self._model_version = "finetuned-mpnet-v1"

        self._encoder = SentenceTransformer(str(encoder_path))
        self._head = _RegressionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=8)
        self._head.load_state_dict(
            torch.load(str(head_path), map_location="cpu", weights_only=True)
        )
        self._head.eval()

        logger.info(
            "finetuned_provider_loaded",
            extra={"model_dir": model_dir, "model_version": self._model_version},
        )

    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunk_ids: list[str],
        source_file: str | None,
    ) -> StateLabelGeneration:
        # Chunk text the same way as training data preparation
        text_chunks = chunk_text(entry_text, max_words=500)
        if not text_chunks:
            text_chunks = [entry_text[:1000]]  # fallback for very short/empty text

        # Encode each chunk, mean-pool across chunks
        with torch.no_grad():
            embeddings = self._encoder.encode(
                text_chunks,
                show_progress_bar=False,
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
            # embeddings shape: (n_chunks, 768)
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, dtype=torch.float32)

            pooled = embeddings.mean(dim=0, keepdim=True)  # (1, 768)
            logits = self._head(pooled)  # (1, 8)
            scores = torch.tanh(logits).squeeze(0).tolist()  # 8 floats in [-1, 1]

        dimensions = []
        for i, dim_name in enumerate(ALL_DIMENSIONS):
            score = round(max(-1.0, min(1.0, scores[i])), 4)
            low_anchor, high_anchor = DIMENSION_ANCHORS[dim_name]

            if score <= -0.25:
                label = low_anchor
            elif score >= 0.25:
                label = high_anchor
            else:
                label = f"between {low_anchor} and {high_anchor}"

            dimensions.append({
                "dimension": dim_name,
                "score": score,
                "label": label,
                "rationale": f"Fine-tuned model prediction: {label}",
            })

        return StateLabelGeneration(
            dimensions=dimensions,
            observed_signals=[],  # Neural model — no discrete signals
            model_version=self._model_version,
            prompt_version=self.prompt_version,
            provider="finetuned",
            mock=False,
        )
