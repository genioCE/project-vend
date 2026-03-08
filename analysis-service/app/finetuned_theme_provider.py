"""Fine-tuned transformer-based theme classifier.

Uses a sentence-transformer encoder fine-tuned on Claude-labeled journal themes
with a multi-label classification head producing theme predictions.
CPU inference, no API calls, ~150ms per entry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .corpus_utils.text_processing import chunk_text

logger = logging.getLogger("theme-provider.finetuned")


class _ClassificationHead(nn.Module):
    """Classification head. MLP when hidden_dim>0, linear otherwise."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 150):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(input_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FinetunedThemeProvider:
    """Fine-tuned theme classifier. Zero API calls, CPU inference.

    Predicts canonical theme labels for journal entries using a multi-label
    classifier trained on Claude-labeled theme data.
    """

    prompt_version = "theme-finetuned-v1"

    def __init__(self, model_dir: str) -> None:
        from sentence_transformers import SentenceTransformer

        model_path = Path(model_dir)
        encoder_path = model_path / "model"
        head_path = model_path / "classification_head.pt"
        config_path = model_path / "training_config.json"
        label_index_path = model_path / "theme_label_index.json"
        threshold_path = model_path / "theme_thresholds.json"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        if not head_path.exists():
            raise FileNotFoundError(f"Classification head not found at {head_path}")
        if not label_index_path.exists():
            raise FileNotFoundError(f"Label index not found at {label_index_path}")

        # Load label index
        with label_index_path.open() as f:
            self._label_index: dict[str, int] = json.load(f)
        self._n_labels = len(self._label_index)
        # Ordered list of label names by index
        self._label_names = sorted(self._label_index.keys(), key=lambda k: self._label_index[k])

        # Load thresholds (default to 0.5 if not available)
        if threshold_path.exists():
            with threshold_path.open() as f:
                threshold_dict = json.load(f)
            self._thresholds = torch.tensor(
                [threshold_dict.get(label, 0.5) for label in self._label_names],
                dtype=torch.float32,
            )
        else:
            self._thresholds = torch.full((self._n_labels,), 0.5)

        # Load training config for architecture params
        input_dim = 768
        hidden_dim = 256
        if config_path.exists():
            with config_path.open() as f:
                config = json.load(f)
                input_dim = config.get("input_dim", 768)
                hidden_dim = config.get("hidden_dim", 256)
                self._model_version = config.get("model_version", "finetuned-theme-v1")
                logger.info(f"Config: base_model={config.get('base_model', 'unknown')}, "
                            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
                            f"n_labels={self._n_labels}")
        else:
            self._model_version = "finetuned-theme-v1"

        # Load encoder
        self._encoder = SentenceTransformer(str(encoder_path))

        # Load classification head
        self._head = _ClassificationHead(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=self._n_labels,
        )
        self._head.load_state_dict(
            torch.load(str(head_path), map_location="cpu", weights_only=True)
        )
        self._head.eval()

        logger.info(
            "finetuned_theme_provider_loaded",
            extra={
                "model_dir": model_dir,
                "model_version": self._model_version,
                "n_labels": self._n_labels,
            },
        )

    def predict_themes(
        self,
        entry_text: str,
        min_themes: int = 2,
        max_themes: int = 8,
    ) -> list[str]:
        """Predict themes for entry text. Returns list of canonical theme strings.

        Guarantees between min_themes and max_themes labels by adjusting
        thresholds if needed.
        """
        # Chunk text same way as training
        text_chunks = chunk_text(entry_text, max_words=500)
        if not text_chunks:
            text_chunks = [entry_text[:1000]]

        with torch.no_grad():
            # Encode chunks and mean-pool
            embeddings = self._encoder.encode(
                text_chunks,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            # Handle various return types from encode()
            if isinstance(embeddings, list):
                embeddings = torch.stack(
                    [e if isinstance(e, torch.Tensor) else torch.tensor(e) for e in embeddings]
                ).float()
            elif not isinstance(embeddings, torch.Tensor):
                import numpy as np
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.from_numpy(embeddings).float()
                else:
                    embeddings = torch.tensor(embeddings, dtype=torch.float32)
            else:
                embeddings = embeddings.float()

            pooled = embeddings.mean(dim=0, keepdim=True)  # (1, dim)
            logits = self._head(pooled)  # (1, n_labels)
            probs = torch.sigmoid(logits).squeeze(0)  # (n_labels,)

        # Apply per-label thresholds
        predicted_mask = probs >= self._thresholds
        predicted_indices = predicted_mask.nonzero(as_tuple=True)[0].tolist()

        # Ensure minimum themes: if too few, lower thresholds by taking top-k
        if len(predicted_indices) < min_themes:
            _, top_indices = probs.topk(min_themes)
            predicted_indices = top_indices.tolist()

        # Cap at max_themes: keep highest probability ones
        if len(predicted_indices) > max_themes:
            # Sort by probability descending, keep top max_themes
            scored = [(idx, probs[idx].item()) for idx in predicted_indices]
            scored.sort(key=lambda x: -x[1])
            predicted_indices = [idx for idx, _ in scored[:max_themes]]

        # Sort by probability descending for output
        scored = [(idx, probs[idx].item()) for idx in predicted_indices]
        scored.sort(key=lambda x: -x[1])

        themes = [self._label_names[idx] for idx, _ in scored]
        return themes
