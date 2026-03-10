"""Fine-tuned transformer-based decision detector.

Uses a sentence-transformer encoder fine-tuned on Claude-labeled decision data
with a binary classification head to detect decision/commitment sentences.

CPU inference, no API calls, ~200ms per entry.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("decision-provider.finetuned")


class _BinaryHead(nn.Module):
    """MLP: Linear→LayerNorm→GELU→Dropout→Linear(1)."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FinetunedDecisionProvider:
    """Fine-tuned decision detector. Zero API calls, CPU inference.

    Splits entry text into sentences, classifies each as decision/not-decision
    using a sentence-transformer encoder + binary head trained on Claude labels.
    """

    prompt_version = "decision-finetuned-v1"

    def __init__(self, model_dir: str) -> None:
        from sentence_transformers import SentenceTransformer

        model_path = Path(model_dir)
        encoder_path = model_path / "model"
        head_path = model_path / "classification_head.pt"
        config_path = model_path / "training_config.json"
        threshold_path = model_path / "decision_threshold.json"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        if not head_path.exists():
            raise FileNotFoundError(f"Classification head not found at {head_path}")

        # Load threshold
        self._threshold = 0.5
        if threshold_path.exists():
            with threshold_path.open() as f:
                thresh_data = json.load(f)
                self._threshold = thresh_data.get("threshold", 0.5)
                logger.info(f"Decision threshold: {self._threshold:.3f}")
        else:
            logger.warning("No threshold file found, using default 0.5")

        # Load training config for architecture params
        input_dim = 768
        hidden_dim = 256
        if config_path.exists():
            with config_path.open() as f:
                config = json.load(f)
                input_dim = config.get("input_dim", 768)
                hidden_dim = config.get("hidden_dim", 256)
                self._model_version = config.get("model_version", "finetuned-decision-v1")
                logger.info(
                    f"Config: base_model={config.get('base_model', 'unknown')}, "
                    f"input_dim={input_dim}, hidden_dim={hidden_dim}"
                )
        else:
            self._model_version = "finetuned-decision-v1"

        # Load encoder
        self._encoder = SentenceTransformer(str(encoder_path))

        # Load binary head
        self._head = _BinaryHead(input_dim=input_dim, hidden_dim=hidden_dim)
        self._head.load_state_dict(
            torch.load(str(head_path), map_location="cpu", weights_only=True)
        )
        self._head.eval()

        logger.info(
            "finetuned_decision_provider_loaded",
            extra={
                "model_dir": model_dir,
                "model_version": self._model_version,
                "threshold": self._threshold,
            },
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _build_context_inputs(
        sentences: list[str],
    ) -> list[tuple[str, str]]:
        """Build context-windowed inputs matching training format.

        Returns list of (bare_sentence, context_input) where context_input
        is ``[prev] [SEP] target [SEP] [next]``.
        """
        results = []
        for i, sent in enumerate(sentences):
            parts = []
            if i > 0:
                parts.append(sentences[i - 1])
            parts.append(sent)
            if i < len(sentences) - 1:
                parts.append(sentences[i + 1])
            results.append((sent, " [SEP] ".join(parts)))
        return results

    def extract_decisions(
        self,
        entry_text: str,
        max_decisions: int = 12,
        min_sentence_words: int = 4,
    ) -> list[str]:
        """Extract decision sentences from entry text.

        Args:
            entry_text: Full entry text.
            max_decisions: Maximum number of decisions to return.
            min_sentence_words: Minimum words for a sentence to be considered.

        Returns:
            List of decision sentence strings, ranked by confidence.
        """
        normalized = " ".join(entry_text.split())
        sentences = self._split_sentences(normalized)

        # Filter short sentences, keeping index mapping for context
        valid_indices = [
            i for i, s in enumerate(sentences) if len(s.split()) >= min_sentence_words
        ]
        if not valid_indices:
            return []

        valid_sentences = [sentences[i] for i in valid_indices]

        # Build context windows: [prev] [SEP] target [SEP] [next]
        context_inputs = []
        for idx in valid_indices:
            parts = []
            if idx > 0:
                parts.append(sentences[idx - 1])
            parts.append(sentences[idx])
            if idx < len(sentences) - 1:
                parts.append(sentences[idx + 1])
            context_inputs.append(" [SEP] ".join(parts))

        # Batch encode and classify
        with torch.no_grad():
            embeddings = self._encoder.encode(
                context_inputs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            import numpy as np
            embeddings = torch.from_numpy(np.asarray(embeddings)).float()

            logits = self._head(embeddings)
            probs = torch.sigmoid(logits).tolist()

        # Collect decisions above threshold, sorted by confidence
        scored = [
            (sent, prob)
            for sent, prob in zip(valid_sentences, probs)
            if prob >= self._threshold
        ]
        scored.sort(key=lambda x: -x[1])

        return [sent for sent, _ in scored[:max_decisions]]
