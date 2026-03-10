"""Fine-tuned transformer-based entity type classifier.

Uses a sentence-transformer encoder fine-tuned on Claude-labeled entity data
with a 5-class classification head (concept, organization, person, place, spiritual).

Stage 1 (span detection) uses existing regex patterns from local_extractor.
Stage 2 (type classification) uses the finetuned model.

CPU inference, no API calls, ~200ms per entry.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("entity-provider.finetuned")

ENTITY_TYPES = ["concept", "organization", "person", "place", "spiritual"]


class _ClassificationHead(nn.Module):
    """MLP: Linear→LayerNorm→GELU→Dropout→Linear(5)."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 5):
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


class FinetunedEntityProvider:
    """Fine-tuned entity type classifier. Zero API calls, CPU inference.

    Classifies entity candidates (detected via regex) into 5 types using a
    sentence-transformer encoder + classification head trained on Claude labels.
    """

    prompt_version = "entity-finetuned-v1"

    def __init__(self, model_dir: str) -> None:
        from sentence_transformers import SentenceTransformer

        model_path = Path(model_dir)
        encoder_path = model_path / "model"
        head_path = model_path / "classification_head.pt"
        config_path = model_path / "training_config.json"
        type_index_path = model_path / "entity_type_index.json"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        if not head_path.exists():
            raise FileNotFoundError(f"Classification head not found at {head_path}")

        # Load entity type index
        if type_index_path.exists():
            with type_index_path.open() as f:
                self._type_index: dict[str, int] = json.load(f)
        else:
            self._type_index = {t: i for i, t in enumerate(ENTITY_TYPES)}

        self._n_classes = len(self._type_index)
        self._index_to_type = {v: k for k, v in self._type_index.items()}

        # Load training config for architecture params
        input_dim = 768
        hidden_dim = 256
        if config_path.exists():
            with config_path.open() as f:
                config = json.load(f)
                input_dim = config.get("input_dim", 768)
                hidden_dim = config.get("hidden_dim", 256)
                self._model_version = config.get("model_version", "finetuned-entity-v1")
                logger.info(
                    f"Config: base_model={config.get('base_model', 'unknown')}, "
                    f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
                    f"n_classes={self._n_classes}"
                )
        else:
            self._model_version = "finetuned-entity-v1"

        # Load encoder
        self._encoder = SentenceTransformer(str(encoder_path))

        # Load classification head
        self._head = _ClassificationHead(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=self._n_classes,
        )
        self._head.load_state_dict(
            torch.load(str(head_path), map_location="cpu", weights_only=True)
        )
        self._head.eval()

        logger.info(
            "finetuned_entity_provider_loaded",
            extra={
                "model_dir": model_dir,
                "model_version": self._model_version,
                "n_classes": self._n_classes,
            },
        )

    def _find_context(self, text: str, entity_name: str, window: int = 200) -> str:
        """Find entity name in text and return a context window around it."""
        text_lower = text.lower()
        name_lower = entity_name.lower()

        idx = text_lower.find(name_lower)
        if idx == -1:
            pattern = re.escape(name_lower)
            match = re.search(pattern, text_lower)
            if match:
                idx = match.start()
            else:
                # Fallback: use first 200 chars as context
                return text[:window].strip()

        half = window // 2
        start = max(0, idx - half)
        end = min(len(text), idx + len(entity_name) + half)
        return text[start:end].strip()

    def classify_entities(
        self,
        entry_text: str,
        candidates: list[dict] | None = None,
    ) -> list[dict]:
        """Classify entity candidates into types.

        Args:
            entry_text: Full entry text for context extraction.
            candidates: Optional list of {"name": str, "type": str} dicts from
                regex detection. If None, uses extract_entities_local().

        Returns:
            List of {"name": str, "type": str} with finetuned type predictions.
        """
        if candidates is None:
            from .local_extractor import extract_entities_local
            candidates = extract_entities_local(entry_text)

        if not candidates:
            return []

        # Build model inputs: "[ENTITY] name [CONTEXT] window"
        texts = []
        valid_candidates = []
        for ent in candidates:
            name = ent.get("name", "").strip()
            if not name:
                continue
            context = self._find_context(entry_text, name)
            texts.append(f"[ENTITY] {name} [CONTEXT] {context}")
            valid_candidates.append(ent)

        if not texts:
            return candidates  # return original if no valid ones

        # Batch encode and classify
        with torch.no_grad():
            embeddings = self._encoder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
            if not isinstance(embeddings, torch.Tensor):
                import numpy as np
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.from_numpy(embeddings).float()
                else:
                    embeddings = torch.tensor(embeddings, dtype=torch.float32)
            else:
                embeddings = embeddings.float()

            logits = self._head(embeddings)
            pred_indices = torch.argmax(logits, dim=1).tolist()

        # Build result with predicted types
        result = []
        for ent, pred_idx in zip(valid_candidates, pred_indices):
            predicted_type = self._index_to_type.get(pred_idx, "concept")
            result.append({"name": ent["name"], "type": predicted_type})

        return result
