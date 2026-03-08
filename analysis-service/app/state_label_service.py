from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from .models import (
    DimensionConfidence,
    EntryChunk,
    InferredStateLabel,
    ObservedTextSignal,
    Provenance,
    SourceSpan,
    StateConfidence,
    StateDimension,
    StateDimensionProfile,
    StateLabelGenerateRequest,
    StateLabelProcessingMetadata,
    StateLabelRecord,
    StateProfile,
)
from .state_label_provider import (
    ALL_DIMENSIONS,
    DIMENSION_ANCHORS,
    StateLabelGeneration,
    StateLabelProvider,
)
from .state_label_store import SQLiteStateLabelStore, StateLabelStore

STATE_LABEL_SCHEMA_VERSION = "state-label-v1"

logger = logging.getLogger("analysis-service.state-label")


class StateLabelService:
    """Service layer orchestrating state label provider inference + persistence."""

    def __init__(
        self,
        store: StateLabelStore,
        providers: dict[str, StateLabelProvider],
        default_provider_name: str,
        fallback_provider_name: str = "local",
    ):
        self.store = store
        self.providers = providers
        self.default_provider_name = default_provider_name
        self.fallback_provider_name = fallback_provider_name

    def generate_and_persist(self, request: StateLabelGenerateRequest) -> StateLabelRecord:
        existing = self.store.get(request.entry_id)
        if existing and not request.force_regenerate:
            return existing

        provider_name = self._resolve_provider_name(request.provider)
        provider = self.providers[provider_name]

        ordered_chunks = list(request.chunks)
        entry_text = "\n\n".join(chunk.text for chunk in ordered_chunks)
        chunk_ids = [chunk.chunk_id for chunk in ordered_chunks]

        generation = self._generate_with_fallback(
            provider_name=provider_name,
            provider=provider,
            entry_id=request.entry_id,
            entry_text=entry_text,
            chunk_ids=chunk_ids,
            source_file=request.source_file,
        )

        record = self._assemble_record(
            request=request,
            generation=generation,
            ordered_chunks=ordered_chunks,
        )

        return self.store.upsert(record)

    def get_state_labels(self, entry_id: str) -> StateLabelRecord | None:
        return self.store.get(entry_id)

    def _resolve_provider_name(self, requested: str | None) -> str:
        if requested and requested != "auto":
            if requested not in self.providers:
                raise ValueError(f"Unsupported provider: {requested}")
            return requested
        return self.default_provider_name

    def _generate_with_fallback(
        self,
        provider_name: str,
        provider: StateLabelProvider,
        entry_id: str,
        entry_text: str,
        chunk_ids: list[str],
        source_file: str | None,
    ) -> StateLabelGeneration:
        try:
            return provider.generate(entry_id, entry_text, chunk_ids, source_file)
        except Exception:
            if provider_name == self.fallback_provider_name:
                raise
            # Retry once after a brief pause (handles transient VPN/network blips)
            logger.warning(
                "state_label_provider_retrying",
                extra={
                    "event": "state_label_provider_retrying",
                    "provider": provider_name,
                    "entry_id": entry_id,
                },
                exc_info=True,
            )
            time.sleep(5)
            try:
                return provider.generate(entry_id, entry_text, chunk_ids, source_file)
            except Exception:
                logger.warning(
                    "state_label_provider_failed",
                    extra={
                        "event": "state_label_provider_failed",
                        "provider": provider_name,
                        "entry_id": entry_id,
                    },
                    exc_info=True,
                )
                fallback = self.providers[self.fallback_provider_name]
                return fallback.generate(entry_id, entry_text, chunk_ids, source_file)

    def _assemble_record(
        self,
        request: StateLabelGenerateRequest,
        generation: StateLabelGeneration,
        ordered_chunks: list[EntryChunk],
    ) -> StateLabelRecord:
        dimension_profiles: list[StateDimensionProfile] = []
        inferred_labels: list[InferredStateLabel] = []
        confidence_rows: list[DimensionConfidence] = []

        for dim_data in generation.dimensions:
            dim_name: StateDimension = dim_data["dimension"]
            score = float(dim_data.get("score", 0.0))
            label = str(dim_data.get("label", ""))
            rationale = str(dim_data.get("rationale", ""))
            low_anchor, high_anchor = DIMENSION_ANCHORS[dim_name]

            confidence_val = 0.6 if not generation.mock else 0.35 + min(abs(score), 0.6)

            dimension_profiles.append(
                StateDimensionProfile(
                    dimension=dim_name,
                    score=score,
                    low_anchor=low_anchor,
                    high_anchor=high_anchor,
                    label=label or f"between {low_anchor} and {high_anchor}",
                )
            )
            inferred_labels.append(
                InferredStateLabel(
                    dimension=dim_name,
                    label=label or f"between {low_anchor} and {high_anchor}",
                    score=score,
                    rationale=rationale,
                    supporting_signal_ids=[],
                    confidence=confidence_val,
                )
            )
            confidence_rows.append(
                DimensionConfidence(dimension=dim_name, value=confidence_val)
            )

        overall_confidence = round(
            sum(row.value for row in confidence_rows) / max(len(confidence_rows), 1),
            4,
        )

        observed_signals: list[ObservedTextSignal] = []
        for idx, sig_data in enumerate(generation.observed_signals):
            sig_id = sig_data.get("signal_id", f"sig-ollama-{idx:04d}")
            sig_dims = sig_data.get("dimensions", [])
            valid_dims = [d for d in sig_dims if d in DIMENSION_ANCHORS]
            if not valid_dims:
                continue
            direction = sig_data.get("direction", "neutral")
            if direction not in ("low", "high", "neutral"):
                direction = "neutral"
            category = sig_data.get("category", "lexical")
            if category not in ("lexical", "pattern", "modal", "temporal", "relational", "structural"):
                category = "lexical"
            observed_signals.append(
                ObservedTextSignal(
                    signal_id=sig_id[:256],
                    signal=str(sig_data.get("signal", ""))[:256],
                    category=category,
                    direction=direction,
                    dimensions=valid_dims,
                    weight=float(sig_data.get("weight", 0.5)),
                )
            )

        # Link signal IDs to inferred labels
        for label in inferred_labels:
            label.supporting_signal_ids = [
                sig.signal_id
                for sig in observed_signals
                if label.dimension in sig.dimensions
            ]

        provenance = self._build_provenance(ordered_chunks, request.source_file)
        created_at = datetime.now(timezone.utc).isoformat()

        return StateLabelRecord(
            entry_id=request.entry_id,
            entry_date=request.entry_date,
            source_file=request.source_file,
            state_profile=StateProfile(dimensions=dimension_profiles),
            observed_text_signals=observed_signals,
            inferred_state_labels=inferred_labels,
            confidence=StateConfidence(overall=overall_confidence, by_dimension=confidence_rows),
            provenance=provenance,
            processing=StateLabelProcessingMetadata(
                model_version=generation.model_version,
                prompt_version=generation.prompt_version,
                schema_version=STATE_LABEL_SCHEMA_VERSION,
                created_at=created_at,
                provider=generation.provider,
                mock=generation.mock,
            ),
        )

    @staticmethod
    def _build_provenance(
        chunks: list[EntryChunk],
        fallback_source_file: str | None,
    ) -> Provenance:
        chunk_ids: list[str] = []
        spans: list[SourceSpan] = []
        cursor = 0

        for chunk in chunks:
            chunk_ids.append(chunk.chunk_id)
            normalized_text = " ".join(chunk.text.split())
            start_char = cursor
            end_char = start_char + len(normalized_text)
            spans.append(
                SourceSpan(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file or fallback_source_file,
                    start_char=start_char,
                    end_char=end_char,
                    excerpt=normalized_text[:220] or None,
                )
            )
            cursor = end_char + 2

        return Provenance(chunk_ids=chunk_ids, spans=spans)


def create_state_label_service() -> StateLabelService:
    from .provider_registry import get_provider_registry

    db_path = os.environ.get("ENTRY_SUMMARY_DB_PATH", "/service/data/analysis.sqlite")
    registry = get_provider_registry()

    store = SQLiteStateLabelStore(db_path)
    return StateLabelService(
        store=store,
        providers=registry.state_label_providers,
        default_provider_name=registry.default_state_label,
        fallback_provider_name=registry.fallback,
    )


STATE_LABEL_SERVICE = create_state_label_service()


def get_state_label_service() -> StateLabelService:
    return STATE_LABEL_SERVICE
