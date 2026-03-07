from __future__ import annotations

import hashlib
import json
import logging

from .entry_summary_provider import EntrySummaryProvider
from .models import (
    AnalysisVersion,
    DimensionConfidence,
    InferredStateLabel,
    ObservedTextSignal,
    Provenance,
    SourceSpan,
    StateConfidence,
    StateDimensionProfile,
    StateLabelRequest,
    StateLabelResponse,
    StateProfile,
    SummarizeEntryRequest,
    SummarizeEntryResponse,
)
from .state_label_provider import (
    DIMENSION_ANCHORS,
    StateLabelProvider,
)

logger = logging.getLogger("analysis-service.oneshot-providers")


def summarize_via_provider(
    request: SummarizeEntryRequest,
    provider: EntrySummaryProvider,
) -> SummarizeEntryResponse:
    from .models import EntryChunk

    clean_text = " ".join(request.text.split())
    chunk_ids = request.chunk_ids or [f"{request.entry_id}::chunk-000"]
    chunks = [
        EntryChunk(
            chunk_id=chunk_ids[0] if chunk_ids else f"{request.entry_id}::chunk-000",
            text=clean_text,
            source_file=request.source_file,
        )
    ]

    generation = provider.generate(request.entry_id, clean_text, chunks)

    summary = generation.short_summary
    key_terms = generation.themes[:6]
    total_words = len(clean_text.split())
    summary_words = len(summary.split())
    coverage_ratio = round(summary_words / max(total_words, 1), 4)

    highlights = [
        f"Primary threads: {', '.join(key_terms[:3]) or 'general reflection'}.",
        f"Summary covers {summary_words} of {total_words} words ({round(coverage_ratio * 100, 1)}%).",
        f"Provider: {generation.provider}, model: {generation.model_version}.",
    ]

    provenance = _build_provenance(
        text=clean_text,
        source_file=request.source_file,
        chunk_ids=list(chunk_ids),
        fallback_chunk_id=f"{request.entry_id}::chunk-000",
    )

    analysis_id = _stable_id(
        prefix="sum",
        payload={
            "entry_id": request.entry_id,
            "text": clean_text,
            "max_summary_sentences": request.max_summary_sentences,
            "chunk_ids": list(chunk_ids),
            "provider": generation.provider,
        },
    )

    return SummarizeEntryResponse(
        analysis_id=analysis_id,
        entry_id=request.entry_id,
        granularity="entry",
        summary=summary,
        highlights=highlights,
        key_terms=key_terms,
        coverage_ratio=coverage_ratio,
        provenance=provenance,
        version=AnalysisVersion(
            schema_version="0.1.0",
            prompt_version=generation.prompt_version,
            model_version=generation.model_version,
            mock=generation.mock,
        ),
    )


def label_state_via_provider(
    request: StateLabelRequest,
    provider: StateLabelProvider,
) -> StateLabelResponse:
    clean_text = " ".join(request.text.split())
    chunk_ids = request.chunk_ids or [f"{request.entry_id}::chunk-000"]
    fallback_chunk_id = chunk_ids[0] if chunk_ids else f"{request.entry_id}::chunk-000"

    generation = provider.generate(
        request.entry_id,
        clean_text,
        list(chunk_ids),
        request.source_file,
    )

    dimension_profiles: list[StateDimensionProfile] = []
    inferred_labels: list[InferredStateLabel] = []
    confidence_rows: list[DimensionConfidence] = []

    for dim_data in generation.dimensions:
        dim_name = dim_data["dimension"]
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
                signal_id=sig_id,
                signal=str(sig_data.get("signal", "")),
                category=category,
                direction=direction,
                dimensions=valid_dims,
                weight=float(sig_data.get("weight", 0.5)),
            )
        )

    for label_obj in inferred_labels:
        label_obj.supporting_signal_ids = [
            sig.signal_id
            for sig in observed_signals
            if label_obj.dimension in sig.dimensions
        ]

    provenance = Provenance(
        chunk_ids=_unique(list(chunk_ids)) or [fallback_chunk_id],
        spans=[],
    )

    analysis_id = _stable_id(
        "state",
        {
            "entry_id": request.entry_id,
            "text": clean_text,
            "chunk_ids": list(chunk_ids),
            "provider": generation.provider,
        },
    )

    return StateLabelResponse(
        analysis_id=analysis_id,
        entry_id=request.entry_id,
        state_profile=StateProfile(dimensions=dimension_profiles),
        observed_text_signals=observed_signals,
        inferred_state_labels=inferred_labels,
        confidence=StateConfidence(overall=overall_confidence, by_dimension=confidence_rows),
        provenance=provenance,
        version=AnalysisVersion(
            schema_version="state-profile-v1",
            prompt_version=generation.prompt_version,
            model_version=generation.model_version,
            mock=generation.mock,
        ),
    )


def _build_provenance(
    text: str,
    source_file: str | None,
    chunk_ids: list[str],
    fallback_chunk_id: str,
) -> Provenance:
    normalized_chunks = _unique(chunk_ids) or [fallback_chunk_id]
    spans: list[SourceSpan] = []
    for index, chunk_id in enumerate(normalized_chunks[:20]):
        start = min(index * 180, max(0, len(text) - 1)) if text else 0
        end = min(start + 160, len(text))
        excerpt = text[start:end] if end > start else None
        spans.append(
            SourceSpan(
                chunk_id=chunk_id,
                source_file=source_file,
                start_char=start,
                end_char=end,
                excerpt=excerpt,
            )
        )
    return Provenance(chunk_ids=normalized_chunks, spans=spans)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _stable_id(prefix: str, payload: dict, digest_len: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:digest_len]
    return f"{prefix}-{digest}"
