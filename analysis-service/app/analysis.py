from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter

from .models import (
    AnalysisVersion,
    ContextGraphItem,
    ContextPacketRequest,
    ContextPacketResponse,
    ContextRetrievalItem,
    Provenance,
    RetrievalHit,
    SourceSpan,
    SummarizeEntryRequest,
    SummarizeEntryResponse,
)
from .state_engine import label_state

logger = logging.getLogger("analysis-service.analysis")

SCHEMA_VERSION = "0.1.0"
PROMPT_VERSION = "analysis-mock-v1"
MODEL_VERSION = "deterministic-heuristic-mock"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}


def summarize_entry(request: SummarizeEntryRequest) -> SummarizeEntryResponse:
    """Build an entry summary using the configured provider, falling back to deterministic."""
    from .provider_registry import get_provider_registry
    from .oneshot_providers import summarize_via_provider

    registry = get_provider_registry()
    if registry.default_summary != "mock":
        provider = registry.summary_providers.get(registry.default_summary)
        if provider is not None:
            try:
                return summarize_via_provider(request, provider)
            except Exception:
                logger.warning(
                    "oneshot_summary_provider_failed",
                    extra={
                        "event": "oneshot_summary_provider_failed",
                        "provider": registry.default_summary,
                        "entry_id": request.entry_id,
                    },
                    exc_info=True,
                )

    clean_text = _normalize_whitespace(request.text)
    sentences = _split_sentences(clean_text)
    selected = sentences[: request.max_summary_sentences]

    if not selected:
        selected = [_truncate_words(clean_text, 40)]

    summary = " ".join(selected).strip()
    key_terms = _extract_key_terms(clean_text, max_terms=6)
    total_words = len(clean_text.split())
    summary_words = len(summary.split())
    coverage_ratio = round(summary_words / max(total_words, 1), 4)

    highlights = [
        f"Primary threads: {', '.join(key_terms[:3]) or 'general reflection'}.",
        f"Summary covers {summary_words} of {total_words} words ({round(coverage_ratio * 100, 1)}%).",
        "Granularity is entry-level and can be rolled up to day/week/month in later passes.",
    ]

    provenance = _build_provenance(
        text=clean_text,
        source_file=request.source_file,
        chunk_ids=request.chunk_ids,
        fallback_chunk_id=f"{request.entry_id}::chunk-000",
    )

    analysis_id = _stable_id(
        prefix="sum",
        payload={
            "entry_id": request.entry_id,
            "text": clean_text,
            "max_summary_sentences": request.max_summary_sentences,
            "chunk_ids": request.chunk_ids,
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
        version=_analysis_version(),
    )


def build_context_packet(request: ContextPacketRequest) -> ContextPacketResponse:
    """Assemble deterministic context packet from temporal + retrieval + graph inputs.

    TODO: Integrate live retrieval/graph fetches and learned ranking before packet assembly.
    """

    retrieval_items, selected_hits = _build_retrieval_context(request)
    graph_items = _build_graph_context(request)

    if request.date_start and request.date_end:
        temporal_focus = f"Bounded to {request.date_start} through {request.date_end}."
    elif request.date_start:
        temporal_focus = f"Bounded from {request.date_start} onward."
    elif request.date_end:
        temporal_focus = f"Bounded through {request.date_end}."
    else:
        temporal_focus = "No explicit date bounds; use corpus-wide temporal context."

    retrieval_line = "; ".join(
        f"{item.chunk_id} ({item.relevance_score})" for item in retrieval_items
    )
    graph_line = "; ".join(
        f"{item.subject} {item.relation} {item.object}" for item in graph_items
    )
    context_brief = (
        f"Query: {request.query}\n"
        f"Temporal: {temporal_focus}\n"
        f"Retrieval anchors: {retrieval_line or 'none'}\n"
        f"Graph anchors: {graph_line or 'none'}"
    )

    provenance = _build_packet_provenance(selected_hits)

    packet_id = _stable_id(
        prefix="ctx",
        payload={
            "query": request.query,
            "date_start": request.date_start,
            "date_end": request.date_end,
            "retrieval_hits": [hit.model_dump() for hit in selected_hits],
            "graph_signals": [item.model_dump() for item in graph_items],
        },
    )

    return ContextPacketResponse(
        packet_id=packet_id,
        query=request.query,
        temporal_focus=temporal_focus,
        retrieval_context=retrieval_items,
        graph_context=graph_items,
        context_brief=context_brief,
        provenance=provenance,
        version=_analysis_version(),
    )


def _build_retrieval_context(
    request: ContextPacketRequest,
) -> tuple[list[ContextRetrievalItem], list[RetrievalHit]]:
    if request.retrieval_hits:
        selected_hits = sorted(
            request.retrieval_hits,
            key=lambda item: (-item.relevance_score, item.chunk_id),
        )[: request.top_k]
        retrieval_items = [
            ContextRetrievalItem(
                chunk_id=hit.chunk_id,
                source_file=hit.source_file,
                relevance_score=round(hit.relevance_score, 4),
                rationale="Provided retrieval hit reused in deterministic context packet.",
            )
            for hit in selected_hits
        ]
        return retrieval_items, selected_hits

    terms = _extract_key_terms(request.query, max_terms=request.top_k)
    if not terms:
        terms = ["context"]

    selected_hits: list[RetrievalHit] = []
    retrieval_items = []
    for index, term in enumerate(terms[: request.top_k]):
        score = max(0.1, round(0.82 - index * 0.11, 4))
        chunk_id = f"mock-{_stable_id('chunk', {'term': term, 'idx': index}, digest_len=8)}"
        excerpt = f"Mock retrieval anchor for term '{term}' from synthetic corpus context."
        selected_hits.append(
            RetrievalHit(
                chunk_id=chunk_id,
                source_file="mock://analysis-service",
                excerpt=excerpt,
                relevance_score=score,
            )
        )
        retrieval_items.append(
            ContextRetrievalItem(
                chunk_id=chunk_id,
                source_file="mock://analysis-service",
                relevance_score=score,
                rationale=f"Synthesized retrieval anchor for key term '{term}'.",
            )
        )

    return retrieval_items, selected_hits


def _build_graph_context(request: ContextPacketRequest) -> list[ContextGraphItem]:
    if request.graph_signals:
        return [
            ContextGraphItem(
                subject=item.subject,
                relation=item.relation,
                object=item.object,
                weight=round(item.weight, 4),
            )
            for item in sorted(
                request.graph_signals,
                key=lambda node: (-node.weight, node.subject, node.object),
            )[: request.top_k]
        ]

    terms = _extract_key_terms(request.query, max_terms=max(2, min(4, request.top_k)))
    if not terms:
        terms = ["context", "theme"]

    graph_items: list[ContextGraphItem] = []
    for index, term in enumerate(terms[: request.top_k]):
        graph_items.append(
            ContextGraphItem(
                subject="query",
                relation="RELATED_TO",
                object=term,
                weight=max(0.1, round(0.78 - index * 0.12, 4)),
            )
        )
    return graph_items


def _build_packet_provenance(selected_hits: list[RetrievalHit]) -> Provenance:
    chunk_ids: list[str] = []
    spans: list[SourceSpan] = []

    for hit in selected_hits:
        chunk_id = getattr(hit, "chunk_id", None)
        if not chunk_id:
            continue
        chunk_ids.append(chunk_id)
        excerpt = _normalize_whitespace(getattr(hit, "excerpt", ""))
        spans.append(
            SourceSpan(
                chunk_id=chunk_id,
                source_file=getattr(hit, "source_file", None),
                start_char=0,
                end_char=max(0, min(len(excerpt), 220)),
                excerpt=excerpt[:220] or None,
            )
        )

    return Provenance(chunk_ids=_unique(chunk_ids), spans=spans)


def _build_provenance(
    text: str,
    source_file: str | None,
    chunk_ids: list[str],
    fallback_chunk_id: str,
) -> Provenance:
    text = _normalize_whitespace(text)
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


def _extract_key_terms(text: str, max_terms: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower())
    filtered = [word for word in words if word not in STOPWORDS]
    counts = Counter(filtered)

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:max_terms]]


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]).strip()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _analysis_version() -> AnalysisVersion:
    return AnalysisVersion(
        schema_version=SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION,
        model_version=MODEL_VERSION,
        mock=True,
    )


def _stable_id(prefix: str, payload: dict, digest_len: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:digest_len]
    return f"{prefix}-{digest}"
