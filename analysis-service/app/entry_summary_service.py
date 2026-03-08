from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from .entry_summary_provider import (
    EntrySummaryGeneration,
    EntrySummaryProvider,
)
from .entry_summary_store import EntrySummaryStore, SQLiteEntrySummaryStore
from .models import (
    EntryChunk,
    EntrySummaryGenerateRequest,
    EntrySummaryProcessingMetadata,
    EntrySummaryRecord,
    Provenance,
    SourceSpan,
    StateLabelGenerateRequest,
    StateLabelRequest,
)
from .state_engine import label_state
from .state_label_service import StateLabelService, create_state_label_service

ENTRY_SUMMARY_SCHEMA_VERSION = "entry-summary-v1"

logger = logging.getLogger("analysis-service.entry-summary")


class EntrySummaryService:
    """Service layer orchestrating provider inference + state labeling + persistence."""

    def __init__(
        self,
        store: EntrySummaryStore,
        providers: dict[str, EntrySummaryProvider],
        default_provider_name: str,
        fallback_provider_name: str = "local",
        state_label_service: StateLabelService | None = None,
    ):
        self.store = store
        self.providers = providers
        self.default_provider_name = default_provider_name
        self.fallback_provider_name = fallback_provider_name
        self.state_label_service = state_label_service

    def generate_and_persist(self, request: EntrySummaryGenerateRequest) -> EntrySummaryRecord:
        existing = self.store.get(request.entry_id)
        if existing and not request.force_regenerate:
            return existing

        provider_name = self._resolve_provider_name(request.provider)
        provider = self.providers[provider_name]

        ordered_chunks = self._ordered_chunks(request.chunks)
        entry_text = self._compose_entry_text(ordered_chunks)

        generation = self._generate_with_fallback(
            provider_name=provider_name,
            provider=provider,
            entry_id=request.entry_id,
            entry_text=entry_text,
            chunks=ordered_chunks,
        )

        if self.state_label_service is not None:
            # Hybrid/local providers use rule-based state labels (no API call)
            sl_provider = request.provider
            if sl_provider in ("hybrid", "local"):
                sl_provider = "local"
            sl_request = StateLabelGenerateRequest(
                entry_id=request.entry_id,
                entry_date=request.entry_date,
                source_file=request.source_file,
                chunks=ordered_chunks,
                force_regenerate=request.force_regenerate,
                provider=sl_provider,
            )
            sl_record = self.state_label_service.generate_and_persist(sl_request)
            state_profile = sl_record.state_profile
        else:
            state = label_state(
                StateLabelRequest(
                    entry_id=request.entry_id,
                    text=entry_text,
                    entry_date=request.entry_date,
                    source_file=request.source_file,
                    chunk_ids=[chunk.chunk_id for chunk in ordered_chunks],
                )
            )
            state_profile = state.state_profile

        created_at = datetime.now(timezone.utc).isoformat()
        record = EntrySummaryRecord(
            entry_id=request.entry_id,
            entry_date=request.entry_date,
            source_file=request.source_file,
            short_summary=generation.short_summary,
            detailed_summary=generation.detailed_summary,
            themes=generation.themes,
            entities=generation.entities,
            decisions_actions=generation.decisions_actions,
            state_profile=state_profile,
            provenance=self._build_provenance(ordered_chunks, request.source_file),
            processing=EntrySummaryProcessingMetadata(
                model_version=generation.model_version,
                prompt_version=generation.prompt_version,
                schema_version=ENTRY_SUMMARY_SCHEMA_VERSION,
                created_at=created_at,
                provider=generation.provider,
                mock=generation.mock,
            ),
        )

        return self.store.upsert(record)

    def get_entry_summary(self, entry_id: str) -> EntrySummaryRecord | None:
        return self.store.get(entry_id)

    # Providers that only handle state labels, not summaries.
    # When one of these is requested, summary generation uses the default provider.
    _STATE_LABEL_ONLY_PROVIDERS = {"finetuned"}

    def _resolve_provider_name(self, requested: str | None) -> str:
        if requested and requested != "auto":
            if requested in self.providers:
                return requested
            if requested in self._STATE_LABEL_ONLY_PROVIDERS:
                # State-label-only provider — fall back to default for summaries
                return self.default_provider_name
            raise ValueError(f"Unsupported provider: {requested}")
        return self.default_provider_name

    def _generate_with_fallback(
        self,
        provider_name: str,
        provider: EntrySummaryProvider,
        entry_id: str,
        entry_text: str,
        chunks: list[EntryChunk],
    ) -> EntrySummaryGeneration:
        try:
            return provider.generate(entry_id, entry_text, chunks)
        except Exception:
            if provider_name == self.fallback_provider_name:
                raise
            # Retry once after a brief pause (handles transient VPN/network blips)
            logger.warning(
                "entry_summary_provider_retrying",
                extra={
                    "event": "entry_summary_provider_retrying",
                    "provider": provider_name,
                    "entry_id": entry_id,
                },
                exc_info=True,
            )
            time.sleep(5)
            try:
                return provider.generate(entry_id, entry_text, chunks)
            except Exception:
                logger.warning(
                    "entry_summary_provider_failed",
                    extra={
                        "event": "entry_summary_provider_failed",
                        "provider": provider_name,
                        "entry_id": entry_id,
                    },
                    exc_info=True,
                )
                fallback = self.providers[self.fallback_provider_name]
                return fallback.generate(entry_id, entry_text, chunks)

    @staticmethod
    def _ordered_chunks(chunks: list[EntryChunk]) -> list[EntryChunk]:
        return list(chunks)

    @staticmethod
    def _compose_entry_text(chunks: list[EntryChunk]) -> str:
        return "\n\n".join(chunk.text for chunk in chunks)

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


def create_entry_summary_service() -> EntrySummaryService:
    from .provider_registry import get_provider_registry

    db_path = os.environ.get("ENTRY_SUMMARY_DB_PATH", "/service/data/analysis.sqlite")
    registry = get_provider_registry()

    state_label_svc = create_state_label_service()

    store = SQLiteEntrySummaryStore(db_path)
    return EntrySummaryService(
        store=store,
        providers=registry.summary_providers,
        default_provider_name=registry.default_summary,
        fallback_provider_name=registry.fallback,
        state_label_service=state_label_svc,
    )


ENTRY_SUMMARY_SERVICE = create_entry_summary_service()


def get_entry_summary_service() -> EntrySummaryService:
    return ENTRY_SUMMARY_SERVICE
