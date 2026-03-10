"""FastAPI app for deterministic analysis-layer stubs."""

import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request

from .analysis import build_context_packet, label_state, summarize_entry
from .entry_summary_service import get_entry_summary_service
from .logging_utils import RequestIdMiddleware, request_id_var, setup_logging
from .models import (
    ContextPacketRequest,
    ContextPacketResponse,
    EntrySummaryGenerateRequest,
    EntrySummaryRecord,
    StateLabelGenerateRequest,
    StateLabelRecord,
    StateLabelRequest,
    StateLabelResponse,
    SummarizeEntryRequest,
    SummarizeEntryResponse,
)
from .state_label_service import get_state_label_service

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("analysis-service")

app = FastAPI(title="Corpus Analysis Service", version="0.1.0")
app.add_middleware(RequestIdMiddleware)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "request_failed",
            extra={
                "event": "request_failed",
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            },
        )
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "request_completed",
        extra={
            "event": "request_completed",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.get("/health")
async def health() -> dict[str, str | bool]:
    from .provider_registry import get_provider_registry

    registry = get_provider_registry()
    is_mock = registry.default_summary == "mock" and registry.default_state_label == "mock"
    return {
        "status": "ok",
        "service": "analysis-service",
        "mock_mode": is_mock,
        "summary_provider": registry.default_summary,
        "state_label_provider": registry.default_state_label,
    }


@app.post("/summarize/entry", response_model=SummarizeEntryResponse)
async def summarize_entry_endpoint(body: SummarizeEntryRequest) -> SummarizeEntryResponse:
    logger.info(
        "summarize_entry_requested",
        extra={
            "event": "summarize_entry_requested",
            "entry_id": body.entry_id,
            "chunk_count": len(body.chunk_ids),
        },
    )
    try:
        return summarize_entry(body)
    except Exception as exc:
        logger.exception(
            "summarize_entry_failed",
            extra={"event": "summarize_entry_failed", "entry_id": body.entry_id},
        )
        raise HTTPException(status_code=500, detail="Failed to generate entry summary") from exc


@app.post("/state/label", response_model=StateLabelResponse)
async def state_label_endpoint(body: StateLabelRequest) -> StateLabelResponse:
    logger.info(
        "state_label_requested",
        extra={
            "event": "state_label_requested",
            "entry_id": body.entry_id,
            "chunk_count": len(body.chunk_ids),
        },
    )
    try:
        return label_state(body)
    except Exception as exc:
        logger.exception(
            "state_label_failed",
            extra={"event": "state_label_failed", "entry_id": body.entry_id},
        )
        raise HTTPException(status_code=500, detail="Failed to label entry state") from exc


@app.post("/context/packet", response_model=ContextPacketResponse)
async def context_packet_endpoint(body: ContextPacketRequest) -> ContextPacketResponse:
    logger.info(
        "context_packet_requested",
        extra={
            "event": "context_packet_requested",
            "query_length": len(body.query),
            "retrieval_hits": len(body.retrieval_hits),
            "graph_signals": len(body.graph_signals),
        },
    )
    try:
        return build_context_packet(body)
    except Exception as exc:
        logger.exception("context_packet_failed", extra={"event": "context_packet_failed"})
        raise HTTPException(status_code=500, detail="Failed to build context packet") from exc


@app.post("/entry-summary/generate", response_model=EntrySummaryRecord)
async def generate_entry_summary_endpoint(
    body: EntrySummaryGenerateRequest,
) -> EntrySummaryRecord:
    logger.info(
        "entry_summary_generate_requested",
        extra={
            "event": "entry_summary_generate_requested",
            "entry_id": body.entry_id,
            "chunk_count": len(body.chunks),
            "force_regenerate": body.force_regenerate,
            "provider": body.provider or "default",
        },
    )
    service = get_entry_summary_service()
    try:
        return service.generate_and_persist(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "entry_summary_generate_failed",
            extra={"event": "entry_summary_generate_failed", "entry_id": body.entry_id},
        )
        raise HTTPException(status_code=500, detail="Failed to generate entry summary") from exc


@app.get("/entry-summary/{entry_id}", response_model=EntrySummaryRecord)
async def get_entry_summary_endpoint(entry_id: str) -> EntrySummaryRecord:
    normalized = entry_id.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="entry_id must not be blank")

    service = get_entry_summary_service()
    record = service.get_entry_summary(normalized)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No summary found for entry_id={normalized}")
    return record


@app.post("/state-label/generate", response_model=StateLabelRecord)
async def generate_state_label_endpoint(
    body: StateLabelGenerateRequest,
) -> StateLabelRecord:
    logger.info(
        "state_label_generate_requested",
        extra={
            "event": "state_label_generate_requested",
            "entry_id": body.entry_id,
            "chunk_count": len(body.chunks),
            "force_regenerate": body.force_regenerate,
            "provider": body.provider or "default",
        },
    )
    service = get_state_label_service()
    try:
        return service.generate_and_persist(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "state_label_generate_failed",
            extra={"event": "state_label_generate_failed", "entry_id": body.entry_id},
        )
        raise HTTPException(status_code=500, detail="Failed to generate state label") from exc


@app.get("/state-label/{entry_id}", response_model=StateLabelRecord)
async def get_state_label_endpoint(entry_id: str) -> StateLabelRecord:
    normalized = entry_id.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="entry_id must not be blank")

    service = get_state_label_service()
    record = service.get_state_labels(normalized)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No state label found for entry_id={normalized}")
    return record


# ─── State Label Audit Endpoints ─────────────────────────────────────────────

from .state_label_audit import (
    get_audit_store,
    get_extreme_entries,
    get_entry_text_preview,
    AuditJudgment,
    DIMENSIONS,
)


@app.get("/state-label/audit/sample")
async def audit_sample(
    dimension: str | None = None,
    threshold: float = 0.6,
    limit: int = 20,
    exclude_audited: bool = True,
) -> dict:
    """
    Get entries with extreme dimension scores for audit review.

    Args:
        dimension: Optional dimension to filter by (e.g., "valence", "agency")
        threshold: Minimum absolute score to include (default 0.6)
        limit: Maximum entries to return (default 20)
        exclude_audited: Whether to exclude already-audited entries (default True)

    Returns:
        List of entries with their scores, signals, and text preview
    """
    if dimension and dimension not in DIMENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension. Must be one of: {', '.join(DIMENSIONS)}"
        )

    db_path = os.environ.get("ENTRY_SUMMARY_DB_PATH", "/service/data/analysis.sqlite")

    entries = get_extreme_entries(
        db_path=db_path,
        dimension=dimension,
        threshold=threshold,
        limit=limit,
        exclude_audited=exclude_audited,
        audit_db_path=db_path,
    )

    # Add text preview to each entry
    for entry in entries:
        entry["text_preview"] = get_entry_text_preview(db_path, entry["entry_id"])

    return {
        "count": len(entries),
        "threshold": threshold,
        "dimension_filter": dimension,
        "entries": entries,
    }


@app.post("/state-label/audit/judgment")
async def record_audit_judgment(body: AuditJudgment) -> dict:
    """
    Record a human judgment on a dimension score.

    The judgment indicates whether the automated score aligns with human assessment:
    - "agree": The score direction and magnitude seem correct
    - "disagree": The score seems wrong (optionally provide correct_direction)
    - "uncertain": Not enough information to judge

    If disagreeing, you can specify the correct_direction ("low", "high", "neutral").
    """
    if body.dimension not in DIMENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension. Must be one of: {', '.join(DIMENSIONS)}"
        )

    if body.judgment not in ("agree", "disagree", "uncertain"):
        raise HTTPException(
            status_code=400,
            detail="Judgment must be 'agree', 'disagree', or 'uncertain'"
        )

    store = get_audit_store()
    result = store.record_judgment(
        entry_id=body.entry_id,
        dimension=body.dimension,
        original_score=body.original_score,
        judgment=body.judgment,
        correct_direction=body.correct_direction,
        notes=body.notes,
    )

    return result


@app.get("/state-label/audit/report")
async def audit_report() -> dict:
    """
    Get aggregated audit statistics.

    Returns:
        - Total judgments recorded
        - Per-dimension accuracy (agree vs disagree rate)
        - Overall accuracy across all dimensions
        - Worst and best performing dimensions
        - Signals that frequently appear in disagreements (potential false positives)
    """
    store = get_audit_store()
    return store.get_report()


@app.get("/state-label/audit/judgments")
async def get_audit_judgments(
    dimension: str | None = None,
    judgment: str | None = None,
    limit: int = 100,
) -> dict:
    """Get recorded audit judgments with optional filters."""
    if dimension and dimension not in DIMENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension. Must be one of: {', '.join(DIMENSIONS)}"
        )

    store = get_audit_store()
    judgments = store.get_judgments(
        dimension=dimension,
        judgment=judgment,
        limit=limit,
    )

    return {
        "count": len(judgments),
        "judgments": judgments,
    }
