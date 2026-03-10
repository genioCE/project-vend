"""Graph service — FastAPI endpoints for knowledge graph queries and GraphRAG."""

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, Field, field_validator

from .graph import (
    get_concept_network,
    get_person_network,
    get_concept_evolution,
    compare_periods,
    get_decision_context,
    get_archetype_patterns,
    get_concept_flows,
    get_subgraph,
    get_graph_stats,
    get_driver,
    get_theme_network,
    get_theme_cooccurrences,
    get_entries_by_state,
    get_organization_network,
)
from .graph_consistency import (
    run_all_consistency_checks,
    check_orphaned_nodes,
    check_duplicate_names,
    check_missing_state_profiles,
    check_abnormal_cooccurrence_weights,
    check_disconnected_entries,
    check_entries_without_concepts,
    cleanup_duplicate_entries,
)
from .feedback import get_feedback_store, GRAPH_AUTO_TUNING_PATH
from .feedback_review import build_feedback_review, suggest_prompts_from_note
from .extractor import extract_entities
from .graphrag import graphrag_query
from .style_profile import reload_style_config
from .corpus_utils import RequestIdMiddleware, setup_logging

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="Corpus Graph Service")
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


@app.on_event("startup")
async def _register_reload_callback() -> None:
    store = get_feedback_store()
    store.set_reload_callback(reload_style_config)


# --- Request models ---

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("query must not be blank")
        return text

class CompareRequest(BaseModel):
    start1: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end1: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    start2: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end2: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

class DecisionRequest(BaseModel):
    keyword: str | None = Field(default=None, max_length=256)
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("keyword")
    @classmethod
    def validate_keyword(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text if text else None


class FeedbackRequest(BaseModel):
    signal: str = Field(pattern=r"^(up|down)$")
    query: str | None = Field(default=None, max_length=2000)
    note: str | None = Field(default=None, max_length=2000)
    concepts: list[str] = Field(default_factory=list, max_length=50)
    people: list[str] = Field(default_factory=list, max_length=50)
    places: list[str] = Field(default_factory=list, max_length=50)
    sources: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("query", "note")
    @classmethod
    def validate_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text if text else None

    @field_validator("concepts", "people", "places", "sources")
    @classmethod
    def normalize_terms(cls, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = " ".join(value.strip().split())
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:20]


# --- Endpoints ---

@app.get("/health")
async def health():
    try:
        get_driver().verify_connectivity()
        return {"status": "ok", "neo4j": "connected"}
    except Exception:
        return {"status": "degraded", "neo4j": "unavailable"}


@app.post("/graph/search")
async def graph_search(body: SearchRequest) -> dict[str, Any]:
    """Combined GraphRAG search — vector + graph traversal."""
    result = await graphrag_query(body.query, body.top_k)
    return result


@app.get("/graph/concept/{name}")
async def concept_network(name: str, limit: int = Query(default=30, ge=1, le=100)):
    network = get_concept_network(name, limit=limit)
    return {"concept": name, "network": network}


@app.get("/graph/person/{name}")
async def person_network(name: str, limit: int = Query(default=30, ge=1, le=100)):
    network = get_person_network(name, limit=limit)
    return {"person": name, "network": network}


@app.get("/graph/evolution/{name}")
async def concept_evolution(name: str, limit: int = Query(default=20, ge=1, le=50)):
    evolution = get_concept_evolution(name, limit=limit)
    return {"concept": name, "evolution": evolution}


@app.post("/graph/compare")
async def compare(body: CompareRequest):
    return compare_periods(body.start1, body.end1, body.start2, body.end2)


@app.get("/graph/archetypes")
async def archetypes(limit: int = Query(default=10, ge=1, le=20)):
    return {"archetypes": get_archetype_patterns(limit)}


@app.get("/graph/flows/{name}")
async def concept_flows(name: str, limit: int = Query(default=20, ge=1, le=100)):
    return {"concept": name, "flows": get_concept_flows(name, limit)}


@app.post("/graph/decision_context")
async def decision_context(body: DecisionRequest):
    return {"decisions": get_decision_context(body.keyword, body.limit)}


@app.post("/graph/feedback")
async def graph_feedback(body: FeedbackRequest):
    query_entities: dict[str, list[str]] | None = None
    if body.query:
        extracted = extract_entities(body.query)
        query_entities = {
            "concepts": extracted.get("concepts", []),
            "people": extracted.get("people", []),
            "places": extracted.get("places", []),
        }

    store = get_feedback_store()
    result = store.record_feedback(
        signal=body.signal,
        query=body.query,
        note=body.note,
        concepts=body.concepts,
        people=body.people,
        places=body.places,
        sources=body.sources,
        query_entities=query_entities,
    )
    if body.signal == "down" and body.note:
        result["suggested_prompts"] = suggest_prompts_from_note(body.note, limit=3)
    return result


@app.get("/graph/feedback_profile")
async def graph_feedback_profile(top_n: int = Query(default=15, ge=1, le=100)):
    store = get_feedback_store()
    return store.profile(top_n=top_n)


@app.get("/graph/feedback_review")
async def graph_feedback_review(
    top_n: int = Query(default=15, ge=1, le=50),
    recent_limit: int = Query(default=40, ge=1, le=200),
):
    store = get_feedback_store()
    events = store.get_event_log(limit=2000)
    review = build_feedback_review(events, top_n=top_n, recent_limit=recent_limit)
    profile = store.profile(top_n=1)
    total_counter = int(profile.get("events", 0))
    logged_total = int(review.get("summary", {}).get("total_events", 0))
    if total_counter > logged_total:
        review["summary"]["total_events"] = total_counter
        review["summary"]["events_with_notes_enabled"] = logged_total
    return review


@app.get("/graph/feedback/health")
async def feedback_health():
    store = get_feedback_store()
    return store.health()


@app.get("/graph/feedback/tuning_preview")
async def feedback_tuning_preview():
    store = get_feedback_store()
    return store.compute_tuning()


@app.post("/graph/feedback/apply_tuning")
async def feedback_apply_tuning():
    store = get_feedback_store()
    result = store.apply_tuning(GRAPH_AUTO_TUNING_PATH)
    reload_style_config()
    return result


@app.post("/graph/feedback/aggregate")
async def feedback_aggregate():
    store = get_feedback_store()
    return store.aggregate_events()


@app.get("/graph/subgraph")
async def subgraph(
    center: str = Query(min_length=1, max_length=200),
    depth: int = Query(default=1, ge=1, le=3),
    limit: int = Query(default=50, ge=1, le=200),
):
    return get_subgraph(center, depth, limit)


@app.get("/graph/theme/{name}")
async def theme_network(name: str, limit: int = Query(default=30, ge=1, le=100)):
    network = get_theme_network(name, limit=limit)
    cooccurrences = get_theme_cooccurrences(name, limit=20)
    return {"theme": name, "entries": network, "cooccurrences": cooccurrences}


@app.get("/graph/organization/{name}")
async def organization_network(name: str, limit: int = Query(default=30, ge=1, le=100)):
    network = get_organization_network(name, limit=limit)
    return {"organization": name, "network": network}


@app.get("/graph/entries_by_state")
async def entries_by_state(
    dimension: str = Query(
        pattern=r"^(valence|activation|agency|certainty|relational_openness|self_trust|time_orientation|integration)$"
    ),
    min_score: float = Query(default=-1.0, ge=-1.0, le=1.0),
    max_score: float = Query(default=1.0, ge=-1.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
):
    entries = get_entries_by_state(dimension, min_score, max_score, limit)
    return {"dimension": dimension, "min_score": min_score, "max_score": max_score, "entries": entries}


@app.get("/graph/stats")
async def stats():
    return get_graph_stats()


@app.get("/graph/consistency")
async def consistency_check():
    """
    Run all graph consistency checks and return a summary.

    Checks include:
    - Orphaned nodes (entities with no Entry relationships)
    - Duplicate normalized names
    - Missing state profiles on entries
    - Abnormal cooccurrence weights
    - Disconnected entries
    - Entries without concepts (>100 words)
    """
    return run_all_consistency_checks()


@app.get("/graph/consistency/{check_name}")
async def consistency_check_single(check_name: str):
    """Run a single consistency check by name."""
    checks = {
        "orphaned_nodes": check_orphaned_nodes,
        "duplicate_names": check_duplicate_names,
        "missing_state_profiles": check_missing_state_profiles,
        "abnormal_cooccurrence_weights": check_abnormal_cooccurrence_weights,
        "disconnected_entries": check_disconnected_entries,
        "entries_without_concepts": check_entries_without_concepts,
    }
    if check_name not in checks:
        return {
            "error": f"Unknown check: {check_name}",
            "available_checks": list(checks.keys()),
        }
    return checks[check_name]()


@app.post("/graph/consistency/cleanup")
async def consistency_cleanup():
    """
    Clean up duplicate entries and cascading entity duplicates.

    This is a destructive operation that:
    1. Removes duplicate Entry nodes (keeps one per filename)
    2. Cleans up orphaned entity nodes
    3. Merges duplicate Person/Place/Concept nodes
    """
    return cleanup_duplicate_entries()
