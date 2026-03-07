"""GraphRAG — combines vector search with knowledge graph traversal."""

import asyncio
import os
import logging

import httpx

from .extractor import extract_entities
from .graph import get_concept_network, get_person_network
from .feedback import get_feedback_store
from .corpus_utils import request_id_var

logger = logging.getLogger(__name__)

EMBEDDINGS_SERVICE_URL = os.environ.get(
    "EMBEDDINGS_SERVICE_URL", "http://embeddings-service:8000"
)
GRAPHRAG_VECTOR_RESULT_LIMIT = max(
    1, int(os.environ.get("GRAPHRAG_VECTOR_RESULT_LIMIT", "4"))
)


async def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """Call the embeddings service for semantic vector search."""
    headers: dict[str, str] = {}
    rid = request_id_var.get("")
    if rid:
        headers["x-request-id"] = rid
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{EMBEDDINGS_SERVICE_URL}/search",
            json={"query": query, "top_k": top_k},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])


async def build_person_context(person: str) -> dict | None:
    """Fetch person-centric graph context without blocking the event loop."""
    try:
        network = await asyncio.to_thread(get_person_network, person, 5)
    except Exception as e:
        logger.warning(f"Person lookup failed for '{person}': {e}")
        return None

    if not network:
        return None

    return {"type": "person_context", "entity": person, "data": network}


async def build_concept_context(concept: str) -> dict | None:
    """Fetch concept-centric graph context without blocking the event loop."""
    try:
        network = await asyncio.to_thread(get_concept_network, concept, 8)
    except Exception as e:
        logger.warning(f"Concept lookup failed for '{concept}': {e}")
        return None

    if not network:
        return None

    return {"type": "concept_context", "entity": concept, "data": network}


def _normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def _dedupe_vector_results(results: list[dict], limit: int) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for item in results:
        text = _normalize_space(str(item.get("text", ""))).lower()[:280]
        date = str(item.get("date", "")).strip().lower()
        source = str(item.get("source_file", "")).strip().lower()
        key = f"{date}|{source}|{text}"
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _dedupe_person_rows(rows: list[dict], limit: int = 3) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        date = str(row.get("date", "")).strip().lower()
        filename = str(row.get("filename", "")).strip().lower()
        concepts = tuple(sorted(str(c).strip().lower() for c in row.get("concepts", []) if c))
        emotions = tuple(sorted(str(e).strip().lower() for e in row.get("emotions", []) if e))
        key = f"{date}|{filename}|{concepts}|{emotions}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def _dedupe_concept_rows(rows: list[dict], limit: int = 5) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        relationship = str(row.get("relationship", "")).strip().lower()
        node_type = str(row.get("node_type", "")).strip().lower()
        node_name = str(row.get("node_name", "")).strip().lower()
        date = str(row.get("date", "")).strip().lower()
        key = f"{relationship}|{node_type}|{node_name}|{date}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def _merge_graph_context(raw_context: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in raw_context:
        ctx_type = str(item.get("type", "")).strip()
        entity = str(item.get("entity", "")).strip()
        if not ctx_type or not entity:
            continue
        key = f"{ctx_type}:{entity.lower()}"
        data = item.get("data", [])
        if key not in merged:
            merged[key] = {"type": ctx_type, "entity": entity, "data": list(data)}
        else:
            merged[key]["data"].extend(list(data))

    deduped_context: list[dict] = []
    for item in merged.values():
        if item["type"] == "person_context":
            rows = _dedupe_person_rows(item["data"], 3)
        elif item["type"] == "concept_context":
            rows = _dedupe_concept_rows(item["data"], 5)
        else:
            rows = item["data"]

        if rows:
            deduped_context.append({
                "type": item["type"],
                "entity": item["entity"],
                "data": rows,
            })
    return deduped_context


async def graphrag_query(query: str, top_k: int = 5) -> dict:
    """
    Combined GraphRAG query:
    1. Extract entities from the query using spaCy
    2. Run vector search (semantic) in parallel with graph lookups (structural)
    3. Merge into an enriched context block for the LLM
    """
    # Step 1: Extract entities from the user query
    entities = extract_entities(query)

    # Step 2: Vector search + graph lookups in parallel
    vector_task = asyncio.create_task(vector_search(query, top_k))
    graph_tasks = [
        asyncio.create_task(build_person_context(person))
        for person in entities["people"][:3]
    ] + [
        asyncio.create_task(build_concept_context(concept))
        for concept in entities["concepts"][:5]
    ]

    gathered = await asyncio.gather(vector_task, *graph_tasks, return_exceptions=True)
    vector_payload = gathered[0]
    rerank_meta: dict = {"applied": False}
    if isinstance(vector_payload, Exception):
        logger.warning(f"Vector search failed for query '{query}': {vector_payload}")
        vector_results: list[dict] = []
    else:
        vector_results = _dedupe_vector_results(
            vector_payload, GRAPHRAG_VECTOR_RESULT_LIMIT
        )
        try:
            feedback_store = get_feedback_store()
            vector_results, rerank_meta = feedback_store.rerank_vector_results(
                vector_results,
                {
                    "concepts": entities.get("concepts", []),
                    "people": entities.get("people", []),
                    "places": entities.get("places", []),
                },
            )
        except Exception as e:
            logger.warning(f"Feedback reranking failed for query '{query}': {e}")
            rerank_meta = {"applied": False, "reason": "rerank_error"}

    raw_graph_context = [
        item
        for item in gathered[1:]
        if isinstance(item, dict)
    ]
    graph_context = _merge_graph_context(raw_graph_context)
    try:
        graph_context = get_feedback_store().rerank_graph_context(graph_context)
    except Exception as e:
        logger.warning(f"Graph context rerank failed for query '{query}': {e}")

    # Step 4: Build formatted context block
    context_parts: list[str] = []

    if vector_results:
        context_parts.append("=== Semantic Search Results ===")
        for i, r in enumerate(vector_results):
            score = r.get("relevance_score", 0)
            context_parts.append(
                f"[{i + 1}] Date: {r.get('date', 'unknown')} (relevance: {score:.2f})"
            )
            context_parts.append(r.get("text", ""))
            context_parts.append("---")

    if graph_context:
        context_parts.append("\n=== Knowledge Graph Context ===")
        for gc in graph_context:
            if gc["type"] == "person_context":
                context_parts.append(f"\nPerson: {gc['entity']}")
                for entry in gc["data"][:3]:
                    context_parts.append(
                        f"  - {entry.get('date', '?')}: "
                        f"concepts={entry.get('concepts', [])}, "
                        f"emotions={entry.get('emotions', [])}"
                    )
            elif gc["type"] == "concept_context":
                context_parts.append(f"\nConcept: {gc['entity']}")
                for conn in gc["data"][:5]:
                    context_parts.append(
                        f"  - [{conn.get('relationship', '?')}] "
                        f"{conn.get('node_type', '?')}: {conn.get('node_name', '?')} "
                        f"(date: {conn.get('date', '?')})"
                    )

    return {
        "query": query,
        "extracted_entities": {
            "people": entities["people"],
            "places": entities["places"],
            "concepts": entities["concepts"],
        },
        "vector_results": vector_results,
        "graph_context": graph_context,
        "tuning": {
            "feedback_rerank": rerank_meta,
        },
        "formatted_context": "\n".join(context_parts),
    }
