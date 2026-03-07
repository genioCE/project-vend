import bisect
import os
import logging
import re
import time
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.embeddings import embed_query, embed_texts
from app.vectorstore import (
    collection_exists,
    get_collection,
    search,
    get_stats,
)
from app.corpus_utils import (
    dedupe_files_by_filename,
    parse_date_from_content,
    parse_date_from_filename,
    RequestIdMiddleware,
    setup_logging,
)

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="Corpus Embeddings Service", version="1.0.0")
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


CORPUS_PATH = os.environ.get("CORPUS_PATH", "/corpus")


def parse_positive_int(value: str | None, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except ValueError:
        pass
    return fallback


CORPUS_CACHE_TTL_SECONDS = parse_positive_int(
    os.environ.get("CORPUS_CACHE_TTL_SECONDS"), 300
)
MAX_KEYWORD_MATCHES = parse_positive_int(
    os.environ.get("MAX_KEYWORD_MATCHES"), 500
)
_corpus_cache: list[dict] | None = None
_corpus_cache_signature: tuple[int, int] | None = None
_corpus_cache_expires_at: float = 0.0


def require_index():
    """Raise 503 if the index hasn't been built yet."""
    if not collection_exists():
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Run: docker compose run --rm ingest",
        )


# --- Request/Response models ---

class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("query must not be blank")
        return text


class ThemesRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("topic must not be blank")
        return text


class KeywordRequest(BaseModel):
    keyword: str = Field(min_length=1, max_length=256)
    context_words: int = Field(default=100, ge=10, le=500)

    @field_validator("keyword")
    @classmethod
    def validate_keyword(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("keyword must not be blank")
        return text


def get_file_date(file_path: Path) -> str:
    date_str = parse_date_from_filename(file_path.name)
    if date_str is None:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            date_str = parse_date_from_content(text)
        except Exception:
            pass
    return date_str or "unknown"


def get_corpus_signature(files: list[Path]) -> tuple[int, int]:
    """Stable enough signal to detect corpus changes without reading file contents."""
    latest_mtime_ns = 0
    for f in files:
        try:
            mtime_ns = f.stat().st_mtime_ns
        except OSError:
            continue
        if mtime_ns > latest_mtime_ns:
            latest_mtime_ns = mtime_ns
    return (len(files), latest_mtime_ns)


def load_corpus_files() -> list[dict]:
    """Load all .md files from corpus with dates and content."""
    global _corpus_cache, _corpus_cache_signature, _corpus_cache_expires_at

    now = time.time()
    if _corpus_cache is not None and now < _corpus_cache_expires_at:
        return _corpus_cache

    root = Path(CORPUS_PATH)
    discovered = sorted(root.rglob("*.md"))
    files, duplicates_skipped = dedupe_files_by_filename(discovered)
    signature = get_corpus_signature(files)

    if _corpus_cache is not None and signature == _corpus_cache_signature:
        _corpus_cache_expires_at = now + CORPUS_CACHE_TTL_SECONDS
        return _corpus_cache

    if duplicates_skipped > 0:
        logger.info(
            "Detected mirrored/duplicate basenames in corpus loader. "
            "Using %s unique files (skipped %s duplicates).",
            len(files),
            duplicates_skipped,
        )

    entries = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            date_str = get_file_date(f)
            word_matches = list(re.finditer(r'\S+', text))
            words = [m.group() for m in word_matches]
            word_starts = [m.start() for m in word_matches]
            entries.append({
                "filename": f.name,
                "date": date_str,
                "text": text,
                "word_count": len(words),
                "words": words,
                "word_starts": word_starts,
            })
        except Exception as e:
            logger.warning(f"Error reading {f.name}: {e}")

    _corpus_cache = entries
    _corpus_cache_signature = signature
    _corpus_cache_expires_at = now + CORPUS_CACHE_TTL_SECONDS

    return entries


# --- Routes ---

@app.get("/health")
async def health():
    index_ready = collection_exists()
    return {"status": "ok", "index_ready": index_ready}


@app.post("/embed")
async def embed_texts_endpoint(req: EmbedRequest):
    """Embed a list of texts into vectors. Stateless — no index required."""
    vectors = embed_texts(req.texts)
    return {"embeddings": vectors}


@app.post("/search")
async def search_corpus(req: SearchRequest):
    require_index()
    query_embedding = embed_query(req.query)
    results = search(query_embedding, top_k=req.top_k)

    passages = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            similarity = round(1 - (dist / 2), 4)
            passages.append({
                "text": doc,
                "date": meta.get("date", "unknown"),
                "source_file": meta.get("source_file", ""),
                "relevance_score": similarity,
                "word_count": meta.get("word_count", 0),
            })

    return {"query": req.query, "results": passages}


@app.get("/entries")
async def get_entries_by_date(start: str, end: str):
    """Fetch entries within a date range from the raw corpus files."""
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD.",
        )
    if start_dt > end_dt:
        raise HTTPException(
            status_code=400,
            detail="start must be less than or equal to end.",
        )

    entries = load_corpus_files()
    filtered = []
    for entry in entries:
        if entry["date"] == "unknown":
            continue
        try:
            entry_dt = datetime.strptime(entry["date"], "%Y-%m-%d")
        except ValueError:
            continue
        if start_dt <= entry_dt <= end_dt:
            filtered.append(entry)

    # Sort chronologically
    filtered.sort(key=lambda e: e["date"])

    return {
        "start_date": start,
        "end_date": end,
        "count": len(filtered),
        "entries": [
            {
                "date": e["date"],
                "filename": e["filename"],
                "word_count": e["word_count"],
                "text": e["text"],
            }
            for e in filtered
        ],
    }


@app.post("/themes")
async def find_themes(req: ThemesRequest):
    require_index()
    query_embedding = embed_query(req.topic)
    results = search(query_embedding, top_k=req.top_k)

    passages = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1 - (dist / 2), 4)
            passages.append({
                "text": doc,
                "date": meta.get("date", "unknown"),
                "source_file": meta.get("source_file", ""),
                "relevance_score": similarity,
            })

    # Sort chronologically to show evolution over time
    def sort_key(p: dict) -> str:
        d = p["date"]
        return d if d != "unknown" else "9999-99-99"

    passages.sort(key=sort_key)

    return {"topic": req.topic, "results": passages}


@app.get("/stats")
async def corpus_stats():
    require_index()
    stats = get_stats()
    return stats


@app.get("/recent")
async def recent_entries(n: int = Query(default=7, ge=1, le=30)):
    """Return the most recent n entries from the raw corpus."""
    entries = load_corpus_files()

    # Filter to entries with valid dates and sort descending
    dated = [e for e in entries if e["date"] != "unknown"]
    dated.sort(key=lambda e: e["date"], reverse=True)

    recent = dated[:n]

    return {
        "count": len(recent),
        "entries": [
            {
                "date": e["date"],
                "filename": e["filename"],
                "word_count": e["word_count"],
                "text": e["text"],
            }
            for e in recent
        ],
    }


@app.post("/keyword")
async def keyword_search(req: KeywordRequest):
    """Literal text search across all corpus files with surrounding context."""
    entries = load_corpus_files()
    keyword_lower = req.keyword.lower()
    matches = []
    truncated = False

    for entry in entries:
        text = entry["text"]
        text_lower = text.lower()
        words = entry["words"]
        word_starts = entry["word_starts"]

        # Find all occurrences
        start_idx = 0
        while True:
            pos = text_lower.find(keyword_lower, start_idx)
            if pos == -1:
                break

            # Find the word index closest to this character position (O(log W))
            word_idx = max(0, bisect.bisect_right(word_starts, pos) - 1)

            # Extract context window
            context_start = max(0, word_idx - req.context_words)
            context_end = min(len(words), word_idx + req.context_words)
            context = " ".join(words[context_start:context_end])

            # Add ellipsis if truncated
            if context_start > 0:
                context = "..." + context
            if context_end < len(words):
                context = context + "..."

            matches.append({
                "date": entry["date"],
                "filename": entry["filename"],
                "context": context,
            })
            if len(matches) >= MAX_KEYWORD_MATCHES:
                truncated = True
                break

            start_idx = pos + len(req.keyword)

        if truncated:
            break

    # Sort by date
    matches.sort(key=lambda m: m["date"] if m["date"] != "unknown" else "9999-99-99")

    return {
        "keyword": req.keyword,
        "total_matches": len(matches),
        "results": matches,
        "truncated": truncated,
    }
