import os
import logging
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "corpus"


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        persist_path = os.environ.get("CHROMA_PERSIST_PATH", "/data/chroma")
        logger.info(f"Initializing ChromaDB at: {persist_path}")
        _client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def collection_exists() -> bool:
    client = get_client()
    collections = client.list_collections()
    return any(c.name == COLLECTION_NAME for c in collections)


def get_or_create_collection() -> chromadb.Collection:
    global _collection
    client = get_client()
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def delete_collection() -> None:
    global _collection
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except ValueError:
        pass
    _collection = None


def search(query_embedding: list[float], top_k: int = 5, where: dict | None = None) -> dict:
    collection = get_collection()
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    return results


def get_stats() -> dict:
    collection = get_collection()
    count = collection.count()

    # Get all metadata to compute stats
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data["metadatas"] or []

    dates = set()
    total_words = 0
    source_files = set()
    years: dict[str, int] = {}

    for meta in metadatas:
        if meta.get("date"):
            dates.add(meta["date"])
        if meta.get("word_count"):
            total_words += meta["word_count"]
        if meta.get("source_file"):
            source_files.add(meta["source_file"])
        if meta.get("year"):
            year_str = str(meta["year"])
            years[year_str] = years.get(year_str, 0) + 1

    sorted_dates = sorted(dates) if dates else []
    entry_count = len(source_files)

    return {
        "total_chunks": count,
        "total_entries": entry_count,
        "total_words": total_words,
        "date_range": {
            "earliest": sorted_dates[0] if sorted_dates else None,
            "latest": sorted_dates[-1] if sorted_dates else None,
        },
        "avg_words_per_entry": round(total_words / entry_count) if entry_count > 0 else 0,
        "entries_per_year": years,
    }
