import json
import logging
import math
import os
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

_model: Optional[Any] = None


def _get_provider() -> str:
    return os.environ.get("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()


def get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")
    return _model


def _normalize_embedding(embedding: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in embedding))
    if norm <= 0:
        return embedding
    return [value / norm for value in embedding]


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    ollama_url = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
    model_name = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    timeout_seconds = max(
        1,
        int(os.environ.get("OLLAMA_EMBEDDING_TIMEOUT_SECONDS", "120")),
    )
    payload = json.dumps({
        "model": model_name,
        "input": texts,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)

    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        return [_normalize_embedding([float(value) for value in row]) for row in embeddings]

    # Compatibility fallback for single-item older Ollama response shape.
    embedding = data.get("embedding")
    if isinstance(embedding, list):
        return [_normalize_embedding([float(value) for value in embedding])]

    raise RuntimeError("Ollama embed response missing embeddings payload")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if _get_provider() == "ollama":
        return _ollama_embed(texts)

    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    if _get_provider() == "ollama":
        embeddings = _ollama_embed([query])
        if not embeddings:
            raise RuntimeError("Ollama embed returned no embeddings")
        return embeddings[0]

    model = get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()
