"""
Shared caching utilities for gravity diagnostics.

Caches decomposition results to avoid repeated Claude API calls.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fragments import DecompositionResult, Fragment, FragmentType

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_PATH = RESULTS_DIR / "cached_decompositions.json"


def _fragment_to_dict(f: Fragment) -> dict:
    """Convert Fragment to JSON-serializable dict (excluding embedding)."""
    return {"type": f.type.value, "text": f.text}


def _fragment_from_dict(d: dict) -> Fragment:
    """Reconstruct Fragment from dict (embedding will be None)."""
    return Fragment(type=FragmentType(d["type"]), text=d["text"])


def _decomposition_to_dict(query: str, result: DecompositionResult) -> dict:
    """Convert DecompositionResult to JSON-serializable dict."""
    return {
        "query": query,
        "fragments": [_fragment_to_dict(f) for f in result.fragments],
        "primary_mass_index": result.primary_mass_index,
        "claude_reasoning": result.claude_reasoning,
    }


def _decomposition_from_dict(d: dict) -> DecompositionResult:
    """Reconstruct DecompositionResult from dict (embeddings will be None)."""
    fragments = [_fragment_from_dict(f) for f in d["fragments"]]
    return DecompositionResult(
        fragments=fragments,
        primary_mass_index=d["primary_mass_index"],
        claude_reasoning=d.get("claude_reasoning", ""),
    )


def load_cache() -> dict[str, DecompositionResult]:
    """Load cached decompositions from disk."""
    if not CACHE_PATH.exists():
        return {}

    with open(CACHE_PATH) as f:
        data = json.load(f)

    cache = {}
    for entry in data.get("decompositions", []):
        query = entry["query"]
        cache[query] = _decomposition_from_dict(entry)

    return cache


def save_cache(cache: dict[str, DecompositionResult]) -> None:
    """Save decomposition cache to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "decompositions": [
            _decomposition_to_dict(query, result)
            for query, result in cache.items()
        ]
    }

    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_or_decompose(
    query: str,
    cache: dict[str, DecompositionResult],
    skip_decompose: bool = False,
) -> DecompositionResult | None:
    """
    Get decomposition from cache or compute it.

    Returns None if skip_decompose=True and query not in cache.
    """
    if query in cache:
        return cache[query]

    if skip_decompose:
        return None

    from decompose import decompose
    result = decompose(query)
    cache[query] = result
    return result
