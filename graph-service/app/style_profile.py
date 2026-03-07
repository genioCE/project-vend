"""Optional style/alias config loaders for GraphRAG extraction tuning."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STYLE_PROFILE_PATH = os.environ.get(
    "GRAPH_STYLE_PROFILE_PATH", "/service/config/style_profile.json"
)
STYLE_PROFILE_JSON = os.environ.get("GRAPH_STYLE_PROFILE_JSON")

AUTO_TUNING_PATH = os.environ.get(
    "GRAPH_AUTO_TUNING_PATH", "/service/data/feedback_tuning.json"
)

ENTITY_ALIASES_PATH = os.environ.get(
    "GRAPH_ENTITY_ALIASES_PATH", "/service/config/entity_aliases.json"
)
ENTITY_ALIASES_JSON = os.environ.get("GRAPH_ENTITY_ALIASES_JSON")


DEFAULT_STYLE_PROFILE: dict[str, Any] = {
    "emotion_lexicon": {},
    "archetype_patterns": {},
    "decision_patterns_add": [],
    "stop_concepts_add": [],
    "stop_concepts_remove": [],
    "concept_boosts": {},
    "phrase_boosts": {},
    "transition_patterns_add": [],
}

DEFAULT_ENTITY_ALIASES: dict[str, dict[str, str]] = {
    "people": {},
    "places": {},
    "concepts": {},
}


def _read_json_string(raw: str, source_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        logger.warning("Failed to parse %s JSON: %s", source_name, err)
        return {}

    if not isinstance(parsed, dict):
        logger.warning("Ignoring %s JSON: root must be an object", source_name)
        return {}

    return parsed


def _read_json_file(path_value: str | None, source_name: str) -> dict[str, Any]:
    if not path_value:
        return {}

    path = Path(path_value)
    if not path.exists():
        return {}

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("Failed to read %s file %s: %s", source_name, path, err)
        return {}

    if not isinstance(parsed, dict):
        logger.warning("Ignoring %s file %s: root must be an object", source_name, path)
        return {}

    return parsed


def _load_config(raw_json: str | None, path_value: str | None, source_name: str) -> dict[str, Any]:
    if raw_json:
        parsed = _read_json_string(raw_json, source_name)
        if parsed:
            return parsed

    return _read_json_file(path_value, source_name)


def _normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        item = " ".join(value.strip().split())
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def _normalize_string_map(values: Any) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}

    normalized: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            continue
        key = raw_key.strip().lower()
        value = " ".join(raw_value.strip().split())
        if key and value:
            normalized[key] = value
    return normalized


def _normalize_number_map(values: Any) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}

    normalized: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip().lower()
        if not key:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        normalized[key] = value
    return normalized


def _normalize_nested_list_map(values: Any) -> dict[str, list[str]]:
    if not isinstance(values, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for raw_key, raw_values in values.items():
        if not isinstance(raw_key, str):
            continue
        key = " ".join(raw_key.strip().split())
        if not key:
            continue
        entries = _normalize_list(raw_values)
        if entries:
            normalized[key] = entries
    return normalized


@lru_cache(maxsize=1)
def get_style_profile() -> dict[str, Any]:
    raw = _load_config(STYLE_PROFILE_JSON, STYLE_PROFILE_PATH, "GRAPH_STYLE_PROFILE")

    profile = dict(DEFAULT_STYLE_PROFILE)
    profile["emotion_lexicon"] = _normalize_nested_list_map(raw.get("emotion_lexicon"))
    profile["archetype_patterns"] = _normalize_nested_list_map(raw.get("archetype_patterns"))
    profile["decision_patterns_add"] = _normalize_list(raw.get("decision_patterns_add"))
    profile["stop_concepts_add"] = _normalize_list(raw.get("stop_concepts_add"))
    profile["stop_concepts_remove"] = _normalize_list(raw.get("stop_concepts_remove"))
    profile["concept_boosts"] = _normalize_number_map(raw.get("concept_boosts"))
    profile["phrase_boosts"] = _normalize_number_map(raw.get("phrase_boosts"))
    profile["transition_patterns_add"] = _normalize_list(raw.get("transition_patterns_add"))

    # Merge auto-tuned concept_boosts (manual values take precedence)
    auto_tuning = _read_json_file(AUTO_TUNING_PATH, "GRAPH_AUTO_TUNING")
    if auto_tuning:
        auto_boosts = _normalize_number_map(auto_tuning.get("concept_boosts"))
        if auto_boosts:
            merged = dict(auto_boosts)
            merged.update(profile["concept_boosts"])  # manual wins
            profile["concept_boosts"] = merged
            logger.info("Merged %d auto-tuned concept boosts", len(auto_boosts))

    if raw:
        logger.info("Loaded GraphRAG style profile")

    return profile


@lru_cache(maxsize=1)
def get_entity_aliases() -> dict[str, dict[str, str]]:
    raw = _load_config(ENTITY_ALIASES_JSON, ENTITY_ALIASES_PATH, "GRAPH_ENTITY_ALIASES")

    aliases = dict(DEFAULT_ENTITY_ALIASES)
    aliases["people"] = _normalize_string_map(raw.get("people"))
    aliases["places"] = _normalize_string_map(raw.get("places"))
    aliases["concepts"] = _normalize_string_map(raw.get("concepts"))

    if raw:
        logger.info("Loaded GraphRAG entity aliases")

    return aliases


def canonicalize_alias(category: str, value: str, aliases: dict[str, dict[str, str]] | None = None) -> str:
    """Return canonical text for a category using configured alias maps."""
    if aliases is None:
        aliases = get_entity_aliases()

    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return ""

    mapping = aliases.get(category, {})
    return mapping.get(cleaned.lower(), cleaned)


def _reload_extractor_globals() -> None:
    """Re-read style profile and update extractor module-level globals."""
    import re as _re

    import app.extractor as ext

    profile = get_style_profile()
    ext.STYLE_PROFILE = profile
    ext.CONCEPT_BOOSTS = {
        k.lower().strip(): float(v)
        for k, v in profile.get("concept_boosts", {}).items()
    }
    ext.PHRASE_BOOSTS = {
        k.lower().strip(): float(v)
        for k, v in profile.get("phrase_boosts", {}).items()
    }
    ext.EMOTION_LEXICON = ext._merge_keyword_map(
        ext.DEFAULT_EMOTION_LEXICON, profile.get("emotion_lexicon", {})
    )
    ext.ARCHETYPE_PATTERNS = ext._merge_keyword_map(
        ext.DEFAULT_ARCHETYPE_PATTERNS, profile.get("archetype_patterns", {})
    )
    ext.STOP_CONCEPTS = ext._build_stop_concepts(ext.DEFAULT_STOP_CONCEPTS, profile)
    ext.DECISION_PATTERNS = ext._merge_list(
        ext.DEFAULT_DECISION_PATTERNS, profile.get("decision_patterns_add", [])
    )
    ext.DECISION_REGEXES = [_re.compile(p, _re.IGNORECASE) for p in ext.DECISION_PATTERNS]
    ext.TRANSITION_PATTERNS = ext._merge_list(
        ext.DEFAULT_TRANSITION_PATTERNS, profile.get("transition_patterns_add", [])
    )
    ext.TRANSITION_REGEXES = []
    for pattern in ext.TRANSITION_PATTERNS:
        try:
            ext.TRANSITION_REGEXES.append(_re.compile(pattern, _re.IGNORECASE))
        except _re.error:
            pass
    ext.EMOTION_KEYWORDS_FLAT = [
        kw for values in ext.EMOTION_LEXICON.values() for kw in values
    ]
    ext.EMOTION_NAMES = {name.lower() for name in ext.EMOTION_LEXICON.keys()}


def reload_style_config() -> None:
    """Clear caches and re-initialize dependent modules."""
    get_style_profile.cache_clear()
    get_entity_aliases.cache_clear()
    _reload_extractor_globals()
