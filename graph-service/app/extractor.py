"""Entity extraction from journal entries using spaCy NER and style-aware heuristics."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TypedDict

import spacy
from spacy.tokens import Span

from .corpus_utils.extraction_lexicons import (
    ARCHETYPE_PATTERNS as _SHARED_ARCHETYPE_PATTERNS,
    DECISION_PATTERNS as _SHARED_DECISION_PATTERNS,
    EMOTION_LEXICON as _SHARED_EMOTION_LEXICON,
    STOP_CONCEPTS as _SHARED_STOP_CONCEPTS,
    TRANSITION_PATTERNS as _SHARED_TRANSITION_PATTERNS,
)
from .style_profile import canonicalize_alias, get_entity_aliases, get_style_profile

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

# Import from shared lexicons (single source of truth)
DEFAULT_EMOTION_LEXICON = _SHARED_EMOTION_LEXICON
DEFAULT_DECISION_PATTERNS = _SHARED_DECISION_PATTERNS
DEFAULT_ARCHETYPE_PATTERNS = _SHARED_ARCHETYPE_PATTERNS
DEFAULT_STOP_CONCEPTS = _SHARED_STOP_CONCEPTS
DEFAULT_TRANSITION_PATTERNS = _SHARED_TRANSITION_PATTERNS

STYLE_CADENCE_WORDS = {
    "presence",
    "inhabited",
    "embody",
    "rhythm",
    "flow",
    "energy",
    "spirit",
    "surrender",
    "alignment",
    "integrity",
    "clarity",
}

TRANSITION_BANNED_HEAD_WORDS = {
    "am",
    "are",
    "be",
    "being",
    "do",
    "doing",
    "get",
    "getting",
    "go",
    "going",
    "had",
    "has",
    "have",
    "having",
    "is",
    "was",
    "were",
}
TRANSITION_BANNED_TOKENS = {
    "today",
    "tomorrow",
    "yesterday",
    "tonight",
    "week",
    "month",
    "year",
}

STYLE_PROFILE = get_style_profile()
ENTITY_ALIASES = get_entity_aliases()


def _merge_keyword_map(defaults: dict[str, list[str]], additions: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for raw_key, values in defaults.items():
        key = raw_key.strip()
        deduped = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        merged[key] = deduped

    for raw_key, values in additions.items():
        key = raw_key.strip()
        if not key:
            continue
        bucket = merged.setdefault(key, [])
        seen = set(bucket)
        for value in values:
            normalized = value.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                bucket.append(normalized)

    return merged


def _merge_list(defaults: list[str], additions: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in defaults + additions:
        item = value.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _build_stop_concepts(defaults: set[str], profile: dict) -> set[str]:
    stop = {s.lower().strip() for s in defaults}
    for raw in profile.get("stop_concepts_add", []):
        normalized = raw.lower().strip()
        if normalized:
            stop.add(normalized)
    for raw in profile.get("stop_concepts_remove", []):
        normalized = raw.lower().strip()
        if normalized:
            stop.discard(normalized)
    return stop


EMOTION_LEXICON = _merge_keyword_map(
    DEFAULT_EMOTION_LEXICON, STYLE_PROFILE.get("emotion_lexicon", {})
)
ARCHETYPE_PATTERNS = _merge_keyword_map(
    DEFAULT_ARCHETYPE_PATTERNS, STYLE_PROFILE.get("archetype_patterns", {})
)
DECISION_PATTERNS = _merge_list(
    DEFAULT_DECISION_PATTERNS, STYLE_PROFILE.get("decision_patterns_add", [])
)
STOP_CONCEPTS = _build_stop_concepts(DEFAULT_STOP_CONCEPTS, STYLE_PROFILE)
CONCEPT_BOOSTS = {
    key.lower().strip(): float(value)
    for key, value in STYLE_PROFILE.get("concept_boosts", {}).items()
}
PHRASE_BOOSTS = {
    key.lower().strip(): float(value)
    for key, value in STYLE_PROFILE.get("phrase_boosts", {}).items()
}
TRANSITION_PATTERNS = _merge_list(
    DEFAULT_TRANSITION_PATTERNS,
    STYLE_PROFILE.get("transition_patterns_add", []),
)

DECISION_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in DECISION_PATTERNS]
TRANSITION_REGEXES: list[re.Pattern[str]] = []
for pattern in TRANSITION_PATTERNS:
    try:
        TRANSITION_REGEXES.append(re.compile(pattern, re.IGNORECASE))
    except re.error as err:
        logger.warning("Skipping invalid transition pattern '%s': %s", pattern, err)

EMOTION_KEYWORDS_FLAT = [kw for values in EMOTION_LEXICON.values() for kw in values]
EMOTION_NAMES = {name.lower() for name in EMOTION_LEXICON.keys()}


class TransitionResult(TypedDict):
    from_concept: str
    to_concept: str
    phrase: str
    strength: float


class ExtractionResult(TypedDict):
    people: list[str]
    places: list[str]
    concepts: list[str]
    emotions: list[dict]
    decisions: list[str]
    archetypes: list[dict]
    transitions: list[TransitionResult]


def _normalize_spaces(value: str) -> str:
    return " ".join(value.split()).strip()


def _normalize_named_entity(raw: str, category: str) -> str:
    normalized = _normalize_spaces(raw)
    if not normalized:
        return ""
    canonical = canonicalize_alias(category, normalized, ENTITY_ALIASES)
    return _normalize_spaces(canonical)


def _normalize_concept_text(raw: str) -> str:
    normalized = raw.lower().strip()
    normalized = re.sub(r"[\n\t]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s'/-]", " ", normalized)
    normalized = _normalize_spaces(normalized)
    normalized = re.sub(r"^(?:the|a|an|my|our|your|his|her|their|this|that)\s+", "", normalized)
    return normalized


def _canonicalize_concept(raw: str) -> str:
    normalized = _normalize_concept_text(raw)
    if not normalized:
        return ""
    canonical = canonicalize_alias("concepts", normalized, ENTITY_ALIASES)
    canonical = _normalize_concept_text(canonical)
    return canonical


def _valid_concept_from_chunk(chunk: Span, concept: str) -> bool:
    if len(concept) < 3:
        return False
    if concept in STOP_CONCEPTS:
        return False
    if chunk.root.is_stop:
        return False
    if all(tok.is_stop or tok.is_punct for tok in chunk):
        return False
    return True


def _extract_people(doc) -> list[str]:
    seen: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        name = _normalize_named_entity(ent.text, "people")
        if len(name) <= 1:
            continue
        key = name.lower()
        if key not in seen:
            seen[key] = name
    return list(seen.values())


def _extract_places(doc) -> list[str]:
    seen: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ not in ("GPE", "LOC", "FAC"):
            continue
        name = _normalize_named_entity(ent.text, "places")
        if len(name) <= 1:
            continue
        key = name.lower()
        if key not in seen:
            seen[key] = name
    return list(seen.values())


def _score_concepts(doc, text_lower: str) -> list[str]:
    base_counts: Counter[str] = Counter()
    style_scores: Counter[str] = Counter()

    for chunk in doc.noun_chunks:
        concept = _canonicalize_concept(chunk.text)
        if not concept or not _valid_concept_from_chunk(chunk, concept):
            continue
        base_counts[concept] += 1

    for sent in doc.sents:
        sent_lower = sent.text.lower()
        emotion_hits = sum(1 for kw in EMOTION_KEYWORDS_FLAT if kw in sent_lower)
        has_decision = any(rx.search(sent.text) for rx in DECISION_REGEXES)
        has_transition = " from " in f" {sent_lower} " and " to " in f" {sent_lower} "
        cadence_hits = sum(1 for word in STYLE_CADENCE_WORDS if word in sent_lower)
        phrase_boost = sum(weight for phrase, weight in PHRASE_BOOSTS.items() if phrase in sent_lower)

        local_boost = 0.0
        local_boost += min(emotion_hits * 0.12, 0.72)
        if has_decision:
            local_boost += 0.55
        if has_transition:
            local_boost += 0.45
        local_boost += min(cadence_hits * 0.1, 0.3)
        local_boost += phrase_boost

        for chunk in sent.noun_chunks:
            concept = _canonicalize_concept(chunk.text)
            if not concept or not _valid_concept_from_chunk(chunk, concept):
                continue
            style_scores[concept] += 1.0 + local_boost + CONCEPT_BOOSTS.get(concept, 0.0)

    combined: list[tuple[str, float]] = []
    for concept, count in base_counts.items():
        score = float(count) + float(style_scores.get(concept, 0.0))
        combined.append((concept, score))

    combined.sort(
        key=lambda item: (
            item[1],
            base_counts[item[0]],
            len(item[0]),
        ),
        reverse=True,
    )

    return [concept for concept, _ in combined[:15]]


def _extract_emotions(text_lower: str) -> list[dict]:
    emotions: list[dict] = []
    for emotion, keywords in EMOTION_LEXICON.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if not matches:
            continue
        intensity = min(len(matches) / 3.0, 1.0)
        emotions.append(
            {
                "emotion": emotion,
                "intensity": round(intensity, 2),
                "keywords": matches[:3],
            }
        )
    return emotions


def _extract_decisions(text: str) -> list[str]:
    decisions: list[str] = []
    for regex in DECISION_REGEXES:
        for match in regex.finditer(text):
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 120)
            context = text[start:end].strip().replace("\n", " ")

            for sep in [".", "!", "?"]:
                idx = context.find(sep, match.end() - start)
                if idx > 0:
                    context = context[: idx + 1]
                    break

            context = _normalize_spaces(context)
            if context and context not in decisions:
                decisions.append(context)
    return decisions[:5]


def _extract_archetypes(text_lower: str) -> list[dict]:
    archetypes: list[dict] = []
    for archetype, keywords in ARCHETYPE_PATTERNS.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if len(matches) < 2:
            continue
        strength = min(len(matches) / 4.0, 1.0)
        archetypes.append(
            {
                "archetype": archetype,
                "strength": round(strength, 2),
                "keywords": matches[:3],
            }
        )
    return archetypes


def _clean_transition_candidate(raw: str) -> str:
    candidate = _normalize_concept_text(raw)
    candidate = re.sub(r"\s+(?:state|mode|energy|feeling|feelings)$", "", candidate)
    words = candidate.split()
    if len(words) > 4:
        return ""
    if words and words[0] in TRANSITION_BANNED_HEAD_WORDS:
        return ""
    if any(word in TRANSITION_BANNED_TOKENS for word in words):
        return ""
    if candidate in STOP_CONCEPTS:
        return ""
    return candidate


def _extract_transitions(text: str) -> list[TransitionResult]:
    transitions: list[TransitionResult] = []
    seen: set[str] = set()

    for regex in TRANSITION_REGEXES:
        for match in regex.finditer(text):
            source_raw = match.groupdict().get("source", "")
            target_raw = match.groupdict().get("target", "")
            source = _canonicalize_concept(_clean_transition_candidate(source_raw))
            target = _canonicalize_concept(_clean_transition_candidate(target_raw))

            if not source or not target or source == target:
                continue

            key = f"{source}->{target}"
            if key in seen:
                continue
            seen.add(key)

            snippet = _normalize_spaces(text[max(0, match.start() - 20): min(len(text), match.end() + 20)])
            strength = 1.0
            lower_match = match.group(0).lower()
            if any(cue in lower_match for cue in ["moved", "move", "shift", "transition", "turned"]):
                strength += 0.4
            if source in EMOTION_NAMES or target in EMOTION_NAMES:
                strength += 0.2

            transitions.append(
                {
                    "from_concept": source,
                    "to_concept": target,
                    "phrase": snippet,
                    "strength": round(strength, 2),
                }
            )

    return transitions[:12]


def extract_entities(text: str) -> ExtractionResult:
    """Extract people, places, concepts, emotions, decisions, archetypes, and transitions from text."""
    doc = nlp(text[:100000])  # cap to avoid memory issues on huge texts
    text_lower = text.lower()

    people = _extract_people(doc)
    places = _extract_places(doc)
    concepts = _score_concepts(doc, text_lower)
    emotions = _extract_emotions(text_lower)
    decisions = _extract_decisions(text)
    archetypes = _extract_archetypes(text_lower)
    raw_transitions = _extract_transitions(text)
    concept_set = set(concepts)
    transitions = [
        item
        for item in raw_transitions
        if item["from_concept"] in EMOTION_NAMES
        or item["to_concept"] in EMOTION_NAMES
        or item["from_concept"] in concept_set
        or item["to_concept"] in concept_set
    ]

    return {
        "people": people,
        "places": places,
        "concepts": concepts,
        "emotions": emotions,
        "decisions": decisions,
        "archetypes": archetypes,
        "transitions": transitions,
    }
