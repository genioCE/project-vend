"""Local extraction engine — pure Python, zero API calls, zero heavy dependencies.

Uses regex + keyword lexicons for entity, emotion, decision, archetype, and theme
extraction. Entity detection patterns ported from the gravity decomposer
(mcp-server/src/gravity/decompose.ts) for hardened multi-word and alphanumeric
entity recognition.
"""

from __future__ import annotations

import re
from collections import Counter

from .corpus_utils.extraction_lexicons import (
    ARCHETYPE_PATTERNS,
    DECISION_PATTERNS,
    EMOTION_LEXICON,
    STOP_CONCEPTS,
    VALID_ENTITY_TYPES,
)

# ─── Entity extraction (ported from gravity decomposer) ──────────────────

# Words that look capitalized but aren't entities (sentence starters, common nouns)
# Merged from gravity decomposer's entitySkipWords + analysis domain words
ENTITY_SKIP_WORDS: set[str] = {
    # Function words / sentence starters
    "how", "what", "when", "where", "why", "who", "which", "the", "am",
    "and", "but", "for", "are", "was", "were", "has", "have", "had",
    "been", "being", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "not", "this", "that", "these",
    "those", "with", "from", "about", "into", "through", "during",
    "before", "after", "above", "below", "between",
    # Imperative verbs
    "tell", "describe", "explain", "show", "list", "give", "find",
    "get", "look", "help", "make", "let", "suggest", "identify",
    "recall", "note", "consider", "notice",
    # Common nouns
    "entries", "entry", "dreams", "dream", "tension", "themes",
    "theme", "patterns", "pattern", "decisions", "decision",
    "relationship", "relationships", "connection", "connections",
    "energy", "writing", "writings", "thoughts", "feelings",
    "my", "times", "moments", "everything", "something", "nothing",
    "top", "early", "late", "most", "gaps", "days", "places",
    "areas", "ways", "words", "signs", "pieces", "parts",
    "points", "steps", "levels",
    # Months
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    # Verbs that appear capitalized at sentence start
    "today", "yesterday", "tomorrow", "sometimes", "recently",
    "maybe", "also", "just", "still", "even", "already",
    "really", "actually", "finally", "suddenly", "basically",
}

# Stop words for concept/theme extraction
THEME_STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    "shall", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "about", "what", "which", "who", "whom", "this", "that",
    "these", "those", "and", "but", "or", "if", "while", "because",
    "until", "although", "since", "whether",
    "i", "me", "my", "mine", "myself", "we", "our", "ours",
    "you", "your", "he", "she", "it", "they", "them", "their",
    # Functional verbs
    "feel", "felt", "feels", "feeling", "think", "thought",
    "know", "knew", "want", "need", "like", "make", "made",
    "go", "went", "going", "see", "saw", "come", "came",
    "take", "took", "give", "gave", "say", "said", "tell", "told",
    "get", "got", "getting", "write", "wrote", "written", "writing",
    "talk", "talked", "look", "looked",
    "happen", "happened", "happens", "start", "started", "starting",
    "keep", "kept", "try", "tried", "seem", "seemed",
    # Common nouns that aren't themes
    "thing", "things", "stuff", "way", "lot", "time", "times",
    "part", "kind", "sort", "type", "day", "days", "week", "month",
    "year", "today", "yesterday", "tomorrow", "moment", "morning",
    "evening", "night", "entry", "entries",
}


def extract_entities_local(text: str) -> list[dict]:
    """Extract typed entities using regex patterns from the gravity decomposer.

    Returns list of {"name": str, "type": EntityType} dicts.
    Handles multi-word capitalized phrases, alphanumeric names (StarSpace46),
    and filters false positives with comprehensive skip-word lists.
    """
    entities: list[dict] = []
    seen: set[str] = set()

    # 1. Alphanumeric entity names (StarSpace46, etc.)
    for match in re.finditer(r"\b([A-Z][a-zA-Z]*\d+[a-zA-Z0-9]*)\b", text):
        name = match.group(1)
        key = name.lower()
        if key not in seen:
            seen.add(key)
            entities.append({"name": name, "type": "organization"})

    # 2. Multi-word and single-word capitalized phrases
    #    (Plaza District, Oklahoma City, Kyle, Mom)
    for match in re.finditer(r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b", text):
        phrase = match.group(1)
        phrase_words = phrase.split()
        key = phrase.lower()

        if key in seen:
            continue

        # Skip single words at sentence boundaries that are common non-entities
        is_at_sentence_start = match.start() == 0 or text[match.start() - 2: match.start()].rstrip() in (".", "!", "?", "\n")
        is_single = len(phrase_words) == 1

        if is_single and is_at_sentence_start and key in ENTITY_SKIP_WORDS:
            continue

        # Filter: all words are skip words → not an entity
        meaningful = [w for w in phrase_words if w.lower() not in ENTITY_SKIP_WORDS]
        if not meaningful:
            continue

        # Guess entity type from heuristics
        entity_type = _guess_entity_type(phrase)

        seen.add(key)
        entities.append({"name": phrase, "type": entity_type})

    return entities[:20]  # cap at 20


def _guess_entity_type(name: str) -> str:
    """Heuristic entity typing based on name patterns."""
    lower = name.lower()

    # Known place indicators
    place_words = {"city", "county", "district", "park", "street", "avenue",
                   "boulevard", "lake", "river", "mountain", "valley", "beach",
                   "island", "bridge", "tower", "plaza", "center", "church",
                   "school", "university", "hospital", "station"}
    if any(w in lower.split() for w in place_words):
        return "place"

    # Known org indicators
    org_words = {"inc", "corp", "llc", "ltd", "company", "group", "foundation",
                 "institute", "association", "studio", "labs", "media", "tech"}
    if any(w in lower.split() for w in org_words):
        return "organization"

    # Single short name → likely person (Kyle, Mom, Dad, Sarah)
    words = name.split()
    if len(words) == 1 and len(name) <= 10:
        return "person"

    # Two-word names are often people (Kyle Smith) or places (Oklahoma City)
    if len(words) == 2:
        return "person"  # default guess for two-word caps

    return "concept"


def extract_emotions_local(text: str) -> list[dict]:
    """Extract emotions using keyword lexicon matching.

    Returns list of {"emotion": str, "intensity": float, "keywords": [str]}.
    """
    text_lower = text.lower()
    emotions: list[dict] = []

    for emotion, keywords in EMOTION_LEXICON.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if not matches:
            continue
        intensity = min(len(matches) / 3.0, 1.0)
        emotions.append({
            "emotion": emotion,
            "intensity": round(intensity, 2),
            "keywords": matches[:3],
        })

    return emotions


def extract_decisions_local(text: str) -> list[str]:
    """Extract decision/commitment statements using regex patterns.

    Returns list of decision text snippets.
    """
    decisions: list[str] = []
    compiled = [re.compile(p, re.IGNORECASE) for p in DECISION_PATTERNS]

    for regex in compiled:
        for match in regex.finditer(text):
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 120)
            context = text[start:end].strip().replace("\n", " ")

            # Truncate at sentence boundary
            for sep in [".", "!", "?"]:
                idx = context.find(sep, match.end() - start)
                if idx > 0:
                    context = context[:idx + 1]
                    break

            context = " ".join(context.split())  # normalize whitespace
            if context and context not in decisions:
                decisions.append(context)

    return decisions[:12]


def extract_archetypes_local(text: str) -> list[dict]:
    """Extract archetype patterns using keyword matching.

    Returns list of {"archetype": str, "strength": float, "keywords": [str]}.
    Requires ≥2 keyword hits to register.
    """
    text_lower = text.lower()
    archetypes: list[dict] = []

    for archetype, keywords in ARCHETYPE_PATTERNS.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if len(matches) < 2:
            continue
        strength = min(len(matches) / 4.0, 1.0)
        archetypes.append({
            "archetype": archetype,
            "strength": round(strength, 2),
            "keywords": matches[:3],
        })

    return archetypes


def extract_themes_local(text: str, max_terms: int = 8) -> list[str]:
    """Extract themes using frequency-based term extraction with emotion-proximity boosting.

    Returns list of abstract theme strings, ranked by relevance.
    Improves on the basic mock _extract_key_terms by:
    - Boosting terms near emotional keywords
    - Preferring multi-word phrases
    - Filtering stop concepts more aggressively
    """
    text_lower = text.lower()

    # Collect all emotion keywords present for proximity boosting
    emotion_keywords_flat: set[str] = set()
    for keywords in EMOTION_LEXICON.values():
        for kw in keywords:
            if kw in text_lower:
                emotion_keywords_flat.add(kw)

    # Extract 3+ letter words, filter stop words
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text_lower)
    all_stop = THEME_STOP_WORDS | STOP_CONCEPTS
    filtered = [w for w in words if w not in all_stop]

    # Count frequencies
    counts: Counter[str] = Counter(filtered)

    # Boost terms that appear near emotion words (within 5-word window)
    for i, word in enumerate(words):
        if word in emotion_keywords_flat:
            # Boost neighbors
            for j in range(max(0, i - 5), min(len(words), i + 6)):
                neighbor = words[j]
                if neighbor not in all_stop and neighbor != word:
                    counts[neighbor] += 0.5

    # Also extract bigrams for richer themes
    bigrams: Counter[str] = Counter()
    for i in range(len(filtered) - 1):
        bigram = f"{filtered[i]} {filtered[i + 1]}"
        bigrams[bigram] += 1

    # Merge: prefer bigrams that appear 2+ times
    merged: list[tuple[str, float]] = []
    used_in_bigrams: set[str] = set()
    for bigram, count in bigrams.most_common(20):
        if count >= 2:
            merged.append((bigram, float(count) * 2))  # boost bigrams
            for w in bigram.split():
                used_in_bigrams.add(w)

    for word, count in counts.most_common(30):
        if word not in used_in_bigrams:
            merged.append((word, float(count)))

    merged.sort(key=lambda x: (-x[1], x[0]))
    return [term for term, _ in merged[:max_terms]]
