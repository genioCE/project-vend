"""
22 tool identity descriptions for the Gravity Model.

Each tool has a rich natural language description that defines its gravitational
signature — what kinds of queries it's native to, what it returns, and what
questions it answers. These descriptions are embedded using the same model as
the corpus (all-mpnet-base-v2) to create identity vectors.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class ToolIdentity:
    name: str
    description: str
    is_meta: bool = False
    always_active: bool = False  # Base tools that always fire (textual grounding)


TOOLS: list[ToolIdentity] = [
    # ── Search & Retrieval ──────────────────────────────────────────
    ToolIdentity(
        name="search_writings",
        description=(
            "Find journal entries about any person, concept, feeling, experience, or topic. "
            "The most general search tool — works for questions about people like Kyle or Matt, "
            "abstract concepts like silence or sovereignty, emotions like shame or fear, "
            "activities like climbing or work, and any open-ended exploration of the writing. "
            "Returns the actual text passages most relevant to the query."
        ),
        always_active=True,  # Provides textual grounding for all queries
    ),
    ToolIdentity(
        name="search_by_keyword",
        description=(
            "Find exact words and phrases in the writing. Literal text matching for "
            "specific names, places, terms, and quoted phrases. Use when the user "
            "wants to locate a specific word or phrase they know appears in the corpus, "
            "not semantic similarity but exact string match."
        ),
    ),
    ToolIdentity(
        name="get_entries_by_date",
        description=(
            "Retrieve all writing from a specific date range. Full text of entries "
            "within a time window, sorted chronologically. Use when the question asks "
            "for what was written during a particular period — days, weeks, or months."
        ),
    ),
    ToolIdentity(
        name="get_recent_entries",
        description=(
            "Get the most recent writing entries. What has been on the writer's mind "
            "lately, newest first. Use for questions about current state, recent "
            "thinking, or what's been happening recently in the writing."
        ),
    ),
    ToolIdentity(
        name="temporal_filter",
        description=(
            "Find entries matching specific psychological metric thresholds within a "
            "time range. Filter by measurable dimensions of inner state — entries where "
            "agency is above 0.7, or valence is below -0.3, or word count exceeds 2000. "
            "Quantitative filtering on numerical psychological dimensions."
        ),
    ),
    ToolIdentity(
        name="search_by_state",
        description=(
            "Find entries by psychological dimension: valence, activation, agency, "
            "certainty, relational openness, self-trust, time orientation, integration. "
            "Filter by inner state to find entries when the writer felt stuck, empowered, "
            "fragmented, uplifted, calm, or activated. Maps feelings to measured dimensions."
        ),
    ),
    # ── Pattern & Evolution ─────────────────────────────────────────
    ToolIdentity(
        name="find_recurring_themes",
        description=(
            "Track how a topic or theme evolves over time across the writing. "
            "Chronological passages showing how thinking about a subject has changed "
            "and grown. Use for questions about personal growth, recurring concerns, "
            "or evolving perspectives on a concept over time."
        ),
    ),
    ToolIdentity(
        name="trace_concept_evolution",
        description=(
            "Timeline of when and how a concept appears across the corpus. Concept "
            "emergence, frequency, and contextual shifts over time. Shows which entries "
            "contain the concept, what emotions and people co-occur with it. Use for "
            "tracing an idea's journey through the writing."
        ),
    ),
    ToolIdentity(
        name="get_concept_flows",
        description=(
            "Directed transitions between concepts. Movement patterns like fear to "
            "action, doubt to clarity, isolation to connection. What flows into and "
            "out of an idea. Use for questions about mindset shifts, psychological "
            "movement patterns, and how one state leads to another."
        ),
    ),
    ToolIdentity(
        name="search_themes",
        description=(
            "Find psychological and narrative theme patterns and their co-occurrences. "
            "Themes are 2-4 word patterns like 'cultivating resilience', 'seeking clarity', "
            "'struggling with discipline'. Returns entries linked to a theme and which "
            "other themes appear alongside it."
        ),
    ),
    # ── Graph & Relationship ────────────────────────────────────────
    ToolIdentity(
        name="find_connected_concepts",
        description=(
            "Explore what ideas, people, emotions, and entries are connected to a concept, "
            "person, or theme. Maps the relational neighborhood: what clusters around silence, "
            "who connects to Kyle, what emotions associate with sovereignty, what concepts "
            "relate to fear or shame. Essential for understanding meaning, context, and "
            "the web of associations around any subject in the writing."
        ),
    ),
    ToolIdentity(
        name="find_entity_relationships",
        description=(
            "Map a person's presence across the writing. All entries mentioning them, "
            "with associated concepts and emotions. Relational mapping. Use for questions "
            "about specific people — how they show up, what themes surround them, "
            "what the relationship looks like across time."
        ),
    ),
    # ── Psychological & Archetypal ──────────────────────────────────
    ToolIdentity(
        name="get_entry_analysis",
        description=(
            "Deep analysis of a single entry: summary, themes, entities, decisions, "
            "and full 8-dimension psychological state profile. Use when a specific entry "
            "has been identified and you need the complete analytical breakdown of that "
            "particular day's writing."
        ),
    ),
    ToolIdentity(
        name="get_archetype_patterns",
        description=(
            "Archetypal patterns found in the writing: Creator, Warrior, Healer, Sage, "
            "Lover, Sovereign, Integrator. Frequency, strength, and associated emotions. "
            "Use for questions about mythic patterns, roles the writer embodies, "
            "archetypal energy, and which archetypes are most active."
        ),
    ),
    # ── Quantitative & Temporal ─────────────────────────────────────
    ToolIdentity(
        name="query_time_series",
        description=(
            "Any metric plotted over time: psychological dimensions, word count, "
            "archetype frequency, concept frequency, theme frequency. Time series data "
            "at entry, daily, or weekly granularity. Use for trend questions, tracking "
            "how a measurable quantity changes across dates."
        ),
    ),
    ToolIdentity(
        name="detect_anomalies",
        description=(
            "Find outlier entries where a metric deviates significantly from baseline. "
            "Statistical anomaly detection using z-scores. Use for questions about "
            "unusual days, extreme states, sudden shifts, or finding when something "
            "was notably different from the norm."
        ),
    ),
    ToolIdentity(
        name="correlate_metrics",
        description=(
            "Discover statistical relationships between two metrics over time. Pearson "
            "correlation coefficient with p-value and interpretation. Use for questions "
            "about whether two dimensions move together — does agency correlate with "
            "valence? Does word count predict integration?"
        ),
    ),
    ToolIdentity(
        name="get_metric_summary",
        description=(
            "Summary statistics for any metric: mean, median, standard deviation, min, "
            "max, entry count, and current trend direction. Quick quantitative overview "
            "of a dimension's behavior over a period. Use for baseline questions and "
            "understanding typical ranges."
        ),
    ),
    ToolIdentity(
        name="compare_periods",
        description=(
            "Side-by-side comparison of two time windows: concepts, emotions, archetypes. "
            "Use for questions about how different periods compare — was January different "
            "from March? How did summer compare to fall? What changed between two eras?"
        ),
    ),
    # ── Meta & Context ──────────────────────────────────────────────
    ToolIdentity(
        name="get_writing_stats",
        description=(
            "Corpus-level overview: total word count, date range, number of entries, "
            "average length, entries per year. Use for meta-questions about the writing "
            "practice itself — how much has been written, how consistent the habit is."
        ),
        is_meta=True,
    ),
    ToolIdentity(
        name="list_available_metrics",
        description=(
            "Discover all queryable metric names and types. Returns dimension metrics, "
            "entry statistics, archetype metrics, theme metrics, and concept metrics "
            "available in the system. Use when needing to know what can be measured "
            "or tracked — a catalog of available data dimensions."
        ),
        is_meta=True,
    ),
    ToolIdentity(
        name="get_decision_context",
        description=(
            "Recorded decisions and their surrounding emotional and conceptual state. "
            "What was decided, when, and what the inner landscape looked like at the time. "
            "Use for questions about choices made, decision patterns, and the emotional "
            "context around commitments and turning points."
        ),
    ),
]

TOOL_NAMES = [t.name for t in TOOLS]
META_TOOLS = {t.name for t in TOOLS if t.is_meta}
ALWAYS_ACTIVE_TOOLS = {t.name for t in TOOLS if t.always_active}


def _descriptions_hash() -> str:
    """Hash of all descriptions for cache invalidation."""
    content = json.dumps([t.description for t in TOOLS], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_identity_vectors(model=None) -> np.ndarray:
    """
    Embed all 22 tool identity descriptions. Returns (22, 768) ndarray.
    Caches to results/identity_vectors.npz; recomputes if descriptions change.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RESULTS_DIR / "identity_vectors.npz"
    current_hash = _descriptions_hash()

    if cache_path.exists():
        data = np.load(cache_path)
        if data.get("hash", "") == current_hash:
            return data["vectors"]

    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")

    descriptions = [t.description for t in TOOLS]
    vectors = model.encode(descriptions, normalize_embeddings=True)
    vectors = np.array(vectors, dtype=np.float32)

    np.savez(cache_path, vectors=vectors, hash=current_hash)
    return vectors
