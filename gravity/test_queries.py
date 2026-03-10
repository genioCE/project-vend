"""
20 test queries with expected activation patterns for validating the gravity field.

Each query specifies:
- query: raw natural language
- archetype: query archetype for grouping in analysis
- expected_primary_mass: the fragment text that should be primary mass
- expected_fragments: ground-truth fragment decomposition (type, text) pairs
- expected_active: tools that SHOULD activate
- expected_inactive: tools that should NOT activate
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QueryArchetype = Literal[
    "person_exploration",
    "concept_exploration",
    "temporal_arc",
    "state_query",
    "period_comparison",
    "theme_deep_dive",
    "decision_review",
    "archetypal_query",
    "relational_query",
    "keyword_search",
    "meta_query",
    "complex_multi_fragment",
]


@dataclass
class ExpectedFragment:
    """Ground-truth fragment annotation."""
    type: str  # concept, entity, temporal, emotional, relational, archetypal
    text: str
    is_primary: bool = False


@dataclass
class TestQuery:
    query: str
    archetype: QueryArchetype
    expected_primary_mass: str
    expected_fragments: list[ExpectedFragment] = field(default_factory=list)
    expected_active: list[str] = field(default_factory=list)
    expected_inactive: list[str] = field(default_factory=list)


TEST_QUERIES: list[TestQuery] = [
    # ── Person Exploration ──────────────────────────────────────────
    TestQuery(
        query="tell me about Kyle",
        archetype="person_exploration",
        expected_primary_mass="Kyle",
        expected_fragments=[
            ExpectedFragment(type="entity", text="Kyle", is_primary=True),
        ],
        expected_active=[
            "find_entity_relationships",
            "search_writings",
            "find_connected_concepts",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "detect_anomalies",
            "correlate_metrics",
        ],
    ),
    TestQuery(
        query="how does Matt show up in my writing",
        archetype="person_exploration",
        expected_primary_mass="Matt",
        expected_fragments=[
            ExpectedFragment(type="entity", text="Matt", is_primary=True),
            ExpectedFragment(type="relational", text="show up in writing"),
        ],
        expected_active=[
            "find_entity_relationships",
            "search_writings",
            "trace_concept_evolution",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "get_archetype_patterns",
        ],
    ),
    # ── Concept Exploration ─────────────────────────────────────────
    TestQuery(
        query="what does silence mean to me",
        archetype="concept_exploration",
        expected_primary_mass="silence",
        expected_fragments=[
            ExpectedFragment(type="concept", text="silence", is_primary=True),
            ExpectedFragment(type="relational", text="meaning"),
        ],
        expected_active=[
            "search_writings",
            "find_connected_concepts",
            "find_recurring_themes",
            "trace_concept_evolution",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "detect_anomalies",
        ],
    ),
    TestQuery(
        query="explore sovereignty in my journal",
        archetype="concept_exploration",
        expected_primary_mass="sovereignty",
        expected_fragments=[
            ExpectedFragment(type="concept", text="sovereignty", is_primary=True),
        ],
        expected_active=[
            "search_writings",
            "find_connected_concepts",
            "search_themes",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "correlate_metrics",
        ],
    ),
    # ── Temporal Arc ────────────────────────────────────────────────
    TestQuery(
        query="how have I changed since summer",
        archetype="temporal_arc",
        expected_primary_mass="change over time",
        expected_fragments=[
            ExpectedFragment(type="temporal", text="change over time", is_primary=True),
            ExpectedFragment(type="temporal", text="since summer"),
        ],
        expected_active=[
            "search_writings",
            "find_recurring_themes",
            "query_time_series",
            "compare_periods",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
        ],
    ),
    TestQuery(
        query="how has my relationship with silence changed since I started climbing",
        archetype="complex_multi_fragment",
        expected_primary_mass="silence",
        expected_fragments=[
            ExpectedFragment(type="concept", text="silence", is_primary=True),
            ExpectedFragment(type="relational", text="relationship with"),
            ExpectedFragment(type="temporal", text="changed since"),
            ExpectedFragment(type="entity", text="climbing"),
        ],
        expected_active=[
            "search_writings",
            "find_recurring_themes",
            "trace_concept_evolution",
            "find_connected_concepts",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
        ],
    ),
    # ── State Query ─────────────────────────────────────────────────
    TestQuery(
        query="when was I most stuck",
        archetype="state_query",
        expected_primary_mass="stuck",
        expected_fragments=[
            ExpectedFragment(type="emotional", text="stuck", is_primary=True),
            ExpectedFragment(type="temporal", text="when"),
        ],
        expected_active=[
            "search_by_state",
            "search_writings",
            "temporal_filter",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "get_archetype_patterns",
        ],
    ),
    TestQuery(
        query="show me high-integration entries",
        archetype="state_query",
        expected_primary_mass="integration",
        expected_fragments=[
            ExpectedFragment(type="emotional", text="integration", is_primary=True),
            ExpectedFragment(type="emotional", text="high"),
        ],
        expected_active=[
            "search_by_state",
            "temporal_filter",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "find_entity_relationships",
        ],
    ),
    # ── Period Comparison ───────────────────────────────────────────
    TestQuery(
        query="compare January to August",
        archetype="period_comparison",
        expected_primary_mass="January",
        expected_fragments=[
            ExpectedFragment(type="temporal", text="January", is_primary=True),
            ExpectedFragment(type="temporal", text="August"),
            ExpectedFragment(type="relational", text="compare"),
        ],
        expected_active=[
            "compare_periods",
            "query_time_series",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "find_entity_relationships",
            "search_by_keyword",
        ],
    ),
    # ── Theme Deep Dive ─────────────────────────────────────────────
    TestQuery(
        query="explore my shadow around shame",
        archetype="theme_deep_dive",
        expected_primary_mass="shame",
        expected_fragments=[
            ExpectedFragment(type="emotional", text="shame", is_primary=True),
            ExpectedFragment(type="archetypal", text="shadow"),
        ],
        expected_active=[
            "search_writings",
            "search_themes",
            "find_connected_concepts",
            "find_recurring_themes",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "correlate_metrics",
        ],
    ),
    TestQuery(
        query="what themes come with fear",
        archetype="theme_deep_dive",
        expected_primary_mass="fear",
        expected_fragments=[
            ExpectedFragment(type="emotional", text="fear", is_primary=True),
            ExpectedFragment(type="relational", text="themes come with"),
        ],
        expected_active=[
            "search_themes",
            "find_connected_concepts",
            "search_writings",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "detect_anomalies",
        ],
    ),
    # ── Decision Review ─────────────────────────────────────────────
    TestQuery(
        query="what decisions did I make about work",
        archetype="decision_review",
        expected_primary_mass="decisions",
        expected_fragments=[
            ExpectedFragment(type="concept", text="decisions", is_primary=True),
            ExpectedFragment(type="entity", text="work"),
        ],
        expected_active=[
            "get_decision_context",
            "search_writings",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "detect_anomalies",
            "correlate_metrics",
        ],
    ),
    # ── Archetypal Query ────────────────────────────────────────────
    TestQuery(
        query="when does the Warrior show up",
        archetype="archetypal_query",
        expected_primary_mass="Warrior",
        expected_fragments=[
            ExpectedFragment(type="archetypal", text="Warrior", is_primary=True),
            ExpectedFragment(type="temporal", text="when"),
        ],
        expected_active=[
            "get_archetype_patterns",
            "search_writings",
            "search_themes",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "correlate_metrics",
        ],
    ),
    # ── Relational ──────────────────────────────────────────────────
    TestQuery(
        query="tension between discipline and freedom",
        archetype="relational_query",
        expected_primary_mass="discipline",
        expected_fragments=[
            ExpectedFragment(type="concept", text="discipline", is_primary=True),
            ExpectedFragment(type="concept", text="freedom"),
            ExpectedFragment(type="relational", text="tension between"),
        ],
        expected_active=[
            "search_writings",
            "find_connected_concepts",
            "get_concept_flows",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "detect_anomalies",
        ],
    ),
    TestQuery(
        query="who do I write about most",
        archetype="relational_query",
        expected_primary_mass="write about most",
        expected_fragments=[
            ExpectedFragment(type="relational", text="write about most", is_primary=True),
        ],
        expected_active=[
            "find_entity_relationships",
            "search_writings",
        ],
        expected_inactive=[
            "list_available_metrics",
            "detect_anomalies",
            "correlate_metrics",
        ],
    ),
    # ── Keyword / Exact ─────────────────────────────────────────────
    TestQuery(
        query="find entries mentioning StarSpace46",
        archetype="keyword_search",
        expected_primary_mass="StarSpace46",
        expected_fragments=[
            ExpectedFragment(type="entity", text="StarSpace46", is_primary=True),
        ],
        expected_active=[
            "search_by_keyword",
            "search_writings",
            "find_entity_relationships",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
            "detect_anomalies",
            "correlate_metrics",
        ],
    ),
    # ── Meta Queries ────────────────────────────────────────────────
    TestQuery(
        query="how much have I written",
        archetype="meta_query",
        expected_primary_mass="how much written",
        expected_fragments=[
            ExpectedFragment(type="concept", text="how much written", is_primary=True),
        ],
        expected_active=[
            "get_writing_stats",
        ],
        expected_inactive=[
            "find_entity_relationships",
            "get_archetype_patterns",
            "get_concept_flows",
            "search_themes",
        ],
    ),
    TestQuery(
        query="what metrics can I track",
        archetype="meta_query",
        expected_primary_mass="metrics",
        expected_fragments=[
            ExpectedFragment(type="concept", text="metrics", is_primary=True),
        ],
        expected_active=[
            "list_available_metrics",
        ],
        expected_inactive=[
            "find_entity_relationships",
            "get_archetype_patterns",
            "search_writings",
            "search_themes",
        ],
    ),
    # ── Complex Multi-Fragment ──────────────────────────────────────
    TestQuery(
        query="how has climbing affected my self-trust since I started at Blocworks",
        archetype="complex_multi_fragment",
        expected_primary_mass="self-trust",
        expected_fragments=[
            ExpectedFragment(type="emotional", text="self-trust", is_primary=True),
            ExpectedFragment(type="entity", text="climbing"),
            ExpectedFragment(type="relational", text="affected"),
            ExpectedFragment(type="temporal", text="since started"),
            ExpectedFragment(type="entity", text="Blocworks"),
        ],
        expected_active=[
            "search_writings",
            "find_recurring_themes",
            "query_time_series",
            "trace_concept_evolution",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
        ],
    ),
    TestQuery(
        query="what connects fear, agency, and the Creator archetype",
        archetype="complex_multi_fragment",
        expected_primary_mass="fear",
        expected_fragments=[
            ExpectedFragment(type="emotional", text="fear", is_primary=True),
            ExpectedFragment(type="emotional", text="agency"),
            ExpectedFragment(type="archetypal", text="Creator"),
            ExpectedFragment(type="relational", text="connects"),
        ],
        expected_active=[
            "find_connected_concepts",
            "search_writings",
            "get_archetype_patterns",
            "get_concept_flows",
        ],
        expected_inactive=[
            "get_writing_stats",
            "list_available_metrics",
        ],
    ),
]
