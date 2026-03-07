"""
20 test queries with expected activation patterns for validating the gravity field.

Each query specifies:
- query: raw natural language
- expected_primary_mass: the fragment text that should be primary mass
- expected_active: tools that SHOULD activate
- expected_inactive: tools that should NOT activate
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestQuery:
    query: str
    expected_primary_mass: str
    expected_active: list[str] = field(default_factory=list)
    expected_inactive: list[str] = field(default_factory=list)


TEST_QUERIES: list[TestQuery] = [
    # ── Person Exploration ──────────────────────────────────────────
    TestQuery(
        query="tell me about Kyle",
        expected_primary_mass="Kyle",
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
        expected_primary_mass="Matt",
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
        expected_primary_mass="silence",
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
        expected_primary_mass="sovereignty",
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
        expected_primary_mass="change over time",
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
        expected_primary_mass="silence",
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
        expected_primary_mass="stuck",
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
        expected_primary_mass="integration",
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
        expected_primary_mass="January",
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
        expected_primary_mass="shame",
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
        expected_primary_mass="fear",
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
        expected_primary_mass="decisions",
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
        expected_primary_mass="Warrior",
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
        expected_primary_mass="discipline",
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
        expected_primary_mass="write about most",
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
        expected_primary_mass="StarSpace46",
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
        expected_primary_mass="how much written",
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
        expected_primary_mass="metrics",
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
        expected_primary_mass="self-trust",
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
        expected_primary_mass="fear",
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
