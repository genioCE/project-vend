import { embedTexts } from "../embeddings-client.js";

export interface ToolIdentity {
  name: string;
  description: string;
  is_meta: boolean;
  always_active: boolean;
}

// 22 tool identity descriptions — verbatim from gravity/tool_identities.py
export const TOOLS: ToolIdentity[] = [
  // ── Search & Retrieval ──────────────────────────────────────────
  {
    name: "search_writings",
    description:
      "Find journal entries about any person, concept, feeling, experience, or topic. " +
      "The most general search tool — works for questions about people like Kyle or Matt, " +
      "abstract concepts like silence or sovereignty, emotions like shame or fear, " +
      "activities like climbing or work, and any open-ended exploration of the writing. " +
      "Returns the actual text passages most relevant to the query.",
    is_meta: false,
    always_active: true,
  },
  {
    name: "search_by_keyword",
    description:
      "Find exact words and phrases in the writing. Literal text matching for " +
      "specific names, places, terms, and quoted phrases. Use when the user " +
      "wants to locate a specific word or phrase they know appears in the corpus, " +
      "not semantic similarity but exact string match.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "get_entries_by_date",
    description:
      "Retrieve all writing from a specific date range. Full text of entries " +
      "within a time window, sorted chronologically. Use when the question asks " +
      "for what was written during a particular period — days, weeks, or months.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "get_recent_entries",
    description:
      "Get the most recent writing entries. What has been on the writer's mind " +
      "lately, newest first. Use for questions about current state, recent " +
      "thinking, or what's been happening recently in the writing.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "temporal_filter",
    description:
      "Find entries matching specific psychological metric thresholds within a " +
      "time range. Filter by measurable dimensions of inner state — entries where " +
      "agency is above 0.7, or valence is below -0.3, or word count exceeds 2000. " +
      "Quantitative filtering on numerical psychological dimensions.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "search_by_state",
    description:
      "Find entries by psychological dimension: valence, activation, agency, " +
      "certainty, relational openness, self-trust, time orientation, integration. " +
      "Filter by inner state to find entries when the writer felt stuck, empowered, " +
      "fragmented, uplifted, calm, or activated. Maps feelings to measured dimensions.",
    is_meta: false,
    always_active: false,
  },
  // ── Pattern & Evolution ─────────────────────────────────────────
  {
    name: "find_recurring_themes",
    description:
      "Track how a topic or theme evolves over time across the writing. " +
      "Chronological passages showing how thinking about a subject has changed " +
      "and grown. Use for questions about personal growth, recurring concerns, " +
      "evolving perspectives, what something means to me, or how understanding " +
      "of a concept has deepened. Answers 'what does X mean to me' by showing " +
      "the pattern of engagement over time.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "trace_concept_evolution",
    description:
      "Timeline of when and how a concept or person appears across the corpus. " +
      "Traces emergence, frequency, and contextual shifts over time. Use for: " +
      "how does X show up in my writing, how has my relationship with X changed, " +
      "how has X evolved, what does X mean to me over time, how X manifests or " +
      "appears across entries. Maps the journey of any idea or person through the writing.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "get_concept_flows",
    description:
      "Directed transitions between concepts. Movement patterns like fear to " +
      "action, doubt to clarity, isolation to connection. What flows into and " +
      "out of an idea. Use for questions about mindset shifts, psychological " +
      "movement patterns, and how one state leads to another.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "search_themes",
    description:
      "Find psychological and narrative theme patterns and their co-occurrences. " +
      "Themes are 2-4 word patterns like 'cultivating resilience', 'seeking clarity', " +
      "'struggling with discipline'. Returns entries linked to a theme and which " +
      "other themes appear alongside it. Also covers shadow work, archetypal themes, " +
      "psychological patterns, and what themes accompany a feeling or archetype.",
    is_meta: false,
    always_active: false,
  },
  // ── Graph & Relationship ────────────────────────────────────────
  {
    name: "find_connected_concepts",
    description:
      "Explore what ideas, people, emotions, and entries are connected to a concept, " +
      "person, or theme. Maps the relational neighborhood: what clusters around silence, " +
      "who connects to Kyle, what emotions associate with sovereignty, what concepts " +
      "relate to fear or shame. Essential for understanding meaning, context, and " +
      "the web of associations around any subject in the writing.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "find_entity_relationships",
    description:
      "Map a person's presence across the writing. All entries mentioning them, " +
      "with associated concepts and emotions. Relational mapping. Use for questions " +
      "about specific people — how they show up, what themes surround them, " +
      "what the relationship looks like across time.",
    is_meta: false,
    always_active: false,
  },
  // ── Psychological & Archetypal ──────────────────────────────────
  {
    name: "get_entry_analysis",
    description:
      "Deep analysis of a single entry: summary, themes, entities, decisions, " +
      "and full 8-dimension psychological state profile. Use when a specific entry " +
      "has been identified and you need the complete analytical breakdown of that " +
      "particular day's writing.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "get_archetype_patterns",
    description:
      "Archetypal patterns found in the writing: Creator, Warrior, Healer, Sage, " +
      "Lover, Sovereign, Integrator. Frequency, strength, and associated emotions. " +
      "Use for questions about mythic patterns, roles the writer embodies, " +
      "archetypal energy, and which archetypes are most active.",
    is_meta: false,
    always_active: false,
  },
  // ── Quantitative & Temporal ─────────────────────────────────────
  {
    name: "query_time_series",
    description:
      "Any metric plotted over time: psychological dimensions, word count, " +
      "archetype frequency, concept frequency, theme frequency. Time series data " +
      "at entry, daily, or weekly granularity. Use for trend questions, tracking " +
      "how a measurable quantity changes across dates.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "detect_anomalies",
    description:
      "Find outlier entries where a metric deviates significantly from baseline. " +
      "Statistical anomaly detection using z-scores. Use for questions about " +
      "unusual days, extreme states, sudden shifts, or finding when something " +
      "was notably different from the norm.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "correlate_metrics",
    description:
      "Discover statistical relationships between two metrics over time. Pearson " +
      "correlation coefficient with p-value and interpretation. Use for questions " +
      "about whether two dimensions move together — does agency correlate with " +
      "valence? Does word count predict integration?",
    is_meta: false,
    always_active: false,
  },
  {
    name: "get_metric_summary",
    description:
      "Summary statistics for any metric: mean, median, standard deviation, min, " +
      "max, entry count, and current trend direction. Quick quantitative overview " +
      "of a dimension's behavior over a period. Use for baseline questions and " +
      "understanding typical ranges.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "compare_periods",
    description:
      "Side-by-side comparison of two time windows: concepts, emotions, archetypes. " +
      "Use for questions about how different periods compare — was January different " +
      "from March? How did summer compare to fall? What changed between two eras?",
    is_meta: false,
    always_active: false,
  },
  // ── Meta & Context ──────────────────────────────────────────────
  {
    name: "get_writing_stats",
    description:
      "Corpus-level overview: total word count, date range, number of entries, " +
      "average length, entries per year. Use for meta-questions about the writing " +
      "practice itself — how much have I written, how many entries, how many words, " +
      "writing volume, word count totals, how consistent the habit is, practice statistics.",
    is_meta: true,
    always_active: false,
  },
  {
    name: "list_available_metrics",
    description:
      "Discover all queryable metric names and types. Returns dimension metrics, " +
      "entry statistics, archetype metrics, theme metrics, and concept metrics " +
      "available in the system. Use when needing to know what can be measured " +
      "or tracked — a catalog of available data dimensions.",
    is_meta: true,
    always_active: false,
  },
  {
    name: "get_decision_context",
    description:
      "Recorded decisions and their surrounding emotional and conceptual state. " +
      "What was decided, when, and what the inner landscape looked like at the time. " +
      "Use for questions about choices made, decision patterns, and the emotional " +
      "context around commitments and turning points.",
    is_meta: false,
    always_active: false,
  },
];

export const TOOL_NAMES = TOOLS.map((t) => t.name);
export const META_TOOLS = new Set(TOOLS.filter((t) => t.is_meta).map((t) => t.name));
export const ALWAYS_ACTIVE_TOOLS = new Set(
  TOOLS.filter((t) => t.always_active).map((t) => t.name)
);

// In-memory cache for identity vectors (embedded on first use)
let _identityVectors: number[][] | null = null;

export async function getIdentityVectors(
  signal?: AbortSignal
): Promise<number[][]> {
  if (_identityVectors) return _identityVectors;
  const descriptions = TOOLS.map((t) => t.description);
  _identityVectors = await embedTexts(descriptions, signal);
  return _identityVectors;
}
