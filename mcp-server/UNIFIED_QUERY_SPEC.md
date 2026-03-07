# Unified Query Layer — Design Spec & Implementation Guide

## What This Is

A new orchestration layer inside the existing `mcp-server` that exposes a single MCP tool (`deep_query`) which decomposes a natural language question, fans out to all four storage engines in parallel, and returns one assembled result.

Right now, Claude (or the web UI agent) has to manually pick which tools to call and stitch results together across turns. The unified query layer moves that orchestration into the server itself.

## The Problem It Solves

A question like "how has my relationship with Kyle changed since summer" currently requires Claude to make 4-8 separate tool calls:
- `find_entity_relationships` (Neo4j) for Kyle's presence across entries
- `search_writings` (ChromaDB) for semantic passages about Kyle
- `query_time_series` (DuckDB) for psychological dimensions over the date range
- `get_entry_analysis` (SQLite) for state profiles of relevant entries
- Maybe `trace_concept_evolution`, `compare_periods`, etc.

Each call is a separate round-trip through the MCP protocol. The unified layer collapses this into one call with one response.

## Architecture

### Where It Lives

```
mcp-server/src/
├── unified/
│   ├── classifier.ts      — Question decomposition (what engines does this need?)
│   ├── executor.ts         — Parallel fan-out to engines
│   ├── assembler.ts        — Result merging and deduplication
│   ├── types.ts            — Shared types for the unified layer
│   └── tools.ts            — MCP tool registration (deep_query + possibly deep_summary)
├── embeddings-client.ts    — (existing, no changes)
├── graph-client.ts         — (existing, no changes)
├── analysis-client.ts      — (existing, no changes)
├── timeseries/             — (existing, no changes)
├── tools.ts                — (existing, no changes — all 23 individual tools stay)
└── index.ts                — Add: registerUnifiedTools(server)
```

The unified layer is additive. It imports the existing client functions and composes them. No existing tools change or break.

### Registration

In `index.ts`, add:
```typescript
import { registerUnifiedTools } from "./unified/tools.js";
// ... existing registrations ...
registerUnifiedTools(server);
```

---

## The Four Engines (What Each Knows)

Understanding what each engine is good at is the core of the classifier's job.

### 1. ChromaDB (embeddings-service, port 8000)
- **Good at:** Finding passages *semantically similar* to a question. Raw text retrieval. Keyword matches.
- **Client functions:** `searchWritings`, `getEntriesByDate`, `findRecurringThemes`, `getRecentEntries`, `searchByKeyword`
- **Returns:** Text passages with date, source_file, relevance_score, word_count

### 2. Neo4j (graph-service, port 8001)
- **Good at:** Relationship traversal. Who co-occurs with whom? What concepts connect? How do themes cluster? Entity presence over time. Period comparison. Archetype patterns.
- **Client functions:** `findConnectedConcepts`, `findEntityRelationships`, `traceConceptEvolution`, `comparePeriods`, `getDecisionContext`, `getArchetypePatterns`, `getConceptFlows`, `getThemeNetwork`, `getEntriesByState`
- **Returns:** Structured graph data (nodes, relationships, dates, co-occurrences)

### 3. SQLite (analysis-service, port 8002)
- **Good at:** Per-entry pre-computed analysis: summaries, themes, typed entities, decisions, 8-dimension psychological state profiles.
- **Client functions:** `getEntrySummary`, `getEntrySummaries`
- **Returns:** Structured analysis records keyed by entry_id

### 4. DuckDB (timeseries, in-process)
- **Good at:** Quantitative trends over time. Aggregations (daily/weekly). Anomaly detection. Correlations between metrics. Filtering entries by metric thresholds.
- **Functions:** Available via the timeseries module's internal query function, or through the higher-level tool handlers
- **Returns:** Rows of (date, value) pairs, statistical summaries, correlation coefficients

---

## Classifier Design (`classifier.ts`)

The classifier takes a natural language question and returns a query plan. This should be **deterministic and rule-based** — not an LLM call. We don't want to add latency or cost to the orchestration layer. An LLM call to decompose a question would take 500ms-1s and defeat the purpose of parallel execution.

### Query Plan Type

```typescript
interface QueryPlan {
  // Which engines to query
  needs_vector: boolean;       // ChromaDB semantic search
  needs_graph: boolean;        // Neo4j relationship/concept queries
  needs_timeseries: boolean;   // DuckDB quantitative trends
  needs_analysis: boolean;     // SQLite entry enrichment (almost always true)

  // Extracted parameters
  date_range?: { start: string; end: string };
  entities?: string[];         // People, places, orgs mentioned in question
  concepts?: string[];         // Abstract concepts/themes mentioned
  metrics?: string[];          // Psychological dimensions or word_count
  archetypes?: string[];       // Archetype names if mentioned

  // Query intent (helps assembler prioritize)
  intent: QueryIntent;

  // The search query to use for vector/graph (may be the original or refined)
  search_query: string;
}

type QueryIntent =
  | "person_exploration"      // "tell me about Kyle" / "how do I relate to Ruth"
  | "concept_exploration"     // "what does sovereignty mean to me"
  | "temporal_arc"            // "how have I changed since summer"
  | "state_query"             // "when was I most stuck" / "my lowest points"
  | "period_comparison"       // "compare Q1 to Q3"
  | "theme_deep_dive"         // "explore my relationship with fear"
  | "decision_review"         // "what decisions did I make about work"
  | "general"                 // catch-all
```

### Classification Rules

The classifier should use keyword/pattern matching to determine the plan. Some heuristics:

**Entities:** Scan for capitalized words that aren't at sentence starts. Known names from a small built-in list would help (Kyle, Matt, Brian, Ruth, Kelsey, Mom, Tony, etc.) but even without one, capitalized nouns are a strong signal. If entities are found → `needs_graph = true`.

**Temporal signals:** Words like "changed", "evolved", "over time", "since", "between", "arc", "trajectory", "trend", dates, month names, season names → `needs_timeseries = true`. If two time references are found → intent may be `period_comparison`.

**State/dimension signals:** Words like "stuck", "empowered", "fragmented", "integrated", "trust", "doubt", "heavy", "uplifted", "anxious", "calm", "agency", "valence" → `needs_timeseries = true` with specific metrics extracted.

**Concept/theme signals:** Abstract nouns like "sovereignty", "fear", "recovery", "silence", "addiction", "gratitude" → `needs_graph = true` (concept exploration).

**Vector search:** Almost always true unless the question is purely quantitative ("what was my average agency in March"). The semantic search provides the textual grounding that makes everything else meaningful.

**Analysis enrichment:** Almost always true. Once we have entry IDs from vector or graph results, enriching with state profiles and summaries is cheap and adds a lot.

### Date Range Extraction

Parse for:
- Explicit dates: "since July 2025", "between January and March"
- Relative dates: "last 3 months", "this year", "since summer"
- Season mapping: summer → June-August, fall → September-November, etc.
- If no date range detected, default to full corpus range

### Implementation Notes

Keep the classifier simple. It doesn't need to be perfect — it needs to be fast and err on the side of querying too many engines rather than too few. A query that hits all four engines and ignores irrelevant results is better than one that misses a relevant engine.

```typescript
export function classify(question: string): QueryPlan {
  // Normalize
  const q = question.toLowerCase();
  
  // Extract entities (capitalized words from original, known names, etc.)
  // Extract date ranges
  // Check for temporal/trend language
  // Check for state/dimension language
  // Check for relationship/person language
  // Check for concept/theme language
  // Determine intent
  
  // Default: needs_vector = true, needs_analysis = true
  // Add graph/timeseries based on signals
}
```

---

## Executor Design (`executor.ts`)

Two-phase parallel execution.

### Phase 1: Fan-out (parallel)

Fire all non-dependent queries simultaneously:

```typescript
const phase1 = await Promise.allSettled([
  // Vector: semantic search for the question
  plan.needs_vector
    ? searchWritings(plan.search_query, 10)
    : null,

  // Graph: depends on intent
  plan.needs_graph
    ? executeGraphQueries(plan)
    : null,

  // Timeseries: quantitative data
  plan.needs_timeseries
    ? executeTimeseriesQueries(plan)
    : null,
]);
```

**Graph query selection** (`executeGraphQueries`): Based on intent and extracted parameters, pick the right graph calls:
- `person_exploration` → `findEntityRelationships(name)` + `traceConceptEvolution(name)`
- `concept_exploration` → `findConnectedConcepts(name)` + `getConceptFlows(name)`
- `temporal_arc` → `comparePeriods(...)` if two periods, else `traceConceptEvolution`
- `theme_deep_dive` → `getThemeNetwork(name)` + `findConnectedConcepts(name)`
- `decision_review` → `getDecisionContext(keyword)`
- `state_query` → `getEntriesByState(dimension, min, max)`

**Timeseries query selection** (`executeTimeseriesQueries`): Based on extracted metrics and date range:
- If specific dimensions mentioned → `query_time_series` for each
- If "anomaly" or "outlier" language → `detect_anomalies`
- If "correlation" or "relationship between" → `correlate_metrics`
- Default for temporal_arc → query valence, agency, self_trust, integration over the date range

### Phase 2: Analysis enrichment (after phase 1)

Collect all unique entry IDs from phase 1 results (vector hits, graph entry references), then batch-fetch analysis:

```typescript
const allEntryIds = collectEntryIds(phase1Results);
const uniqueIds = [...new Set(allEntryIds)];
const analyses = await getEntrySummaries(uniqueIds);
```

This is the only sequential dependency. Everything else is parallel.

### Error Handling

Use `Promise.allSettled`, not `Promise.all`. If Neo4j is down, we still return ChromaDB + DuckDB results. Each engine's failure is noted in the response metadata but doesn't block the others.

```typescript
interface EngineResult {
  engine: "vector" | "graph" | "timeseries" | "analysis";
  status: "ok" | "error" | "skipped";
  data: unknown;
  duration_ms: number;
  error?: string;
}
```

### Timeouts

Each engine call should have a per-engine timeout (e.g., 8 seconds). If a single engine is slow, don't let it block the whole response. Use `AbortController` per engine.

---

## Assembler Design (`assembler.ts`)

The assembler takes raw results from all four engines and produces a unified context packet. This is the hardest part.

### Unified Result Type

```typescript
interface UnifiedResult {
  // The original question
  question: string;
  intent: QueryIntent;
  
  // Assembled evidence grouped by entry
  entries: UnifiedEntry[];
  
  // Quantitative trends (if timeseries was queried)
  trends?: TrendSummary[];
  
  // Graph-level insights (relationships, flows, patterns not tied to specific entries)
  graph_insights?: GraphInsight[];
  
  // Metadata about what engines contributed
  engines: EngineResult[];
  
  // Total execution time
  total_ms: number;
}

interface UnifiedEntry {
  entry_id: string;
  date: string;
  
  // From ChromaDB (if this entry appeared in vector results)
  passages?: string[];
  relevance_score?: number;
  
  // From SQLite analysis
  summary?: string;
  themes?: string[];
  state_profile?: Record<string, number>;  // dimension -> score
  
  // From Neo4j (if this entry appeared in graph results)
  connected_entities?: string[];
  connected_concepts?: string[];
  emotions?: string[];
  archetypes?: Array<{ name: string; strength: number }>;
}

interface TrendSummary {
  metric: string;
  date_range: { start: string; end: string };
  direction: "rising" | "falling" | "stable";
  mean: number;
  current: number;  // most recent value
  // Optional: key inflection points
  inflections?: Array<{ date: string; value: number; note: string }>;
}

interface GraphInsight {
  type: "concept_network" | "entity_map" | "theme_cluster" | "flow" | "archetype_pattern";
  description: string;  // human-readable summary
  data: unknown;        // raw graph data for Claude to interpret
}
```

### Assembly Logic

1. **Collect all entry IDs** from vector results and graph results
2. **Deduplicate** — same entry from multiple engines gets merged, not repeated
3. **Sort by date** — chronological ordering
4. **Attach analysis** — for each entry, attach its SQLite summary/state profile if available
5. **Attach graph context** — for each entry that appeared in graph results, attach connected entities/concepts/emotions
6. **Attach passages** — for each entry that appeared in vector results, attach the relevant text snippets
7. **Build trend summaries** — condense timeseries data into directional summaries rather than raw rows
8. **Extract graph insights** — concept networks, flows, archetype patterns that aren't tied to individual entries

### Deduplication

The key dedup logic: entries are identified by `entry_id` (filename stem like `7-15-2025`). When the same entry appears in both vector and graph results, merge the data rather than creating two entries. Vector provides the text passage. Graph provides the entity/concept context. Analysis provides the state profile. All three perspectives on the same entry belong together.

### Size Management

The unified result could get large. Cap it:
- Max 15-20 entries in the response (take the most relevant by combining relevance_score and recency)
- Truncate passages to ~500 chars each
- Limit trend data to weekly granularity
- Limit graph insights to top 10 connections per type

---

## MCP Tool Registration (`tools.ts`)

### Tool: `deep_query`

```typescript
server.tool(
  "deep_query",
  "Unified query across all four storage engines (ChromaDB, Neo4j, SQLite, DuckDB). " +
  "Decomposes a natural language question, queries relevant engines in parallel, " +
  "and returns assembled evidence with text passages, psychological state profiles, " +
  "graph relationships, and quantitative trends — all deduplicated and grouped by entry. " +
  "Use this for complex questions that span multiple dimensions of the corpus. " +
  "For simple lookups, the individual tools may be faster.",
  {
    question: z.string().describe(
      "Natural language question about the writing corpus"
    ),
    date_range: z.object({
      start: z.string(),
      end: z.string(),
    }).optional().describe(
      "Optional explicit date range (YYYY-MM-DD). If not provided, the classifier will try to extract dates from the question or default to full corpus."
    ),
    max_entries: z.number().int().min(1).max(30).default(15).describe(
      "Maximum number of entries to include in the result (default 15)"
    ),
    include_passages: z.boolean().default(true).describe(
      "Include raw text passages from vector search (default true)"
    ),
    include_trends: z.boolean().default(true).describe(
      "Include quantitative trend data from DuckDB (default true)"
    ),
  },
  async ({ question, date_range, max_entries, include_passages, include_trends }) => {
    // 1. Classify
    const plan = classify(question);
    if (date_range) {
      plan.date_range = date_range;
    }
    
    // 2. Execute (parallel fan-out + sequential enrichment)
    const engineResults = await execute(plan, { include_passages, include_trends });
    
    // 3. Assemble
    const result = assemble(plan, engineResults, { max_entries });
    
    // 4. Format for MCP response
    return {
      content: [{
        type: "text" as const,
        text: formatUnifiedResult(result),
      }],
    };
  }
);
```

### Output Formatting

The `formatUnifiedResult` function should produce a structured but readable text output. Something like:

```
## Deep Query: "how has my relationship with Kyle changed since summer"
Intent: person_exploration | Engines: vector ✓ graph ✓ timeseries ✓ analysis ✓ | 1,247ms

### Trends (July–December 2025)
- **Agency:** 0.73 → 0.69 (stable)
- **Relational Openness:** 0.61 → 0.55 (falling)

### Graph Insights
- Kyle co-occurs with: Matt, Justice, Hix, Morgan, work family, shop, leadership
- Connected themes: "creative partnership", "boundary setting", "community stewardship"

### Entries (12 matched, showing top 8)

**2025-07-14** | agency: 0.77 | valence: 0.67
Summary: Reflecting on Kyle's leadership at the shop and Ford's energy...
Passage: "Kyle's got this way of just being present without needing to..."
Entities: Kyle, Ford, Justice, Hix
Themes: creative partnership, witnessing others

**2025-08-23** | agency: 0.81 | valence: 0.73
Summary: Big climbing day followed by time at the shop...
...
```

---

## Implementation Order

1. **`types.ts`** — Define all the types (QueryPlan, UnifiedResult, EngineResult, etc.)
2. **`classifier.ts`** — Build the rule-based classifier. Write tests against known question patterns.
3. **`executor.ts`** — Wire up the parallel fan-out using existing client functions. Handle timeouts and errors.
4. **`assembler.ts`** — Merge results, deduplicate by entry_id, attach analysis, build trend summaries.
5. **`tools.ts`** — Register the `deep_query` MCP tool.
6. **`index.ts`** — Add `registerUnifiedTools(server)` call.
7. **Test manually** — Ask questions through Claude Desktop and verify the unified results.

## Testing

Test the classifier against these question types:
- "tell me about Kyle" → person_exploration, needs_graph, needs_vector
- "how has my agency changed this year" → temporal_arc, needs_timeseries, needs_vector
- "compare January to August" → period_comparison, needs_graph, needs_timeseries
- "what does silence mean to me" → concept_exploration, needs_graph, needs_vector
- "when was I most fragmented" → state_query, needs_timeseries, needs_graph
- "what decisions did I make about work in Q3" → decision_review, needs_graph, date_range extracted
- "explore my shadow around shame" → theme_deep_dive, needs_graph, needs_vector
- "how did climbing change me" → temporal_arc + concept_exploration, needs all four

## What NOT to Do

- **Don't add an LLM call to the classifier.** The whole point is to be fast. Rule-based classification at the orchestration layer, LLM interpretation at the Claude conversation layer.
- **Don't modify existing tools or clients.** The unified layer is purely additive — it composes existing functions.
- **Don't try to generate a narrative in the assembler.** The assembler returns structured data. Claude (the conversation model) turns that into narrative. The assembler's job is to make Claude's job easier by deduplicating and organizing.
- **Don't block on any single engine.** `Promise.allSettled` with timeouts. Always return what you got.

## Dependencies

No new npm packages needed. Everything uses:
- Existing client functions from `embeddings-client.ts`, `graph-client.ts`, `analysis-client.ts`
- Existing timeseries query functions from `timeseries/db.ts` and `timeseries/metrics.ts`
- `zod` for tool schema (already installed)
- Built-in `AbortController` for timeouts

## Docker / Infrastructure

No changes. The unified layer runs inside the existing `mcp-server` container. It calls the same internal service URLs. No new ports, no new volumes, no new services.
