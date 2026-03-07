# Corpus Intelligence — System Summary

## What It Is

A private, multi-engine knowledge system for exploring a personal writing corpus (~1M words of daily journal entries). It combines semantic vector search, a knowledge graph, LLM-backed psychological state profiling, a quantitative time series engine, and automatic multi-tool orchestration via semantic gravity fields — all accessible through 22 MCP tools + 1 gravity orchestrator that plug directly into Claude.

The system turns unstructured writing (journal entries in markdown) into a structured, queryable knowledge base that Claude can navigate on the user's behalf during conversation. The **Gravity Model** orchestrates tool selection automatically — queries are decomposed into typed semantic fragments, embedded alongside tool identity vectors, and a gravitational pull matrix determines which tools to activate. Claude calls one tool (`orchestrated_query`) instead of manually selecting from 22 individual tools.

---

## Architecture

Five core services + in-process DuckDB time series, all containerized via Docker Compose on an isolated network (`corpus-net`) with health checks and named volumes for persistence.

### Services

| Service | Stack | Port | Purpose |
|---------|-------|------|---------|
| **embeddings-service** | Python/FastAPI, ChromaDB, Sentence-Transformers | 8000 | Semantic vector search — chunks entries, embeds with `all-mpnet-base-v2`, stores in ChromaDB. Also provides `/embed` endpoint for gravity vectors |
| **graph-service** | Python/FastAPI, Neo4j, spaCy | 8001 | Knowledge graph construction and GraphRAG queries |
| **analysis-service** | Python/FastAPI, SQLite, Anthropic SDK | 8002 | LLM-backed entry summarization, typed entity/theme extraction, and 8-dimension psychological state profiling |
| **mcp-server** | TypeScript/Node.js | 3001 | MCP stdio server with 22 tools + gravity orchestrator. DuckDB time series in-process |
| **web-ui** | React | 3000 | Dark-themed chat interface with GraphRAG toggle and feedback buttons |

### Supporting Infrastructure

- **Neo4j 5 Community** — graph database (port 7687)
- **DuckDB** — in-process time series database for trend analysis, anomaly detection, metric correlation
- **Anthropic API** — Claude Haiku 4.5 for analysis batch processing and query decomposition (gravity model)
- **Ollama** (optional) — local LLM (llama3.1:8b) fallback; runs on Linux server (4070 Super) or macOS (Metal)

---

## Knowledge Graph Schema

### Node Types (10)

| Node | Properties | What It Represents |
|------|------------|--------------------|
| **Entry** | date, filename, word_count, title, short_summary, state_valence/activation/agency/certainty/relational_openness/self_trust/time_orientation/integration, state_enriched | A single journal entry |
| **Person** | name, normalized_name | Someone mentioned in writing |
| **Place** | name, normalized_name | A geographic location |
| **Organization** | name, normalized_name | A company, group, or institution (e.g., InterWorks, AA) |
| **Spiritual** | name, normalized_name | A spiritual entity (God, Higher Power) |
| **Concept** | name, normalized_name | An idea or topic (noun chunks from spaCy + LLM-refined concepts) |
| **Theme** | name, normalized_name | A 2-4 word psychological/narrative pattern (e.g., "cultivating resilience", "seeking clarity") |
| **Emotion** | name | An emotional state |
| **Archetype** | name | A narrative/Jungian archetype (Warrior, Sage, Creator, Healer, etc.) |
| **Decision** | text | A decision or action recorded in writing |

### Relationship Types (9)

| Relationship | Direction | Properties | Meaning |
|--------------|-----------|------------|---------|
| **MENTIONS** | Entry → Person/Place/Organization/Spiritual | entry_valence, entry_activation, entry_agency | Entry references an entity; state scores from the entry's psychological profile |
| **CONTAINS** | Entry → Concept | — | Entry discusses a concept |
| **HAS_THEME** | Entry → Theme | — | Entry is linked to a psychological/narrative theme |
| **EXPRESSES** | Entry → Emotion | intensity (0–1) | Entry carries an emotion at some intensity |
| **RECORDS** | Entry → Decision | — | Entry captures a decision |
| **INVOKES** | Entry → Archetype | strength (0–1) | Entry activates an archetypal pattern |
| **COOCCURS_WITH** | Concept ↔ Concept | weight | Two concepts appear together across entries |
| **THEME_COOCCURS** | Theme ↔ Theme | weight | Two themes appear together in the same entry |
| **FLOWS_TO** | Concept → Concept | weight, first_seen, last_seen, sample_phrase | A directed transition pattern (e.g., "fear → action") |

All relationships use MERGE for idempotent ingestion.

---

## Psychological State Profiling

Each entry gets an 8-dimension state profile, scored from -1.0 to +1.0:

| Dimension | Low Anchor (-1) | High Anchor (+1) |
|-----------|-----------------|-------------------|
| **Valence** | Heavy | Uplifted |
| **Activation** | Calm | Activated |
| **Agency** | Stuck | Empowered |
| **Certainty** | Conflicted | Resolved |
| **Relational Openness** | Guarded | Open |
| **Self-Trust** | Doubt | Trust |
| **Time Orientation** | Past-looping | Future-building |
| **Integration** | Fragmented | Coherent |

Signals are detected via 6 categories: lexical (keyword matches), pattern (multi-word phrases), modal (ability verbs), temporal (time words), relational (social words), and structural (coherence words).

### Analysis Pipeline

Each entry receives:
- **short_summary**: 1-2 sentence emotional/thematic core
- **detailed_summary**: 3-5 sentence arc covering events, emotions, transitions
- **themes**: 3-8 abstract psychological/behavioral themes, tightened to 2-4 words (e.g., "cultivating resilience", "seeking clarity")
- **entities**: Typed objects with name and type (person, place, organization, concept, spiritual)
- **decisions_actions**: Explicit decisions, commitments, realizations

**Dual-provider architecture:**
- **Anthropic Claude** (preferred): Single API call per entry, handles full context natively. No chunk-and-merge needed. Produces higher-quality typed entities, more nuanced themes, and more accurate state scores.
- **Ollama** (fallback): Chunk-and-merge for entries >1500 words (llama3.1:8b loses JSON compliance on long inputs). Each ~500-word chunk is processed individually, then synthesized.
- Provider auto-resolves based on `ANTHROPIC_API_KEY` availability. Can be overridden with `--provider` flag.

---

## Gravity Orchestration System

### How It Works

The Gravity Model is a semantic activation framework that replaces manual tool selection:

1. **Decompose** — Claude Haiku breaks the query into typed semantic fragments (concept, entity, temporal, emotional, relational, archetypal) and extracts structured parameters (entities, concepts, date ranges, metrics)
2. **Embed** — Each fragment + the full query are embedded via the embeddings service (`/embed` endpoint, same `all-mpnet-base-v2` model as the corpus)
3. **Gravity Field** — Cosine similarity between each tool's identity vector and each query vector produces a pull matrix (22 tools x N+1 vectors). L2 norm across pulls gives composite activation scores
4. **Gap Detection** — Sorted composite scores are analyzed for the largest relative gap (natural elbow), determining the activation cutoff (min 3, max 10 tools)
5. **Dispatch** — All activated tools execute in parallel with 10s per-tool timeouts via `Promise.allSettled`
6. **Assemble** — Results sorted by composite score (highest gravity = most relevant) for Claude's synthesis

### Key Math

```
pull[tool_i, vector_j] = dot(identity_i, vector_j)    // cosine similarity (L2-normalized)
composite[tool_i] = sqrt(Σ pull[i,j]²)                 // L2 norm — multi-fragment pulls compound
cutoff = adaptive gap detection in sorted composites    // natural elbow, bounded [3, 10] tools
```

### Activation Rules

- **Always-active**: `search_writings` fires unconditionally (textual grounding)
- **Gravity-gated**: Most tools — fire if composite score >= adaptive cutoff
- **Meta-gated**: `get_writing_stats`, `list_available_metrics` — only if in top 3 overall
- **Semantic boost**: If metrics extracted, force-include `query_time_series` + `get_metric_summary`
- **Dispatch guards**: `correlate_metrics` needs 2+ metrics, `compare_periods` needs 2+ date ranges

### Gravity Mode (default)

When `GRAVITY_MODE` is on (default), Claude only sees 4 tools:

| Tool | Purpose |
|------|---------|
| `orchestrated_query` | Multi-tool gravity dispatch |
| `get_entry_analysis` | Follow-up deep dive on specific entries |
| `get_entries_by_date` | Follow-up date-range retrieval |
| `get_recent_entries` | Follow-up recent entries |

### Gravity Files

| File | Purpose |
|------|---------|
| `mcp-server/src/gravity/types.ts` | Core types (Fragment, GravityField, ActivatedTool, OrchestratedResult) |
| `mcp-server/src/gravity/decompose.ts` | Claude Haiku query decomposition |
| `mcp-server/src/gravity/identities.ts` | 22 tool identity descriptions + vector caching |
| `mcp-server/src/gravity/field.ts` | Gravity math (pull matrix, L2 norm, gap detection, reliability bias) |
| `mcp-server/src/gravity/dispatch.ts` | Parallel tool dispatch with canDispatch guards |
| `mcp-server/src/gravity/assemble.ts` | Result assembly |
| `mcp-server/src/gravity/orchestrate.ts` | Main pipeline entry point |
| `mcp-server/src/gravity/ledger.ts` | Gravity learning loop — outcome recording and reliability aggregation |
| `mcp-server/src/gravity/tools.ts` | MCP tool registration |

### Gravity Learning Loop

The gravity system learns from query outcomes to improve tool selection over time. After every orchestrated query, outcomes are recorded and aggregated into per-tool reliability profiles.

**Outcome Recording**

Each orchestration logs a `GravityOutcome` to `data/gravity-ledger.jsonl`:
- Query text and decomposed fragments
- List of activated tools
- Per-tool outcome: composite score, duration, errored flag, empty flag, result size

**Emptiness Detection**

A tool result is considered "empty" (non-useful) if:
- It's an explicit error object `{"error": ...}`
- It's an empty array `[]`
- It's an object where all array values are empty
- The result string is less than 50 bytes after removing whitespace

**Reliability Aggregation**

Tool reliability profiles are computed with **exponential decay** (half-life: 50 queries) so recent outcomes matter more than old ones:

```
weight = exp(-0.0139 * queriesAgo)
```

Each tool gets a `ToolReliability` profile:
- `useful_rate`: proportion of activations that returned useful (non-error, non-empty) results
- `error_rate`, `empty_rate`: failure modes
- `avg_result_size`: average size of useful results
- `by_fragment_type`: useful_rate broken down by which fragment types were in the query

**Reliability Bias**

Before gap detection, composite scores are multiplied by a learned reliability bias:

```
biased_score[i] = composite_score[i] * bias[i]
bias[i] = max(floor, reliability_score[i])
```

- **floor** (default 0.3): Minimum bias so no tool is fully suppressed
- **minActivations** (default 10): Tools with fewer activations than this use raw scores (not enough data)
- **Fragment-conditioned**: When the tool has enough per-fragment-type data, uses a weighted average of useful_rate for the fragment types in the current query

**Effect**: Tools that consistently return useful results get boosted; tools that frequently error or return empty results are dampened. The learning is incremental and query-local (no global retraining needed).

### 3D Gravity Visualization

A diagnostic visualization renders the gravity field as a 3D orbital system using Three.js:

```bash
cd mcp-server && source ../.env && npx tsx gravity-viz.ts
# Open http://localhost:4000
```

- Fragment nodes on a ring, tools orbiting around their attracting fragments in 3D
- Orbit radius inversely proportional to composite score
- Tilted orbital planes per tool (golden angle spacing) for visual depth
- Pull lines connecting activated tools to fragments
- Mouse drag to rotate, scroll to zoom, auto-rotation
- Sidebar shows activation scores, fragments, extracted params, and pipeline timing

---

## MCP Tools (22 + 1 orchestrator)

### Orchestrator (1 tool)

1. **orchestrated_query** — Automatic multi-tool dispatch via semantic gravity field

### Semantic Search (6 tools)

2. **search_writings** — Find passages most similar in meaning to a query
3. **get_entries_by_date** — Retrieve full entries within a date range
4. **find_recurring_themes** — Trace how a topic evolves chronologically
5. **get_writing_stats** — Corpus statistics (word count, entries, date range, entries per year)
6. **get_recent_entries** — Last N entries, newest first
7. **search_by_keyword** — Exact text search with surrounding context

### Analysis (1 tool)

8. **get_entry_analysis** — Per-entry summary: themes, entities, decisions, and full 8-dimension state profile (Claude-backed)

### Knowledge Graph (10 tools)

9. **find_connected_concepts** — Concepts, people, emotions connected to a given concept
10. **trace_concept_evolution** — How a concept appears over time with co-occurring entities
11. **find_entity_relationships** — Map a person's presence across the corpus
12. **compare_periods** — Compare concepts, emotions, archetypes, themes, and average state between two time periods
13. **get_decision_context** — Decisions with emotional and conceptual context
14. **get_archetype_patterns** — Archetypal patterns, frequency, strength, associated emotions
15. **get_concept_flows** — Directed transitions (X → Y) for a concept
16. **search_themes** — Find entries linked to a theme with co-occurring themes and emotional context
17. **search_by_state** — Filter entries by psychological state dimension (e.g., high agency, low valence)
18. **temporal_filter** — Filter entries by metric thresholds in a date range

### Quantitative & Time Series (5 tools)

19. **query_time_series** — Plot any metric over time at entry/daily/weekly granularity
20. **detect_anomalies** — Flag outlier entries where a metric deviates significantly from baseline (z-score)
21. **correlate_metrics** — Statistical correlation between two metrics (Pearson r, p-value)
22. **get_metric_summary** — Summary statistics for a metric (mean, median, std, min, max, trend)
23. **list_available_metrics** — Discover all queryable metrics (8 dimensions, word_count, archetypes, themes, concepts)

---

## Data Flow

### Ingestion (batch, run on-demand)

1. **Vector ingest** (`docker compose run --rm ingest`)
   - Reads `.md` files from corpus directory
   - Chunks and embeds with Sentence-Transformers
   - Stores in ChromaDB with metadata (date, filename, word count)
   - Tracks state in SQLite manifest for incremental re-ingestion

2. **Graph ingest** (`docker compose run --rm graph-ingest`)
   - Reads corpus files
   - Extracts entities using spaCy + custom style profile
   - Creates nodes and relationships in Neo4j
   - Applies entity aliases for name normalization
   - Tracks state in separate SQLite manifest

3. **Batch analysis** (`docker compose --profile batch-analysis run --rm batch-analysis [--full] [--provider anthropic] [--workers 5]`)
   - Dual-provider: Anthropic Claude (preferred) or Ollama (fallback)
   - Claude: single API call per entry for summary + state label (2 calls total)
   - Ollama: chunk-and-merge for entries >1500 words
   - `--provider anthropic|ollama|mock|auto` — override provider selection
   - `--workers N` — parallel workers (Claude handles 5 well; Ollama serializes GPU)
   - Retry logic with mock fallback for resilience
   - Caches results in SQLite (`analysis.sqlite`)
   - Typed entities: `{"name": str, "type": person|place|organization|concept|spiritual}`

4. **Graph enrichment** (`docker compose --profile graph-enrich run --rm graph-enrich [--full]`)
   - Reads from the analysis SQLite database (mounted read-only)
   - Enriches existing Entry nodes with 8 state dimension properties and short summaries
   - Creates Theme, Organization, and Spiritual nodes from typed entities
   - Builds theme co-occurrence network (THEME_COOCCURS relationships)
   - Sets state scores (valence, activation, agency) on MENTIONS relationships
   - Must run **after** graph-ingest (needs Entry nodes to exist)

### Query (runtime — gravity mode)

1. User sends a query via Claude Desktop (MCP)
2. Claude calls `orchestrated_query` with the natural language question
3. **Decompose**: Claude Haiku breaks query into typed fragments + extracts params
4. **Embed**: Fragment vectors + query vector via `/embed` endpoint
5. **Gravity field**: Pull matrix → L2 norm composite → gap detection → activated tools
6. **Dispatch**: All activated tools execute in parallel:
   - **Vector path**: embeddings-service finds semantically similar passages
   - **Graph path**: graph-service traverses relationships, finds connected entities
   - **Analysis path**: analysis-service provides state profiles and summaries
   - **Time series path**: DuckDB queries for trends, anomalies, correlations
7. **Assemble**: Results ranked by composite score
8. Claude synthesizes assembled results into a grounded response
9. Claude may call follow-up tools (`get_entry_analysis`, `get_entries_by_date`, `get_recent_entries`) for deeper exploration

---

## Tuning & Configuration

| File | Purpose |
|------|---------|
| `style_profile.json` | Custom emotion keywords, archetype patterns, concept boosts, transition patterns, stop lists |
| `entity_aliases.json` | Normalize nicknames and phrase variants (e.g., "Mom" → "Mother") |
| `graphrag_feedback.json` | Accumulated user feedback for retrieval improvement |
| `.env` | Corpus path, embedding model, LLM provider, Neo4j credentials, Ollama settings, Anthropic API key |

Supports both **Anthropic Claude API** (preferred) and **Ollama** (local fallback) as analysis backends.

---

## What This Represents

This system was designed and built iteratively — not generated from a template. Key design decisions include:

- **Semantic gravity for tool selection**: Instead of relying on Claude to manually pick tools, the gravity model uses vector similarity to automatically activate relevant tools. This is faster, more deterministic, and catches tools Claude might miss.
- **L2 norm for composite activation**: Multiple moderate pulls from different fragments compound meaningfully (e.g., [0.5, 0.5] → 0.707) without over-weighting any single pull. Physically analogous to gravitational force magnitude.
- **Adaptive gap detection**: No manual tuning — the system finds the natural elbow in activation scores, bounded by [3, 10] tools.
- **Four storage engines**: Vector search finds *similar* passages. Graph search finds *related* ones. Analysis provides psychological depth. Time series enables quantitative trend analysis. Together they give far richer context than any alone.
- **Idempotent ingestion**: MERGE-based graph writes and manifest-tracked vector ingestion mean re-running is safe and incremental.
- **Privacy-conscious**: Can run fully locally with Ollama. When using the Anthropic API, only entry text is sent for analysis — no metadata, credentials, or personal identifiers beyond what's in the writing itself.
- **MCP as the integration layer**: By exposing tools via MCP, the system plugs directly into Claude Desktop, giving Claude structured access to the full corpus as grounding context for conversation.
- **Feedback loop**: User feedback on results is captured and stored, closing the loop between retrieval quality and user experience.

The architecture is domain-agnostic. The same pattern — ingest, embed, graph, gravity orchestrate, expose via MCP — could be applied to any structured or semi-structured corpus.
