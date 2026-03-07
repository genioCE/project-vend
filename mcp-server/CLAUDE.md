# CLAUDE.md — Unified Query Layer

## Project Context

This is the `corpus-intelligence` MCP server — a personal knowledge system built on ~1M words of daily journal entries. It has 4 storage engines:

1. **ChromaDB** (embeddings-service:8000) — semantic vector search, keyword search, date retrieval
2. **Neo4j** (graph-service:8001) — knowledge graph with entities, concepts, themes, archetypes, emotions, decisions, and their relationships
3. **SQLite** (analysis-service:8002) — per-entry pre-computed analysis: summaries, typed entities, themes, 8-dimension psychological state profiles
4. **DuckDB** (timeseries, in-process in mcp-server) — quantitative time series, anomaly detection, correlations, metric summaries

The MCP server (TypeScript/Node.js) exposes 23 individual tools that each query one engine. Claude (the conversation model) currently orchestrates multi-engine queries by making multiple tool calls and stitching results together manually.

## Task

Build a **unified query layer** — a new MCP tool called `deep_query` that takes a natural language question, decomposes it into sub-queries for each relevant engine, executes them in parallel, and returns one assembled result.

**Read `UNIFIED_QUERY_SPEC.md` in this directory before starting.** It contains the full design spec: types, classifier design, executor design, assembler design, tool registration, and implementation order.

## Codebase Orientation

```
mcp-server/
├── src/
│   ├── index.ts                 — Entry point. Registers tool sets. ADD registerUnifiedTools() here.
│   ├── tools.ts                 — 17 existing MCP tools (vector + graph + analysis)
│   ├── embeddings-client.ts     — HTTP client for ChromaDB service
│   ├── graph-client.ts          — HTTP client for Neo4j service
│   ├── analysis-client.ts       — HTTP client for SQLite analysis service
│   ├── timeseries/
│   │   ├── tools.ts             — 6 timeseries MCP tools
│   │   ├── db.ts                — DuckDB query executor
│   │   ├── metrics.ts           — Metric resolution and SQL builders
│   │   ├── schema.ts            — DuckDB table schema
│   │   └── types.ts             — Dimension types
│   ├── agent/                   — Web UI agent loop (not relevant to this task)
│   ├── agent.ts                 — Web UI agent (not relevant to this task)
│   └── llm/                     — LLM provider abstraction (not relevant)
├── UNIFIED_QUERY_SPEC.md        — THE FULL DESIGN SPEC. READ THIS FIRST.
├── package.json
└── tsconfig.json
```

### Key client functions you'll import:

**From `embeddings-client.ts`:**
- `searchWritings(query, topK, signal?, requestId?)` → semantic search results with text, date, source_file, relevance_score
- `searchByKeyword(keyword, contextWords, signal?, requestId?)` → exact text matches
- `getEntriesByDate(startDate, endDate, signal?, requestId?)` → full entries in a date range

**From `graph-client.ts`:**
- `findConnectedConcepts(name, limit, signal?, requestId?)` → concept network
- `findEntityRelationships(name, limit, signal?, requestId?)` → person's entries + co-occurring concepts/emotions
- `traceConceptEvolution(name, limit, signal?, requestId?)` → concept over time with emotions/people
- `comparePeriods(start1, end1, start2, end2, signal?, requestId?)` → two period comparison
- `getDecisionContext(keyword?, limit, signal?, requestId?)` → decisions with emotional context
- `getArchetypePatterns(limit, signal?, requestId?)` → archetype frequency/strength/emotions
- `getConceptFlows(name, limit, signal?, requestId?)` → directed concept transitions
- `getThemeNetwork(name, limit, signal?, requestId?)` → theme entries + co-occurring themes
- `getEntriesByState(dimension, min, max, limit, signal?, requestId?)` → entries filtered by psychological state

**From `analysis-client.ts`:**
- `getEntrySummary(entryId, signal?, requestId?)` → single entry analysis
- `getEntrySummaries(entryIds, signal?, requestId?)` → batch entry analysis (returns Map)

**From `timeseries/db.ts`:**
- `query(sql, params?)` → raw DuckDB query

**From `timeseries/metrics.ts`:**
- `resolveMetric(name)` → ResolvedMetric | null
- `buildTimeSeriesQuery(metric, start, end, granularity, options?)` → { sql, params }

## New files to create

```
mcp-server/src/unified/
├── types.ts            — QueryPlan, QueryIntent, UnifiedResult, UnifiedEntry, EngineResult, etc.
├── classifier.ts       — classify(question) → QueryPlan (rule-based, no LLM)
├── executor.ts         — execute(plan, options) → EngineResult[] (parallel fan-out)
├── assembler.ts        — assemble(plan, results, options) → UnifiedResult
└── tools.ts            — registerUnifiedTools(server) — registers deep_query MCP tool
```

Then modify `index.ts` to add: `import { registerUnifiedTools } from "./unified/tools.js";` and call `registerUnifiedTools(server);`

## Constraints

- **No LLM calls in the classifier or assembler.** Rule-based decomposition only. Claude (conversation model) handles interpretation.
- **No new npm packages.** Use existing deps (zod, node builtins).
- **No changes to existing tools or clients.** This is purely additive.
- **Use `Promise.allSettled`** for parallel execution. Never let one engine's failure block others.
- **Per-engine timeouts** via `AbortController` (8s default). Slow engines get cut off gracefully.
- **The assembler returns structured data, not narrative.** Claude turns it into narrative.

## Build & Test

```bash
cd mcp-server
npm run build                    # runs tsc via local typescript
cd ..
docker compose build mcp-server  # rebuild container
docker compose restart mcp-server
```

Do NOT use `npx tsc` directly — the project uses its own local typescript via `npm run build`.

## Known Bug (fixed in clients, don't reintroduce)

All three HTTP clients previously had a double-read bug in error handling:
```typescript
// BAD — response.json() consumes the stream, then response.text() fails
try { await response.json(); } catch { await response.text(); }

// GOOD — read text first, then try to parse as JSON
const rawText = await response.text();
try { JSON.parse(rawText); } catch { /* use rawText directly */ }
```
This has been fixed in embeddings-client.ts, graph-client.ts, and analysis-client.ts. If you add any new HTTP clients for the unified layer, use the text-first pattern.

To test the tool, use the web UI at localhost:3000 or query via Claude Desktop MCP.

For unit testing the classifier, create `src/test/classifier.test.ts` with known question → plan mappings.

## Known Entity Names (for classifier)

These are frequently occurring people in the corpus. The classifier can use these for entity extraction:
Kyle, Matt, Brian, Ruth, Kelsey, Morgan, Donny, Justice, Antonio, Luke, Dylan, Vell, Tyler, Mom, Tony, Johnny, Lindsey, Cam, Milan, Butch, Arica, Lane, Steven, Adina, Gene, Nelson, Chuck, David, Megan, Amy, Cassie, Johna, Ford, Jett, Brenna, Barry, Taylor, Sean, Heather, Tiffany, Jordan, Rachel, Ashley

## Known Psychological Dimensions (for classifier)

valence, activation, agency, certainty, relational_openness, self_trust, time_orientation, integration

Colloquial mappings the classifier should handle:
- "stuck" / "empowered" / "agency" → agency
- "heavy" / "uplifted" / "mood" → valence  
- "fragmented" / "coherent" / "together" / "falling apart" → integration
- "doubt" / "trust" / "self-trust" / "confident" → self_trust
- "calm" / "activated" / "energy" / "intensity" → activation
- "conflicted" / "resolved" / "certain" / "clarity" → certainty
- "guarded" / "open" / "vulnerable" / "closed off" → relational_openness
- "past" / "future" / "forward" / "dwelling" / "looping" → time_orientation
