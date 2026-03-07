# Corpus Intelligence

**Turn any corpus of unstructured writing into a queryable intelligence system with psychological profiling, knowledge graphs, and gravity-based semantic orchestration.**

Corpus Intelligence is an open-source platform that transforms personal writing, journal entries, or any body of unstructured text into a multi-engine knowledge system — searchable by meaning, traversable by relationship, and profiled across 8 psychological dimensions. It was built and battle-tested on a 1.5M-word personal writing corpus spanning 6 years.

The system exposes 22 specialized tools via MCP (Model Context Protocol), orchestrated by the **Gravity Model** — a novel semantic activation framework where queries decompose into typed fragments that exert gravitational pull on tools proportionally to their relevance. No static routing. No keyword matching. Pure semantic gravity.

### What makes this different

- **Gravity-based orchestration**: Queries decompose into semantic fragments (concept, entity, temporal, emotional, relational, archetypal). Each fragment pulls on 22 tool identity vectors via cosine similarity. L2 norm composite activation with adaptive gap detection determines which tools fire. One tool call (`orchestrated_query`) replaces manual tool selection.
- **8-dimensional psychological profiling**: Every entry is scored across valence, activation, agency, certainty, relational openness, self-trust, time orientation, and integration. This isn't sentiment analysis — it's the full shape of a person's internal state.
- **Four storage engines in concert**: Vector search (ChromaDB) finds similar passages. Graph search (Neo4j) finds related entities and concepts. LLM analysis (Claude/Ollama) provides psychological depth. Time series (DuckDB) enables quantitative trend analysis.
- **Domain-agnostic architecture**: The same pattern — ingest, embed, graph, profile, gravity-orchestrate, expose via MCP — works for personal writing, oil and gas documents, legal archives, medical records, or any corpus of unstructured text.
- **Fully local option**: Runs entirely on local infrastructure with Ollama. Anthropic API is optional (recommended for higher quality analysis).

### Why Claude

This system was designed to give Claude the privilege of being the reasoning model for a deeply personal intelligence platform. The gravity model handles orchestration — decomposition, dispatch, and assembly. Claude handles interpretation and meaning-making. Each does what it's best at. The system doesn't ask Claude to be a search engine or a database. It asks Claude to be what it actually is: a reasoning engine that can synthesize insight across multiple dimensions of a human life, grounded in real data surfaced by specialized tools. That's the partnership. The corpus is mine. The orchestration is mine. The reasoning is Claude's.

## Architecture

- **mcp-server** (TypeScript) — MCP stdio server with 22 tools + gravity orchestrator for Claude Desktop
- **embeddings-service** (Python/FastAPI) — Sentence-transformers embeddings + ChromaDB vector search + `/embed` endpoint for gravity vectors
- **graph-service** (Python/FastAPI) — spaCy entity extraction + Neo4j knowledge graph + GraphRAG
- **analysis-service** (Python/FastAPI) — LLM-backed entry summarization, typed entity extraction, theme extraction, and 8-dimension psychological state profiling. Supports Anthropic Claude (preferred) and Ollama (fallback)
- **neo4j** — Graph database storing entities, relationships, and co-occurrences
- **web-ui** (React) — Dark-themed chat interface at localhost:3000
- **ollama** (optional) — Local LLM for analysis when no API key is configured
- **ingest** (Python, one-shot) — Indexes your corpus into ChromaDB
- **graph-ingest** (Python, one-shot) — Builds the knowledge graph in Neo4j
- **gravity-viz** (TypeScript/Three.js) — 3D orbital visualization of the gravity field at localhost:4000

All services communicate over an internal Docker network. By default, only the web UI (port 3000) is exposed to the host.

## Requirements

- Docker Desktop
- 8GB RAM minimum, **16GB recommended**
- ~4GB disk for embeddings + graph
- **One of:**
  - `ANTHROPIC_API_KEY` in `.env` (recommended — uses Claude Haiku for analysis)
  - [Ollama](https://ollama.com) installed natively (local, no API key needed)

## Setup

### 1. Set your corpus path

Edit `.env` and set `CORPUS_PATH` to your writing directory, `NEO4J_PASSWORD`, and optionally your Anthropic API key:

```
CORPUS_PATH=/path/to/your/writing
NEO4J_PASSWORD=your-strong-password
ANTHROPIC_API_KEY=sk-ant-...      # optional: enables Claude-backed analysis
ANTHROPIC_MODEL=claude-haiku-4-5-20251001  # optional: defaults to Haiku
```

### 2. (Optional) Install Ollama for local LLM

Only needed if you don't have an `ANTHROPIC_API_KEY` or want a fully local setup:

```bash
brew install ollama
ollama pull llama3.1:8b
ollama serve   # or: brew services start ollama
```

### 3. Build the Docker images

```bash
docker compose build
```

First build takes ~5 minutes (downloads embedding model + spaCy model + npm/pip dependencies).

### 4. Index your corpus (vector)

```bash
docker compose run --rm ingest
```

### 5. Build the knowledge graph

```bash
docker compose up neo4j -d          # start neo4j first
docker compose run --rm graph-ingest
```

This extracts people, places, concepts, emotions, decisions, archetypes, and transition flows from every entry, then creates nodes and relationships in Neo4j.

### 5b. Run batch analysis (optional)

```bash
# With Anthropic API key (recommended, ~50 min for 719 entries with 5 workers):
source .env && docker compose --profile batch-analysis run --rm batch-analysis --provider anthropic --full --workers 5

# With Ollama (local, slower):
docker compose --profile batch-analysis run --rm batch-analysis --provider ollama --full --workers 1
```

This runs LLM-backed summarization, typed entity extraction, theme extraction, and 8-dimension psychological state profiling for every entry. Results are stored in SQLite. The provider auto-resolves to Claude when `ANTHROPIC_API_KEY` is set.

### 5c. Enrich the graph with analysis data (optional, requires step 5b)

```bash
docker compose --profile graph-enrich run --rm graph-enrich --full
```

This reads the analysis data and enriches the graph with Theme nodes, Organization/Spiritual entity nodes, 8-dimension state profiles on Entry nodes, and state-weighted MENTIONS relationships.

### 6. Start everything

```bash
docker compose up -d
```

Make sure Ollama is running (`ollama serve` or `brew services start ollama`), then:
```bash
open http://localhost:3000
```

Optionally expose Neo4j Browser locally and open it:
```bash
docker compose -f docker-compose.yml -f docker-compose.neo4j-ports.yml up -d neo4j
open http://localhost:7474
```
(Login: neo4j / your `NEO4J_PASSWORD`)

## GraphRAG Mode

Toggle **GraphRAG** in the web UI header. When enabled:

1. Your query is analyzed with spaCy to extract entities
2. Both vector search AND graph traversal run in parallel
3. The enriched context (semantic matches + structural relationships) is sent to the LLM
4. Seven additional graph tools become available for follow-up queries

This is especially powerful for questions about:
- Relationships between people and concepts
- How ideas evolve over time
- Comparing different time periods
- Decisions and their emotional context
- Archetypal patterns in the writing

## Analysis Layer

The `analysis-service` provides LLM-backed per-entry analysis with a dual-provider architecture:

- **Anthropic Claude** (preferred): Single API call per entry, handles full context natively. Model: `claude-haiku-4-5-20251001` (configurable).
- **Ollama** (fallback): Chunk-and-merge strategy for entries >1500 words. Model: `llama3.1:8b`.
- **Auto-resolution**: When `ANTHROPIC_API_KEY` is set, Claude is preferred. Otherwise falls back to Ollama.

Each entry receives:
- **Summaries**: short (1-2 sentences) + detailed (3-5 sentences)
- **Typed entities**: person, place, organization, concept, spiritual
- **Themes**: 3-8 abstract psychological/behavioral themes (2-4 words each)
- **Decisions/actions**: explicit commitments, realizations, choices
- **State profile**: 8-dimension psychological scoring (valence, activation, agency, certainty, relational_openness, self_trust, time_orientation, integration)

Analysis data flows into the graph via the `graph-enrich` pipeline, adding Theme/Organization/Spiritual nodes and state-weighted relationships.

## Example Questions

**Standard mode:**
- "How has my thinking about stillness evolved this year?"
- "What was I processing emotionally in January?"
- "Find everything I wrote about making big decisions"

**GraphRAG mode:**
- "What concepts are connected to recovery in my writing?"
- "How does the Warrior archetype show up compared to the Healer?"
- "Compare my emotional landscape in Q1 vs Q2"
- "What people are most associated with themes of growth?"
- "What decisions have I recorded and what emotions surrounded them?"
- "Where do I explicitly move from one state to another (for example fear -> action)?"

## Personal Style Tuning

Graph extraction is now tunable through two files:

- `graph-service/config/style_profile.json`
- `graph-service/config/entity_aliases.json`

Use these to:

- Add custom emotion/archetype keywords.
- Boost concepts and phrases that matter in your writing voice.
- Add alias normalization (for example nicknames, shorthand, repeated phrase variants).
- Add extra transition patterns for flow detection.

### Feedback-based reranking

In GraphRAG mode, assistant messages now include `Helpful` / `Off-target` feedback buttons when graph evidence is used.

- Positive feedback boosts similar concepts/entities/sources in future retrieval.
- Negative feedback downweights those signals.
- `Off-target` now supports an optional free-text note (`How can we improve?`) for qualitative guidance.
- A `Review Feedback` module in the web UI generates actionable prompt backlogs you can paste into Codex/Claude Code to improve system language behavior.
- Feedback profile is persisted in `graph-feedback-data` volume (`/service/data/graphrag_feedback.json`).

## Claude Desktop Integration

The MCP stdio server still works alongside the web UI. Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "corpus-intelligence": {
      "command": "docker",
      "args": [
        "compose",
        "-f", "/path/to/corpus-intelligence/docker-compose.yml",
        "run", "--rm", "-T", "mcp-server", "node", "dist/index.js"
      ]
    }
  }
}
```

Restart Claude Desktop. In gravity mode (default), Claude sees 4 tools: `orchestrated_query` + 3 follow-up tools. Set `GRAVITY_MODE=0` in the environment to expose all 23 tools individually.

## Gravity Orchestration

The gravity model is the primary query interface. Instead of Claude manually selecting from 22 tools, it calls `orchestrated_query` with a natural language question. The system then:

1. **Decomposes** the query into typed semantic fragments (concept, entity, temporal, emotional, relational, archetypal) using Claude Haiku
2. **Embeds** each fragment + the full query via the embeddings service
3. **Computes a gravity field** — cosine similarity between fragment vectors and 22 tool identity vectors produces a pull matrix. L2 norm across pulls gives each tool a composite activation score.
4. **Activates tools** via adaptive gap detection (finds the natural elbow in sorted scores, min 3 / max 10 tools)
5. **Dispatches** all activated tools in parallel with 10s per-tool timeouts
6. **Assembles** results ranked by composite score for Claude to synthesize

### Gravity Mode (default)

When `GRAVITY_MODE` is on (default), only 4 tools are exposed to Claude:

| Tool | Purpose |
|------|---------|
| `orchestrated_query` | Multi-tool gravity dispatch |
| `get_entry_analysis` | Follow-up deep dive on specific entries |
| `get_entries_by_date` | Follow-up date-range retrieval |
| `get_recent_entries` | Follow-up recent entries |

### 3D Gravity Visualization

A diagnostic visualization renders the gravity field as a 3D orbital system:

```bash
cd mcp-server && source ../.env && npx tsx gravity-viz.ts
# Open http://localhost:4000
```

- Fragment nodes on a ring, tools orbiting around their attracting fragments
- Orbit radius = inverse of composite score (stronger pull = tighter orbit)
- Orbital plane inclination per tool for 3D depth
- Pull lines connecting activated tools to fragments
- Mouse drag to rotate, scroll to zoom
- Sidebar shows activation scores, fragments, and extracted params

## Available Tools (22 + 1 orchestrator)

| Tool | Category | Description |
|------|----------|-------------|
| `orchestrated_query` | Orchestrator | Automatic multi-tool dispatch via semantic gravity |
| `search_writings` | Search | Semantic search — finds passages similar in meaning to your query |
| `search_by_keyword` | Search | Exact text search with surrounding context |
| `get_entries_by_date` | Search | Retrieve all entries within a date range |
| `get_recent_entries` | Search | Get the N most recent entries |
| `find_recurring_themes` | Pattern | Trace how a topic evolves over time |
| `get_writing_stats` | Meta | Corpus statistics (word count, date range, entries) |
| `get_entry_analysis` | Analysis | Per-entry summary, themes, entities, 8-D state profile |
| `find_connected_concepts` | Graph | Find concepts, people, and emotions connected to a concept |
| `trace_concept_evolution` | Graph | Trace how a concept appears over time with co-occurring entities |
| `get_concept_flows` | Graph | Find directed transition flows (X -> Y) connected to a concept |
| `find_entity_relationships` | Graph | Map a person's presence across the corpus |
| `compare_periods` | Graph | Compare concepts, emotions, and archetypes between two time periods |
| `get_decision_context` | Graph | Find decisions and their emotional/conceptual context |
| `get_archetype_patterns` | Graph | Get archetypal patterns (Warrior, Sage, Creator, etc.) |
| `search_themes` | Graph | Find entries linked to a theme with co-occurring themes |
| `search_by_state` | Graph | Filter entries by psychological state dimension |
| `temporal_filter` | Graph | Filter entries by metric thresholds in a date range |
| `query_time_series` | Quantitative | Plot any metric over time (entry/daily/weekly granularity) |
| `detect_anomalies` | Quantitative | Flag outlier entries via z-score deviation |
| `correlate_metrics` | Quantitative | Pearson correlation between two metrics |
| `get_metric_summary` | Quantitative | Summary stats for a metric (mean, median, std, trend) |
| `list_available_metrics` | Meta | Discover all queryable metrics |

## Knowledge Graph Schema

**Node types:** Entry, Person, Place, Concept, Emotion, Decision, Archetype, Theme, Organization, Spiritual

**Relationships:** MENTIONS, CONTAINS, HAS_THEME, EXPRESSES, RECORDS, INVOKES, COOCCURS_WITH, THEME_COOCCURS, FLOWS_TO

## Switching Analysis Providers

**Claude (recommended):**
```bash
# Set in .env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001  # or claude-sonnet-4-6 for higher quality
```

**Ollama (local):**
```bash
# Set in .env
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1:8b
```

## Common Operations

**Update the vector index after adding new writing:**
```bash
docker compose run --rm ingest
```

**Rebuild the knowledge graph after adding new writing:**
```bash
docker compose run --rm graph-ingest
```

**Stop services:**
```bash
docker compose down
```

**Fully reset and re-index:**
```bash
docker compose down -v    # WARNING: deletes vector index + graph data
docker compose run --rm ingest
docker compose up neo4j -d && sleep 10
docker compose run --rm graph-ingest
```

**View logs:**
```bash
docker compose logs -f mcp-server
docker compose logs -f graph-service
docker compose logs -f embeddings-service
docker compose logs -f analysis-service
```

## Quality Checks

```bash
# Type/lint/test checks for the MCP server
cd mcp-server && npm run check

# Type/lint/build checks for the web UI
cd ../web-ui && npm run check

# Python utility tests (embeddings service)
cd ../embeddings-service && pytest -q
```

## Roadmap

This system works. It's been battle-tested on 1,414 entries and 1.5M words of real data across six years. But it was built fast, by one person, in the heat of discovery. The code reflects that — it's functional, not polished. Here's where it's going.

### Now

- **Open source release** — you're looking at it. The architecture is public. Bring your own corpus.
- **Wellfile-Intelligence** — same architecture adapted for oil and gas document processing. Different tool registry, different fragment taxonomy, same gravity math. Proof that the pattern is domain-agnostic.

### Next

- **Codebase refactor** — the current code was written to prove the architecture works. It did. Now it needs to be cleaned up, better typed, better tested, and better documented at the function level. The bones are right. The skin needs work.
- **Gravity model hardening** — the L2 norm activation and gap detection work well empirically. Need formal benchmarking against manually-evaluated query sets. The learning loop (ledger) needs more data before reliability bias is fully trusted.
- **Packaging for others** — right now setup requires Docker knowledge and manual corpus preparation. The goal is: clone, point at a folder of text files, run one command, start querying.
- **Predictive layer** — with enough longitudinal data, the system can forecast psychological vulnerability windows, optimal performance periods, and pre-crisis linguistic markers. The math is straightforward time series forecasting on the 8-dimensional profiles. The data just needs depth.

### Later

- **Language refactor evaluation** — once the orchestration pattern is stable and proven at scale, evaluate Rust (zero-cost concurrency), Go (goroutine-native dispatch), or Elixir/OTP (fault-tolerant supervised agent processes) for the orchestrator core.
- **Multi-corpus federation** — a single gravity orchestrator dispatching across multiple corpora simultaneously. Personal writing + oil and gas documents + legal archives, all queryable through one interface.
- **Voice and real-time input** — the architecture doesn't care if text comes from a keyboard, a voice transcript, or eventually a neural interface. The capture method is temporary. The processing layer is permanent.
- **Clinical and recovery applications** — longitudinal psychological profiling for therapy, recovery programs, and mental health monitoring. The 8-dimensional model was built for one person. It was designed to work for anyone.

### Philosophy

This system was built by hand because no one else was building it. It came from a personal writing practice, a recovery journey, and the realization that a million words of honest self-examination is both the hardest dataset to produce and the most valuable one to process. The code will get better. The architecture will scale. The corpus keeps growing every day.

Contributions, questions, and forks are welcome. If you build something on this, I want to hear about it.

---

*Built in Oklahoma City by [Jer Nguyen](https://www.linkedin.com/in/jernguyen/) — self-taught AI engineer, systems architect, writer.*
