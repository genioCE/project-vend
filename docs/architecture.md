# Corpus Intelligence Architecture (C4)

This document describes the architecture of Corpus Intelligence using a C4-style structure: Context, Containers, Components, and runtime/data flows.

## 1) Scope and Goals

Corpus Intelligence is a local-first AI assistant for exploring a personal writing corpus.

Goals:

1. Keep data local to the machine.
2. Support conversational retrieval with semantic search.
3. Support structural reasoning with a knowledge graph (GraphRAG).
4. Expose the same capabilities to both a web UI and MCP clients.

## 2) C1: System Context

External actors and systems:

1. User: interacts through browser UI or optionally Claude Desktop.
2. Local corpus: markdown files on host filesystem, mounted read-only into containers.
3. Anthropic API (optional): Claude Haiku for analysis batch processing.
4. Ollama (optional, host process): local LLM fallback for analysis.
5. Claude Desktop (optional): calls MCP stdio mode for the same toolset.

System boundary contains:

1. `web-ui` (React + Nginx).
2. `mcp-server` (Node/TypeScript orchestration).
3. `embeddings-service` (FastAPI + SentenceTransformers + ChromaDB).
4. `graph-service` (FastAPI + spaCy + Neo4j + GraphRAG composition).
5. `analysis-service` (FastAPI + SQLite + Anthropic SDK / Ollama).
6. `neo4j` database.

## 3) C2: Container View

### `web-ui`

1. Tech: React (Vite build), served by Nginx.
2. Responsibilities: render chat UI and GraphRAG toggle, stream SSE updates, persist UI conversation state in localStorage.
3. Interfaces: incoming `http://localhost:3000`; outgoing proxied `/api/*` to `mcp-server:3001`.

### `mcp-server`

1. Tech: Node.js + TypeScript + Express + MCP SDK.
2. Responsibilities: run agent loop, expose HTTP API for web UI, expose MCP stdio server, manage in-memory history and tool-result cache.
3. Interfaces: incoming HTTP from `web-ui`; outgoing HTTP to `embeddings-service`, `graph-service`, and host Ollama; incoming stdio from MCP clients.

### `embeddings-service`

1. Tech: FastAPI + SentenceTransformers + ChromaDB.
2. Responsibilities: semantic search, theme tracing, corpus stats, raw-file date-range/recent/keyword retrieval.
3. Data: Chroma persistent store (`chroma-data` volume), read-only corpus mount at `/corpus`.

### `graph-service`

1. Tech: FastAPI + spaCy + Neo4j driver + httpx.
2. Responsibilities: graph query APIs and GraphRAG composition endpoint (`/graph/search`).
3. Data: reads/writes Neo4j; calls `embeddings-service` during GraphRAG composition.

### `analysis-service`

1. Tech: FastAPI + SQLite + Anthropic SDK + Ollama client.
2. Responsibilities: LLM-backed entry summarization, typed entity/theme extraction, 8-dimension psychological state profiling.
3. Providers: Anthropic Claude (preferred, single call per entry) and Ollama (fallback, chunk-and-merge for long entries).
4. Data: SQLite persistent store (`analysis-data` volume) for entry summaries and state labels.
5. Interfaces: incoming HTTP from `mcp-server`; outgoing HTTPS to Anthropic API or HTTP to Ollama.

### `neo4j`

1. Tech: Neo4j Community.
2. Responsibilities: persist entities and relationships extracted from corpus; serve graph query workloads.
3. Data: persistent graph/log volumes.

## 4) C3: Component View

### Components inside `mcp-server`

1. `api-server.ts`: HTTP endpoints (`/chat`, `/health`, `/graph/subgraph`) and SSE event delivery.
2. `agent.ts`: orchestration loop, LLM calls, tool execution, GraphRAG context injection, history/cache limits.
3. `embeddings-client.ts`: typed client for embeddings endpoints.
4. `graph-client.ts`: typed client for graph endpoints.
5. `tools.ts`: MCP tool registration and schema validation.
6. `index.ts`: MCP stdio bootstrap.

### Components inside `embeddings-service`

1. `main.py`: API routes, request handling, corpus file cache/date parsing, result shaping.
2. `embeddings.py`: embedding model load and encoding.
3. `vectorstore.py`: Chroma collection lifecycle and query operations.
4. `ingest.py`: batch ingest pipeline (markdown strip -> chunk -> embed -> index).

### Components inside `graph-service`

1. `main.py`: graph API surface and GraphRAG search endpoint.
2. `graph.py`: Neo4j driver management, schema/indexes, graph query/write functions.
3. `extractor.py`: spaCy + rule extraction for people, places, concepts, emotions, decisions, archetypes.
4. `graphrag.py`: entity extraction from query, parallel vector+graph retrieval, formatted context synthesis.
5. `graph_ingest.py`: batch graph rebuild from corpus.

## 5) Data Model Summary

### Vector index (Chroma)

Chunk metadata fields include:

1. `date`
2. `source_file`
3. `chunk_index`
4. `total_chunks`
5. `word_count`
6. `total_entry_words`
7. `year`
8. `month`

### Analysis store (SQLite)

Tables:

1. `entry_summaries` — entry_id, payload_json, model_version, prompt_version, schema_version, created_at
2. `state_labels` — entry_id, payload_json, model_version, prompt_version, schema_version, created_at

### Knowledge graph (Neo4j)

Node labels:

1. `Entry`
2. `Person`
3. `Place`
4. `Concept`
5. `Emotion`
6. `Decision`
7. `Archetype`
8. `Theme`
9. `Organization`
10. `Spiritual`

Relationship types:

1. `MENTIONS`
2. `CONTAINS`
3. `HAS_THEME`
4. `EXPRESSES`
5. `RECORDS`
6. `INVOKES`
7. `COOCCURS_WITH`
8. `THEME_COOCCURS`
9. `FLOWS_TO`

## 6) Key Runtime Flows

### A. Standard chat

1. Browser posts message to `/api/chat`.
2. `web-ui` proxy forwards to `mcp-server`.
3. `mcp-server` runs LLM turn with tool schemas.
4. LLM may call vector tools.
5. `mcp-server` executes tool calls on `embeddings-service`.
6. Tool outputs are fed back into LLM until final response.
7. SSE streams tokens and tool status back to browser.

### B. GraphRAG chat

1. Same as standard chat, with GraphRAG enabled.
2. `mcp-server` calls `graph-service /graph/search` for pre-context.
3. `graph-service` extracts entities from query.
4. `graph-service` runs vector search and graph lookups in parallel.
5. `graph-service` returns merged `formatted_context`.
6. `mcp-server` injects that context into system prompt.
7. Normal tool-calling loop continues with graph tools also available.

### C. Vector ingestion batch

1. `ingest` scans corpus markdown files.
2. Markdown is stripped and entries are chunked.
3. Chunks are embedded in batches.
4. Chroma collection is rebuilt with vectors and metadata.

### D. Graph ingestion batch

1. `graph-ingest` scans corpus files.
2. Existing graph is cleared and indexes are created.
3. Entities/patterns are extracted per entry.
4. Nodes and relationships are written to Neo4j.

### E. Batch analysis

1. `batch-analysis` scans corpus files.
2. Provider is resolved (Claude preferred, Ollama fallback).
3. Each entry gets two LLM calls: entry summary + state label.
4. Results are persisted to SQLite (`analysis.sqlite`).
5. Claude processes entries in a single call; Ollama uses chunk-and-merge for entries >1500 words.

### F. Graph enrichment (post-analysis)

1. `graph-enrich` reads analysis data from SQLite (read-only mount).
2. Enriches existing Entry nodes with 8 state dimension properties and short summaries.
3. Creates Theme, Organization, and Spiritual nodes from typed entities.
4. Builds theme co-occurrence network (THEME_COOCCURS relationships).
5. Sets state scores (valence, activation, agency) on MENTIONS relationships.

## 7) Deployment and Networking

1. Orchestration: Docker Compose on internal network `corpus-net`.
2. Host-exposed ports: `3000` (web UI). Neo4j ports are internal by default and can be exposed via compose override when needed.
3. Internal service ports: `3001` (`mcp-server`), `8000` (`embeddings-service`), `8001` (`graph-service`).
4. LLM endpoint from containers: `http://host.docker.internal:11434`.

## 8) Reliability and Operations

1. Health checks exist for Neo4j, embeddings-service, and graph-service.
2. GraphRAG gracefully degrades if graph-service is unavailable.
3. Conversation memory and caches are in-process and reset on restart.
4. Durable data persistence is in Chroma/Neo4j volumes and corpus files.

## 9) Security and Privacy Characteristics

1. Local-first design with no cloud API dependency in normal operation.
2. Corpus is mounted read-only into processing services.
3. Main exposure surface is host-published ports.

## 10) Constraints and Tradeoffs

1. Agent state is not distributed and not shared across replicas.
2. Graph extraction is heuristic (spaCy + rules), not canonical NLP parsing.
3. Graph ingest scans `*.md` recursively at the configured corpus path.
4. Tool loop is bounded by max turns and prompt/context limits.
5. Analysis requires either an Anthropic API key or a running Ollama instance.
6. Graph enrichment is a second-order projection — must run after both graph-ingest and batch-analysis.
