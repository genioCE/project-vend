# Analysis Service

FastAPI service for LLM-backed entry analysis: summarization, typed entity/theme extraction, and 8-dimension psychological state profiling.

## Providers

Dual-provider architecture with automatic resolution:

- **Anthropic Claude** (preferred): Single API call per entry. No chunking needed. Requires `ANTHROPIC_API_KEY`.
  - Model: `claude-haiku-4-5-20251001` (configurable via `ANTHROPIC_MODEL`)
  - Prompt versions: `entry-summary-prompt-claude-v1`, `state-label-prompt-claude-v1`
- **Ollama** (fallback): Chunk-and-merge for entries >1500 words. Requires Ollama running at `OLLAMA_URL`.
  - Model: `llama3.1:8b` (configurable via `OLLAMA_MODEL`)

When both are configured, Claude is preferred. Override with `--provider` flag in batch mode.

## API Endpoints

- `POST /entry-summary/generate` — Generate analysis for a single entry
- `GET /entry-summary/{entry_id}` — Retrieve cached analysis
- `POST /state/label` — Generate state profile for entry text
- `GET /health` — Service health check

## Analysis Output

Each entry receives:
- **short_summary** + **detailed_summary**
- **themes**: 3-8 psychological/behavioral themes (2-4 words each)
- **entities**: Typed (person, place, organization, concept, spiritual)
- **decisions_actions**: Explicit decisions, commitments, realizations
- **state_profile**: 8 dimensions scored -1.0 to +1.0 (valence, activation, agency, certainty, relational_openness, self_trust, time_orientation, integration)

Results are persisted in SQLite (`ENTRY_SUMMARY_DB_PATH`, default `/service/data/analysis.sqlite`).

## Batch Processing

```bash
# Claude (recommended, ~50 min for 719 entries):
source .env && docker compose --profile batch-analysis run --rm batch-analysis --provider anthropic --full --workers 5

# Ollama (local):
docker compose --profile batch-analysis run --rm batch-analysis --provider ollama --full --workers 1

# Dry run:
docker compose --profile batch-analysis run --rm batch-analysis --dry-run
```

Flags:
- `--full` — Force re-analysis of all entries
- `--skip-done` — Skip already-analyzed entries
- `--workers N` — Parallel workers (Claude: 5 recommended, Ollama: 1)
- `--provider anthropic|ollama|mock|auto` — Override provider

## Run (Docker Compose)

```bash
docker compose up -d analysis-service
docker compose logs -f analysis-service
```

## Run (local dev)

```bash
cd analysis-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

## Test

```bash
cd analysis-service
pytest -q
```

## Key Files

| File | Purpose |
|------|---------|
| `entry_summary_provider.py` | `ClaudeEntrySummaryProvider` + `OllamaEntrySummaryProvider` |
| `state_label_provider.py` | `ClaudeStateLabelProvider` + `OllamaStateLabelProvider` |
| `provider_registry.py` | Auto-resolution, provider registration |
| `entry_summary_service.py` | Orchestrates provider + state label + persistence |
| `state_label_service.py` | State label generation + persistence |
| `batch_analyze.py` | CLI batch runner |
| `models.py` | Pydantic models, entity coercion, provider types |
