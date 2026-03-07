# Time Series Layer

DuckDB-based temporal query layer for corpus-intelligence. Enables trend analysis, anomaly detection, metric correlation, and temporal filtering across all computed entry-level metrics.

## Setup

Dependencies are included in the main `package.json`. After `npm install`, everything is ready.

### Backfill DuckDB from analysis SQLite

```bash
# From mcp-server/
npm run backfill -- ../analysis.sqlite /path/to/corpus/data/2025

# Or with environment variables
ANALYSIS_SQLITE_PATH=../analysis.sqlite CORPUS_PATH=../data/2025 npm run backfill

# Force full re-import (replaces all entries)
npm run backfill -- ../analysis.sqlite ../data/2025 --full
```

Arguments:
1. Path to `analysis.sqlite` (default: `../analysis.sqlite`)
2. Path to corpus directory for word counts (optional)
3. `--full` flag to replace all entries instead of incremental upsert

### DuckDB file location

Default: `./data/timeseries.duckdb` (relative to mcp-server working directory).
Override with `DUCKDB_PATH` environment variable.

## Schema

### `entries` — one row per writing session
| Column | Type | Description |
|--------|------|-------------|
| entry_id | TEXT PK | Matches IDs in ChromaDB/Neo4j |
| entry_date | DATE | Date of writing session |
| word_count | INTEGER | Word count (from corpus files) |
| valence | DOUBLE | -1 (heavy) to +1 (uplifted) |
| activation | DOUBLE | -1 (calm) to +1 (activated) |
| agency | DOUBLE | -1 (stuck) to +1 (empowered) |
| certainty | DOUBLE | -1 (conflicted) to +1 (resolved) |
| relational_openness | DOUBLE | -1 (guarded) to +1 (open) |
| self_trust | DOUBLE | -1 (doubt) to +1 (trust) |
| time_orientation | DOUBLE | -1 (past-looping) to +1 (future-building) |
| integration | DOUBLE | -1 (fragmented) to +1 (coherent) |

### `entry_archetypes` — one row per archetype per entry
| Column | Type |
|--------|------|
| entry_id | TEXT |
| archetype | TEXT |
| strength | DOUBLE |

### `entry_concepts` — themes + entities per entry
| Column | Type |
|--------|------|
| entry_id | TEXT |
| concept | TEXT |
| concept_type | TEXT (theme, person, place, organization, concept, spiritual) |
| weight | DOUBLE |

## MCP Tools

### `query_time_series`
Flexible time series query for any metric over a date range.
- **metric**: dimension name, `word_count`, `archetype:Name`, `theme:name`, `concept:name`
- **granularity**: `entry`, `daily`, `weekly`

### `detect_anomalies`
Flag entries where a metric deviates significantly from baseline.
- Returns z-scores against lookback window mean/std
- Column metrics only (dimensions, word_count)

### `correlate_metrics`
Pearson correlation between two metrics with p-value.
- Returns paired data points and human-readable interpretation
- Column metrics only

### `temporal_filter`
Find entry IDs matching metric conditions — pipe results into existing tools.
- Supports comparison operators: `>`, `>=`, `<`, `<=`, `=`, `!=`
- Column metrics only

### `get_metric_summary`
Summary statistics: mean, median, std, min, max, trend.
- Trend compares last 30 days to overall mean

### `list_available_metrics`
Discovery tool — enumerates all queryable metrics from the data.

## Architecture

```
┌─────────────────────┐
│  MCP Server (Node)  │
│                     │
│  existing tools ────┼──► ChromaDB, Neo4j, Analysis API
│                     │
│  timeseries tools ──┼──► DuckDB (timeseries.duckdb)
└─────────────────────┘
         ▲
         │ backfill ETL
         │
    analysis.sqlite + corpus .md files
```

DuckDB is embedded — no server process, no Docker container. The `.duckdb` file is the only artifact.

## Testing

```bash
npm run build && node --test dist/test/timeseries.test.js
```
