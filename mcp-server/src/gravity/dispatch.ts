/**
 * Parallel tool dispatch.
 *
 * Maps activated tool names to actual backend client calls using
 * extracted parameters from the decomposition step.
 */

import type { ActivatedTool, ExtractedParams, ToolResult } from "./types.js";

// Embeddings service clients
import {
  searchWritings,
  getEntriesByDate,
  findRecurringThemes,
  getWritingStats,
  getRecentEntries,
  searchByKeyword,
} from "../embeddings-client.js";

// Graph service clients
import {
  findConnectedConcepts,
  findEntityRelationships,
  traceConceptEvolution,
  comparePeriods,
  getDecisionContext,
  getArchetypePatterns,
  getConceptFlows,
  getThemeNetwork,
  getEntriesByState,
} from "../graph-client.js";

// Timeseries (in-process DuckDB)
import { query as duckQuery } from "../timeseries/db.js";
import { ensureSchema } from "../timeseries/schema.js";
import {
  resolveMetric,
  buildTimeSeriesQuery,
  pearsonPValue,
  interpretCorrelation,
} from "../timeseries/metrics.js";

const TOOL_TIMEOUT_MS = 10_000;

let _schemaReady = false;
async function readyTimeseries(): Promise<void> {
  if (!_schemaReady) {
    await ensureSchema();
    _schemaReady = true;
  }
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function defaultStart(): string {
  const d = new Date();
  d.setDate(d.getDate() - 90);
  return d.toISOString().slice(0, 10);
}

type DispatchFn = (
  params: ExtractedParams,
  query: string,
  signal?: AbortSignal
) => Promise<string>;

const DISPATCH_TABLE: Record<string, DispatchFn> = {
  // ── Embeddings service tools ────────────────────────────────────

  search_writings: async (p, q, s) => {
    const r = await searchWritings(p.search_query || q, 8, s);
    return JSON.stringify(r, null, 2);
  },

  search_by_keyword: async (p, q, s) => {
    const keyword = p.entities[0] || p.concepts[0] || q;
    const r = await searchByKeyword(keyword, 100, s);
    return JSON.stringify(r, null, 2);
  },

  get_entries_by_date: async (p, _q, s) => {
    const range = p.date_ranges[0];
    if (!range?.start || !range?.end) {
      return JSON.stringify({ error: "No date range extracted from query" });
    }
    const r = await getEntriesByDate(range.start, range.end, s);
    return JSON.stringify(r, null, 2);
  },

  get_recent_entries: async (_p, _q, s) => {
    const r = await getRecentEntries(7, s);
    return JSON.stringify(r, null, 2);
  },

  find_recurring_themes: async (p, q, s) => {
    const topic = p.search_query || p.concepts[0] || q;
    const r = await findRecurringThemes(topic, 10, s);
    return JSON.stringify(r, null, 2);
  },

  get_writing_stats: async (_p, _q, s) => {
    const r = await getWritingStats(s);
    return JSON.stringify(r, null, 2);
  },

  // ── Graph service tools ─────────────────────────────────────────

  find_connected_concepts: async (p, q, s) => {
    const name = p.concepts[0] || p.entities[0] || q;
    const r = await findConnectedConcepts(name, 30, s);
    return JSON.stringify(r, null, 2);
  },

  find_entity_relationships: async (p, q, s) => {
    const name = p.entities[0] || q;
    const r = await findEntityRelationships(name, 20, s);
    return JSON.stringify(r, null, 2);
  },

  trace_concept_evolution: async (p, q, s) => {
    const name = p.concepts[0] || p.entities[0] || q;
    const r = await traceConceptEvolution(name, 20, s);
    return JSON.stringify(r, null, 2);
  },

  get_concept_flows: async (p, q, s) => {
    const name = p.concepts[0] || q;
    const r = await getConceptFlows(name, 20, s);
    return JSON.stringify(r, null, 2);
  },

  search_themes: async (p, q, s) => {
    const name = p.concepts[0] || q;
    const r = await getThemeNetwork(name, 30, s);
    return JSON.stringify(r, null, 2);
  },

  search_by_state: async (p, _q, s) => {
    const dim = p.metrics[0] || "valence";
    const r = await getEntriesByState(dim, -1, 1, 20, s);
    return JSON.stringify(r, null, 2);
  },

  compare_periods: async (p, _q, s) => {
    if (p.date_ranges.length < 2) {
      return JSON.stringify({ error: "Need two date ranges for comparison" });
    }
    const [r1, r2] = p.date_ranges;
    const r = await comparePeriods(
      r1.start || "",
      r1.end || "",
      r2.start || "",
      r2.end || "",
      s
    );
    return JSON.stringify(r, null, 2);
  },

  get_decision_context: async (p, _q, s) => {
    const keyword = p.concepts[0] || undefined;
    const r = await getDecisionContext(keyword, 10, s);
    return JSON.stringify(r, null, 2);
  },

  get_archetype_patterns: async (_p, _q, s) => {
    const r = await getArchetypePatterns(10, s);
    return JSON.stringify(r, null, 2);
  },

  // ── Timeseries tools (in-process DuckDB) ────────────────────────

  query_time_series: async (p) => {
    await readyTimeseries();
    const metricName = p.metrics[0];
    if (!metricName) {
      return JSON.stringify({ error: "No metric extracted from query" });
    }
    const resolved = resolveMetric(metricName);
    if (!resolved) {
      return JSON.stringify({ error: `Unknown metric: ${metricName}` });
    }
    const range = p.date_ranges[0];
    const sd = range?.start || defaultStart();
    const ed = range?.end || today();
    const { sql, params } = buildTimeSeriesQuery(resolved, sd, ed, "daily");
    const rows = await duckQuery(sql, params);
    return JSON.stringify(
      { metric: metricName, granularity: "daily", start_date: sd, end_date: ed, count: rows.length, data: rows },
      null,
      2
    );
  },

  detect_anomalies: async (p) => {
    await readyTimeseries();
    const metricName = p.metrics[0];
    if (!metricName) {
      return JSON.stringify({ error: "No metric extracted from query" });
    }
    const resolved = resolveMetric(metricName);
    if (!resolved || resolved.kind !== "column") {
      return JSON.stringify({ error: `Unsupported metric for anomaly detection: ${metricName}` });
    }
    const col = resolved.column!;
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - 180);
    const cutoffStr = cutoff.toISOString().slice(0, 10);
    const [stats] = await duckQuery(
      `SELECT AVG(${col}) AS mean, STDDEV_SAMP(${col}) AS std, COUNT(*) AS n
       FROM entries WHERE ${col} IS NOT NULL AND entry_date >= ? AND data_quality = 'clean'`,
      [cutoffStr]
    );
    const mean = stats.mean as number;
    const std = stats.std as number;
    if (!std || std === 0) {
      return JSON.stringify({ metric: metricName, anomaly_count: 0, anomalies: [] });
    }
    const anomalies = await duckQuery(
      `SELECT entry_id, entry_date::TEXT AS date, ${col} AS value,
              ABS((${col} - ?) / ?) AS z_score
       FROM entries
       WHERE ${col} IS NOT NULL AND entry_date >= ? AND data_quality = 'clean'
         AND ABS((${col} - ?) / ?) > 2.0
       ORDER BY z_score DESC`,
      [mean, std, cutoffStr, mean, std]
    );
    return JSON.stringify(
      { metric: metricName, baseline: { mean, std }, anomaly_count: anomalies.length, anomalies },
      null,
      2
    );
  },

  correlate_metrics: async (p) => {
    await readyTimeseries();
    if (p.metrics.length < 2) {
      return JSON.stringify({ error: "Need two metrics for correlation" });
    }
    const [nameA, nameB] = p.metrics;
    const ra = resolveMetric(nameA);
    const rb = resolveMetric(nameB);
    if (!ra || !rb || ra.kind !== "column" || rb.kind !== "column") {
      return JSON.stringify({ error: `Unsupported metrics for correlation: ${nameA}, ${nameB}` });
    }
    const colA = ra.column!;
    const colB = rb.column!;
    const sd = p.date_ranges[0]?.start || defaultStart();
    const ed = p.date_ranges[0]?.end || today();
    const [corrRow] = await duckQuery(
      `WITH paired AS (
        SELECT DATE_TRUNC('week', entry_date)::DATE::TEXT AS date,
               AVG(${colA}) AS va, AVG(${colB}) AS vb
        FROM entries
        WHERE entry_date BETWEEN ? AND ?
          AND ${colA} IS NOT NULL AND ${colB} IS NOT NULL AND data_quality = 'clean'
        GROUP BY DATE_TRUNC('week', entry_date)::DATE
      )
      SELECT CORR(va, vb) AS r, COUNT(*) AS n FROM paired`,
      [sd, ed]
    );
    const r = corrRow.r as number;
    const n = corrRow.n as number;
    if (r == null || isNaN(r)) {
      return JSON.stringify({ error: "Could not compute correlation" });
    }
    const pValue = pearsonPValue(r, n);
    return JSON.stringify(
      { metric_a: nameA, metric_b: nameB, pearson_r: +r.toFixed(4), p_value: +pValue.toFixed(6), n, interpretation: interpretCorrelation(r, pValue) },
      null,
      2
    );
  },

  get_metric_summary: async (p) => {
    await readyTimeseries();
    const metricName = p.metrics[0];
    if (!metricName) {
      return JSON.stringify({ error: "No metric extracted from query" });
    }
    const resolved = resolveMetric(metricName);
    if (!resolved || resolved.kind !== "column") {
      return JSON.stringify({ error: `Unsupported metric for summary: ${metricName}` });
    }
    const col = resolved.column!;
    const sd = p.date_ranges[0]?.start || "1900-01-01";
    const ed = p.date_ranges[0]?.end || today();
    const [stats] = await duckQuery(
      `SELECT AVG(${col}) AS mean, MEDIAN(${col}) AS median,
              STDDEV_SAMP(${col}) AS std, MIN(${col}) AS min,
              MAX(${col}) AS max, COUNT(*) AS entries_count
       FROM entries
       WHERE ${col} IS NOT NULL AND entry_date BETWEEN ? AND ? AND data_quality = 'clean'`,
      [sd, ed]
    );
    return JSON.stringify(
      { metric: metricName, ...stats },
      null,
      2
    );
  },

  temporal_filter: async (p) => {
    await readyTimeseries();
    const metricName = p.metrics[0];
    if (!metricName) {
      return JSON.stringify({ error: "No metric extracted from query" });
    }
    const resolved = resolveMetric(metricName);
    if (!resolved || resolved.kind !== "column") {
      return JSON.stringify({ error: `Unsupported metric for filter: ${metricName}` });
    }
    const col = resolved.column!;
    const sd = p.date_ranges[0]?.start || "1900-01-01";
    const ed = p.date_ranges[0]?.end || today();
    // Default: find entries where metric > 0.5
    const rows = await duckQuery(
      `SELECT entry_id, entry_date::TEXT AS date
       FROM entries
       WHERE ${col} IS NOT NULL AND entry_date BETWEEN ? AND ?
         AND data_quality = 'clean' AND ${col} > 0.5
       ORDER BY entry_date`,
      [sd, ed]
    );
    return JSON.stringify(
      { metric: metricName, operator: ">", value: 0.5, match_count: rows.length, entries: rows },
      null,
      2
    );
  },

  list_available_metrics: async () => {
    await readyTimeseries();
    const archetypes = await duckQuery(
      `SELECT DISTINCT archetype FROM entry_archetypes ORDER BY archetype`
    );
    const themes = await duckQuery(
      `SELECT concept, COUNT(*) AS freq FROM entry_concepts
       WHERE concept_type = 'theme' GROUP BY concept ORDER BY freq DESC LIMIT 30`
    );
    return JSON.stringify(
      {
        dimensions: [
          "valence", "activation", "agency", "certainty",
          "relational_openness", "self_trust", "time_orientation", "integration",
        ],
        entry_stats: ["word_count"],
        weather: [
          "temperature (or temp, temp_max, temp_min, feels_like)",
          "precipitation (or precip, rain, snow)",
          "cloud_cover (or clouds, cloudiness)",
          "wind (or wind_speed)",
          "daylight_hours (or daylight, sunlight)",
        ],
        archetypes: archetypes.map((r) => `archetype:${r.archetype}`),
        top_themes: themes.map((r) => `theme:${r.concept}`),
      },
      null,
      2
    );
  },

  // get_entry_analysis: skipped — requires entry_id from prior results.
  // search_writings already enriches with analysis data.
};

/**
 * Check if a tool's hard prerequisites are met by the extracted params.
 * Tools that can't possibly produce useful results are skipped at dispatch.
 */
function canDispatch(toolName: string, params: ExtractedParams): boolean {
  switch (toolName) {
    case "correlate_metrics":
      return params.metrics.length >= 2;
    case "compare_periods":
      return params.date_ranges.length >= 2;
    case "get_entries_by_date":
      return params.date_ranges.length >= 1 && !!params.date_ranges[0]?.start;
    default:
      return true;
  }
}

export async function dispatchTools(
  activated: ActivatedTool[],
  params: ExtractedParams,
  query: string,
  signal?: AbortSignal
): Promise<ToolResult[]> {
  const promises = activated
    .filter((tool) => canDispatch(tool.name, params))
    .map(async (tool): Promise<ToolResult> => {
    const fn = DISPATCH_TABLE[tool.name];
    if (!fn) {
      return {
        tool: tool.name,
        result: "",
        composite_score: tool.composite_score,
        duration_ms: 0,
        error: `no dispatch function for ${tool.name}`,
      };
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TOOL_TIMEOUT_MS);

    // Merge the external signal with our per-tool timeout
    let mergedSignal: AbortSignal;
    if (signal) {
      mergedSignal = AbortSignal.any([signal, controller.signal]);
    } else {
      mergedSignal = controller.signal;
    }

    const start = performance.now();
    try {
      const result = await fn(params, query, mergedSignal);
      return {
        tool: tool.name,
        result,
        composite_score: tool.composite_score,
        duration_ms: performance.now() - start,
      };
    } catch (err) {
      return {
        tool: tool.name,
        result: "",
        composite_score: tool.composite_score,
        duration_ms: performance.now() - start,
        error: String(err),
      };
    } finally {
      clearTimeout(timer);
    }
  });

  const settled = await Promise.allSettled(promises);
  return settled.map((r) =>
    r.status === "fulfilled"
      ? r.value
      : {
          tool: "unknown",
          result: "",
          composite_score: 0,
          duration_ms: 0,
          error: String(r.reason),
        }
  );
}
