import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { query } from "./db.js";
import { ensureSchema } from "./schema.js";
import {
  resolveMetric,
  buildTimeSeriesQuery,
  pearsonPValue,
  interpretCorrelation,
  isValidOperator,
} from "./metrics.js";
import { DIMENSIONS } from "./types.js";
import type { MetricInfo } from "./types.js";

let schemaReady = false;

async function ready(): Promise<void> {
  if (!schemaReady) {
    await ensureSchema();
    schemaReady = true;
  }
}

function defaultStart(): string {
  const d = new Date();
  d.setDate(d.getDate() - 90);
  return d.toISOString().slice(0, 10);
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export function registerTimeSeriesTools(server: McpServer): void {
  // ── 1. query_time_series ──────────────────────────────────────────
  server.tool(
    "query_time_series",
    "Query any metric over a date range at entry, daily, or weekly granularity. Metrics include the 8 psychological dimensions (valence, activation, agency, certainty, relational_openness, self_trust, time_orientation, integration), word_count, or prefixed metrics like 'archetype:Creator', 'theme:cultivating resilience', 'concept:recovery'. Use list_available_metrics to discover all queryable metrics.",
    {
      metric: z
        .string()
        .describe(
          "Metric to query (e.g., 'valence', 'word_count', 'archetype:Creator', 'theme:seeking clarity')"
        ),
      start_date: z
        .string()
        .optional()
        .describe("Start date (YYYY-MM-DD). Defaults to 90 days ago."),
      end_date: z
        .string()
        .optional()
        .describe("End date (YYYY-MM-DD). Defaults to today."),
      granularity: z
        .enum(["entry", "daily", "weekly"])
        .default("daily")
        .describe("Aggregation level: entry (raw), daily, or weekly"),
      include_suspect: z
        .boolean()
        .default(false)
        .describe("Include suspect-quality entries (all dimensions at 0.0). Default: false."),
    },
    async ({ metric, start_date, end_date, granularity, include_suspect }) => {
      try {
        await ready();
        const resolved = resolveMetric(metric);
        if (!resolved) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Unknown metric: "${metric}". Use list_available_metrics to see valid options.`,
              },
            ],
            isError: true,
          };
        }

        const sd = start_date || defaultStart();
        const ed = end_date || today();
        const { sql, params } = buildTimeSeriesQuery(
          resolved,
          sd,
          ed,
          granularity,
          { includeSuspect: include_suspect }
        );
        const rows = await query(sql, params);

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  metric,
                  granularity,
                  start_date: sd,
                  end_date: ed,
                  count: rows.length,
                  data: rows,
                },
                null,
                2
              ),
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // ── 2. detect_anomalies ───────────────────────────────────────────
  server.tool(
    "detect_anomalies",
    "Flag entries where a metric deviates significantly from the person's baseline. Returns anomalous data points with z-scores. Useful for finding unusually high or low psychological states, outlier writing sessions, or sudden shifts.",
    {
      metric: z
        .string()
        .describe("Metric to analyze for anomalies"),
      lookback_days: z
        .number()
        .int()
        .default(180)
        .describe(
          "How far back to compute baseline mean and std (default: 180 days)"
        ),
      threshold: z
        .number()
        .default(2.0)
        .describe(
          "Number of standard deviations for anomaly detection (default: 2.0)"
        ),
    },
    async ({ metric, lookback_days, threshold }) => {
      try {
        await ready();
        const resolved = resolveMetric(metric);
        if (!resolved) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Unknown metric: "${metric}".`,
              },
            ],
            isError: true,
          };
        }

        if (resolved.kind !== "column") {
          return {
            content: [
              {
                type: "text" as const,
                text: `Anomaly detection currently supports column metrics (dimensions and word_count). Got: "${metric}".`,
              },
            ],
            isError: true,
          };
        }

        const col = resolved.column!;
        const cutoff = new Date();
        cutoff.setDate(cutoff.getDate() - lookback_days);
        const cutoffStr = cutoff.toISOString().slice(0, 10);

        // Compute baseline stats (exclude suspect entries)
        const [stats] = await query(
          `SELECT AVG(${col}) AS mean, STDDEV_SAMP(${col}) AS std, COUNT(*) AS n
           FROM entries
           WHERE ${col} IS NOT NULL AND entry_date >= ? AND data_quality = 'clean'`,
          [cutoffStr]
        );

        const mean = stats.mean as number;
        const std = stats.std as number;
        const n = stats.n as number;

        if (!std || std === 0 || n < 3) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Insufficient data for anomaly detection (n=${n}, std=${std}).`,
              },
            ],
          };
        }

        // Find anomalies (exclude suspect entries)
        const anomalies = await query(
          `SELECT entry_id, entry_date::TEXT AS date, ${col} AS value,
                  ABS((${col} - ?) / ?) AS z_score
           FROM entries
           WHERE ${col} IS NOT NULL
             AND entry_date >= ?
             AND data_quality = 'clean'
             AND ABS((${col} - ?) / ?) > ?
           ORDER BY z_score DESC`,
          [mean, std, cutoffStr, mean, std, threshold]
        );

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  metric,
                  baseline: { mean: +mean.toFixed(4), std: +std.toFixed(4), n },
                  threshold,
                  lookback_days,
                  anomaly_count: anomalies.length,
                  anomalies: anomalies.map((a) => ({
                    ...a,
                    baseline_mean: +mean.toFixed(4),
                    baseline_std: +std.toFixed(4),
                    z_score: +(a.z_score as number).toFixed(3),
                  })),
                },
                null,
                2
              ),
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // ── 3. correlate_metrics ──────────────────────────────────────────
  server.tool(
    "correlate_metrics",
    "Discover correlations between any two column metrics (dimensions or word_count) over time. Returns Pearson correlation coefficient, p-value, paired data points, and a human-readable interpretation.",
    {
      metric_a: z.string().describe("First metric"),
      metric_b: z.string().describe("Second metric"),
      start_date: z.string().optional().describe("Start date (YYYY-MM-DD)"),
      end_date: z.string().optional().describe("End date (YYYY-MM-DD)"),
      granularity: z
        .enum(["daily", "weekly"])
        .default("weekly")
        .describe("Aggregation level for correlation"),
    },
    async ({ metric_a, metric_b, start_date, end_date, granularity }) => {
      try {
        await ready();
        const ra = resolveMetric(metric_a);
        const rb = resolveMetric(metric_b);
        if (!ra || !rb) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Unknown metric: ${!ra ? metric_a : metric_b}.`,
              },
            ],
            isError: true,
          };
        }
        if (ra.kind !== "column" || rb.kind !== "column") {
          return {
            content: [
              {
                type: "text" as const,
                text: "Correlation currently supports column metrics only (dimensions and word_count).",
              },
            ],
            isError: true,
          };
        }

        const colA = ra.column!;
        const colB = rb.column!;
        const sd = start_date || defaultStart();
        const ed = end_date || today();

        const dateExpr =
          granularity === "weekly"
            ? "DATE_TRUNC('week', entry_date)::DATE"
            : "entry_date";
        const aggA = colA === "word_count" ? "SUM" : "AVG";
        const aggB = colB === "word_count" ? "SUM" : "AVG";

        const rows = await query(
          `SELECT ${dateExpr}::TEXT AS date,
                  ${aggA}(${colA}) AS value_a,
                  ${aggB}(${colB}) AS value_b
           FROM entries
           WHERE entry_date BETWEEN ? AND ?
             AND ${colA} IS NOT NULL AND ${colB} IS NOT NULL
             AND data_quality = 'clean'
           GROUP BY ${dateExpr}
           ORDER BY date`,
          [sd, ed]
        );

        if (rows.length < 3) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Insufficient paired data points (n=${rows.length}). Need at least 3.`,
              },
            ],
          };
        }

        // Compute correlation using DuckDB (exclude suspect entries)
        const [corrRow] = await query(
          `WITH paired AS (
            SELECT ${dateExpr}::TEXT AS date,
                   ${aggA}(${colA}) AS va,
                   ${aggB}(${colB}) AS vb
            FROM entries
            WHERE entry_date BETWEEN ? AND ?
              AND ${colA} IS NOT NULL AND ${colB} IS NOT NULL
              AND data_quality = 'clean'
            GROUP BY ${dateExpr}
          )
          SELECT CORR(va, vb) AS r, COUNT(*) AS n FROM paired`,
          [sd, ed]
        );

        const r = corrRow.r as number;
        const n = corrRow.n as number;

        if (r == null || isNaN(r)) {
          return {
            content: [
              {
                type: "text" as const,
                text: "Could not compute correlation (constant values or insufficient variance).",
              },
            ],
          };
        }

        const pValue = pearsonPValue(r, n);
        const interpretation = interpretCorrelation(r, pValue);

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  metric_a,
                  metric_b,
                  granularity,
                  pearson_r: +r.toFixed(4),
                  p_value: +pValue.toFixed(6),
                  n,
                  interpretation,
                  data_points: rows.map((row) => ({
                    date: row.date,
                    value_a: +(row.value_a as number).toFixed(4),
                    value_b: +(row.value_b as number).toFixed(4),
                  })),
                },
                null,
                2
              ),
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // ── 4. temporal_filter ────────────────────────────────────────────
  server.tool(
    "temporal_filter",
    "Return entry IDs matching temporal and metric criteria. Use this to find entries that meet specific thresholds, then pipe those IDs into other corpus-intelligence tools (search_writings, get_entry_analysis, etc.). Supports column metrics (dimensions, word_count) with comparison operators.",
    {
      filters: z
        .array(
          z.object({
            metric: z.string().describe("Metric name"),
            operator: z
              .enum([">", ">=", "<", "<=", "=", "!="])
              .describe("Comparison operator"),
            value: z.number().describe("Threshold value"),
          })
        )
        .describe("Array of filter conditions (all must match)"),
      start_date: z.string().optional().describe("Start date (YYYY-MM-DD)"),
      end_date: z.string().optional().describe("End date (YYYY-MM-DD)"),
      include_suspect: z
        .boolean()
        .default(false)
        .describe("Include suspect-quality entries. Default: false."),
    },
    async ({ filters, start_date, end_date, include_suspect }) => {
      try {
        await ready();
        const sd = start_date || "1900-01-01";
        const ed = end_date || today();

        const whereClauses: string[] = [
          "entry_date BETWEEN ? AND ?",
        ];
        const params: unknown[] = [sd, ed];

        if (!include_suspect) {
          whereClauses.push("data_quality = 'clean'");
        }

        for (const f of filters) {
          const resolved = resolveMetric(f.metric);
          if (!resolved || resolved.kind !== "column") {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Unsupported metric for filtering: "${f.metric}". Only column metrics (dimensions, word_count) are supported.`,
                },
              ],
              isError: true,
            };
          }
          if (!isValidOperator(f.operator)) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Invalid operator: "${f.operator}".`,
                },
              ],
              isError: true,
            };
          }
          whereClauses.push(`${resolved.column!} ${f.operator} ?`);
          params.push(f.value);
        }

        const rows = await query(
          `SELECT entry_id, entry_date::TEXT AS date
           FROM entries
           WHERE ${whereClauses.join(" AND ")}
           ORDER BY entry_date`,
          params
        );

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  filters,
                  start_date: sd,
                  end_date: ed,
                  match_count: rows.length,
                  entries: rows,
                },
                null,
                2
              ),
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // ── 5. get_metric_summary ─────────────────────────────────────────
  server.tool(
    "get_metric_summary",
    "Quick summary statistics for a metric over a period: mean, median, standard deviation, min, max, entry count, and current trend (rising/falling/stable based on recent values vs overall mean).",
    {
      metric: z.string().describe("Metric to summarize"),
      start_date: z.string().optional().describe("Start date (YYYY-MM-DD)"),
      end_date: z.string().optional().describe("End date (YYYY-MM-DD)"),
    },
    async ({ metric, start_date, end_date }) => {
      try {
        await ready();
        const resolved = resolveMetric(metric);
        if (!resolved || resolved.kind !== "column") {
          return {
            content: [
              {
                type: "text" as const,
                text: `Unsupported metric for summary: "${metric}". Only column metrics (dimensions, word_count) are supported.`,
              },
            ],
            isError: true,
          };
        }

        const col = resolved.column!;
        const sd = start_date || "1900-01-01";
        const ed = end_date || today();

        const [stats] = await query(
          `SELECT
             AVG(${col}) AS mean,
             MEDIAN(${col}) AS median,
             STDDEV_SAMP(${col}) AS std,
             MIN(${col}) AS min,
             MAX(${col}) AS max,
             COUNT(*) AS entries_count
           FROM entries
           WHERE ${col} IS NOT NULL AND entry_date BETWEEN ? AND ? AND data_quality = 'clean'`,
          [sd, ed]
        );

        if (!stats || (stats.entries_count as number) === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: `No data for metric "${metric}" in the specified range.`,
              },
            ],
          };
        }

        // Determine trend: compare last 30 days mean to overall mean
        const recentCutoff = new Date();
        recentCutoff.setDate(recentCutoff.getDate() - 30);
        const recentStr = recentCutoff.toISOString().slice(0, 10);

        const [recent] = await query(
          `SELECT AVG(${col}) AS recent_mean
           FROM entries
           WHERE ${col} IS NOT NULL AND entry_date >= ? AND entry_date <= ? AND data_quality = 'clean'`,
          [recentStr, ed]
        );

        let currentTrend: "rising" | "falling" | "stable" = "stable";
        if (recent?.recent_mean != null && stats.std) {
          const diff =
            (recent.recent_mean as number) - (stats.mean as number);
          const halfStd = (stats.std as number) * 0.5;
          if (diff > halfStd) currentTrend = "rising";
          else if (diff < -halfStd) currentTrend = "falling";
        }

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  metric,
                  start_date: sd,
                  end_date: ed,
                  mean: +(stats.mean as number).toFixed(4),
                  median: +(stats.median as number).toFixed(4),
                  std: +((stats.std as number) || 0).toFixed(4),
                  min: stats.min,
                  max: stats.max,
                  entries_count: stats.entries_count,
                  current_trend: currentTrend,
                },
                null,
                2
              ),
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // ── 6. list_available_metrics ─────────────────────────────────────
  server.tool(
    "list_available_metrics",
    "List all queryable metric names and types. Returns dimension metrics, entry stats, and any archetype/concept metrics discovered in the data. Use this to find out what you can query with query_time_series, detect_anomalies, correlate_metrics, etc.",
    {},
    async () => {
      try {
        await ready();

        const metrics: MetricInfo[] = [];

        // Dimension metrics
        const dimDescriptions: Record<string, string> = {
          valence: "Emotional tone (-1 heavy/+1 uplifted)",
          activation: "Energy level (-1 calm/+1 activated)",
          agency: "Sense of control (-1 stuck/+1 empowered)",
          certainty: "Clarity of mind (-1 conflicted/+1 resolved)",
          relational_openness:
            "Interpersonal openness (-1 guarded/+1 open)",
          self_trust: "Self-confidence (-1 doubt/+1 trust)",
          time_orientation:
            "Temporal focus (-1 past-looping/+1 future-building)",
          integration:
            "Inner coherence (-1 fragmented/+1 coherent)",
        };

        for (const dim of DIMENSIONS) {
          metrics.push({
            metric_name: dim,
            metric_type: "dimension",
            description: dimDescriptions[dim] || dim,
          });
        }

        // Entry stats
        metrics.push({
          metric_name: "word_count",
          metric_type: "entry_stat",
          description: "Word count of the writing entry",
        });

        // Dynamic: archetypes in data
        const archetypes = await query(
          `SELECT DISTINCT archetype FROM entry_archetypes ORDER BY archetype`
        );
        for (const row of archetypes) {
          metrics.push({
            metric_name: `archetype:${row.archetype}`,
            metric_type: "archetype",
            description: `Archetype strength for ${row.archetype}`,
          });
        }

        // Dynamic: top themes
        const themes = await query(
          `SELECT concept, COUNT(*) AS freq
           FROM entry_concepts
           WHERE concept_type = 'theme'
           GROUP BY concept
           ORDER BY freq DESC
           LIMIT 50`
        );
        for (const row of themes) {
          metrics.push({
            metric_name: `theme:${row.concept}`,
            metric_type: "concept",
            description: `Theme frequency for "${row.concept}" (${row.freq} entries)`,
          });
        }

        // Dynamic: top entity types
        const entityTypes = await query(
          `SELECT concept_type, COUNT(DISTINCT concept) AS n
           FROM entry_concepts
           WHERE concept_type != 'theme'
           GROUP BY concept_type
           ORDER BY n DESC`
        );
        for (const row of entityTypes) {
          metrics.push({
            metric_name: `concept:<name> (type: ${row.concept_type})`,
            metric_type: "concept",
            description: `${row.n} unique ${row.concept_type} entities tracked. Query specific ones with 'concept:<name>'.`,
          });
        }

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                { total: metrics.length, metrics },
                null,
                2
              ),
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );
}
