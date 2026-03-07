import { DIMENSIONS, type Dimension } from "./types.js";

export type MetricKind = "column" | "archetype" | "concept";

export interface ResolvedMetric {
  kind: MetricKind;
  column?: string;
  filterValue?: string;
  conceptType?: string;
}

const VALID_OPERATORS = new Set([">", ">=", "<", "<=", "=", "!="]);

export function resolveMetric(name: string): ResolvedMetric | null {
  // Normalize for dimension matching: hyphens → underscores, lowercase, trim
  const normalized = name.toLowerCase().trim().replace(/-/g, "_");

  if ((DIMENSIONS as readonly string[]).includes(normalized)) {
    return { kind: "column", column: normalized };
  }
  if (normalized === "word_count") {
    return { kind: "column", column: "word_count" };
  }

  // For prefixed values (archetype:X, theme:X, concept:X), preserve original case
  // but still normalize the prefix for matching
  const trimmed = name.trim();
  const prefixed = trimmed.match(/^(archetype|theme|concept):(.+)$/i);
  if (prefixed) {
    const [, prefix, value] = prefixed;
    const normalizedPrefix = prefix.toLowerCase();
    if (normalizedPrefix === "archetype") {
      return { kind: "archetype", filterValue: value };
    }
    return {
      kind: "concept",
      filterValue: value,
      conceptType: normalizedPrefix === "theme" ? "theme" : undefined,
    };
  }
  return null;
}

export function isValidOperator(op: string): boolean {
  return VALID_OPERATORS.has(op);
}

export function buildTimeSeriesQuery(
  metric: ResolvedMetric,
  startDate: string,
  endDate: string,
  granularity: "entry" | "daily" | "weekly",
  options?: { includeSuspect?: boolean }
): { sql: string; params: unknown[] } {
  const dateExpr =
    granularity === "weekly"
      ? "DATE_TRUNC('week', e.entry_date)::DATE"
      : "e.entry_date";

  const qualityFilter = options?.includeSuspect
    ? ""
    : " AND e.data_quality = 'clean'";

  if (metric.kind === "column") {
    const col = metric.column!;
    if (granularity === "entry") {
      return {
        sql: `SELECT e.entry_id, e.entry_date::TEXT AS date, e.${col} AS value
              FROM entries e
              WHERE e.entry_date BETWEEN ? AND ? AND e.${col} IS NOT NULL${qualityFilter}
              ORDER BY e.entry_date`,
        params: [startDate, endDate],
      };
    }
    const agg = col === "word_count" ? "SUM" : "AVG";
    return {
      sql: `SELECT ${dateExpr}::TEXT AS date, ${agg}(e.${col}) AS value
            FROM entries e
            WHERE e.entry_date BETWEEN ? AND ? AND e.${col} IS NOT NULL${qualityFilter}
            GROUP BY ${dateExpr}
            ORDER BY date`,
      params: [startDate, endDate],
    };
  }

  if (metric.kind === "archetype") {
    if (granularity === "entry") {
      return {
        sql: `SELECT e.entry_id, e.entry_date::TEXT AS date, a.strength AS value
              FROM entries e
              JOIN entry_archetypes a ON e.entry_id = a.entry_id
              WHERE a.archetype = ? AND e.entry_date BETWEEN ? AND ?${qualityFilter}
              ORDER BY e.entry_date`,
        params: [metric.filterValue!, startDate, endDate],
      };
    }
    return {
      sql: `SELECT ${dateExpr}::TEXT AS date, AVG(a.strength) AS value
            FROM entries e
            JOIN entry_archetypes a ON e.entry_id = a.entry_id
            WHERE a.archetype = ? AND e.entry_date BETWEEN ? AND ?${qualityFilter}
            GROUP BY ${dateExpr}
            ORDER BY date`,
      params: [metric.filterValue!, startDate, endDate],
    };
  }

  // concept/theme
  const typeFilter = metric.conceptType ? " AND c.concept_type = ?" : "";
  const typeParams = metric.conceptType ? [metric.conceptType] : [];

  if (granularity === "entry") {
    return {
      sql: `SELECT e.entry_id, e.entry_date::TEXT AS date, c.weight AS value
            FROM entries e
            JOIN entry_concepts c ON e.entry_id = c.entry_id
            WHERE c.concept = ?${typeFilter} AND e.entry_date BETWEEN ? AND ?${qualityFilter}
            ORDER BY e.entry_date`,
      params: [metric.filterValue!, ...typeParams, startDate, endDate],
    };
  }
  return {
    sql: `SELECT ${dateExpr}::TEXT AS date, COUNT(*)::DOUBLE AS value
          FROM entries e
          JOIN entry_concepts c ON e.entry_id = c.entry_id
          WHERE c.concept = ?${typeFilter} AND e.entry_date BETWEEN ? AND ?${qualityFilter}
          GROUP BY ${dateExpr}
          ORDER BY date`,
    params: [metric.filterValue!, ...typeParams, startDate, endDate],
  };
}

/** Normal CDF approximation (Abramowitz & Stegun) */
function normalCDF(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1.0 / (1.0 + p * ax);
  const y =
    1.0 -
    ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) *
      t *
      Math.exp(-ax * ax);
  return 0.5 * (1.0 + sign * y);
}

/** Approximate two-tailed p-value for Pearson correlation */
export function pearsonPValue(r: number, n: number): number {
  if (n <= 2 || Math.abs(r) >= 1) return n <= 2 ? 1.0 : 0.0;
  const t = (r * Math.sqrt(n - 2)) / Math.sqrt(1 - r * r);
  return Math.max(0, Math.min(1, 2 * (1 - normalCDF(Math.abs(t)))));
}

export function interpretCorrelation(r: number, p: number): string {
  const absR = Math.abs(r);
  const direction = r > 0 ? "positive" : "negative";
  const sig = p < 0.01 ? "p<0.01" : p < 0.05 ? "p<0.05" : `p=${p.toFixed(3)}`;

  let strength: string;
  if (absR < 0.1) strength = "Negligible";
  else if (absR < 0.3) strength = "Weak";
  else if (absR < 0.5) strength = "Moderate";
  else if (absR < 0.7) strength = "Strong";
  else strength = "Very strong";

  const rising = r > 0 ? "rise together" : "move inversely";
  return `${strength} ${direction} correlation (r=${r.toFixed(3)}, ${sig}) — the two metrics tend to ${rising}.`;
}
