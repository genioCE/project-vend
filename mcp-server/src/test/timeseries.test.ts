import test from "node:test";
import assert from "node:assert/strict";
import duckdb from "duckdb";
import {
  resolveMetric,
  buildTimeSeriesQuery,
  pearsonPValue,
  interpretCorrelation,
  isValidOperator,
} from "../timeseries/metrics.js";
import { classifyDataQuality } from "../timeseries/etl.js";
import { DIMENSIONS } from "../timeseries/types.js";

// ── Metric resolution ───────────────────────────────────────────────

test("resolveMetric resolves all 8 dimensions", () => {
  for (const dim of DIMENSIONS) {
    const r = resolveMetric(dim);
    assert.ok(r, `should resolve ${dim}`);
    assert.equal(r!.kind, "column");
    assert.equal(r!.column, dim);
  }
});

test("resolveMetric resolves word_count", () => {
  const r = resolveMetric("word_count");
  assert.ok(r);
  assert.equal(r!.kind, "column");
  assert.equal(r!.column, "word_count");
});

test("resolveMetric resolves archetype: prefix", () => {
  const r = resolveMetric("archetype:Creator");
  assert.ok(r);
  assert.equal(r!.kind, "archetype");
  assert.equal(r!.filterValue, "Creator");
});

test("resolveMetric resolves theme: prefix", () => {
  const r = resolveMetric("theme:cultivating resilience");
  assert.ok(r);
  assert.equal(r!.kind, "concept");
  assert.equal(r!.filterValue, "cultivating resilience");
  assert.equal(r!.conceptType, "theme");
});

test("resolveMetric resolves concept: prefix", () => {
  const r = resolveMetric("concept:recovery");
  assert.ok(r);
  assert.equal(r!.kind, "concept");
  assert.equal(r!.filterValue, "recovery");
  assert.equal(r!.conceptType, undefined);
});

test("resolveMetric returns null for unknown metric", () => {
  assert.equal(resolveMetric("nonexistent"), null);
  assert.equal(resolveMetric(""), null);
});

// ── Data quality classification ─────────────────────────────────────

test("classifyDataQuality returns suspect_zero when all dims are 0", () => {
  const scores = new Map<string, number>();
  for (const d of DIMENSIONS) scores.set(d, 0.0);
  assert.equal(classifyDataQuality(scores), "suspect_zero");
});

test("classifyDataQuality returns clean when any dim is non-zero", () => {
  const scores = new Map<string, number>();
  for (const d of DIMENSIONS) scores.set(d, 0.0);
  scores.set("valence", 0.5);
  assert.equal(classifyDataQuality(scores), "clean");
});

test("classifyDataQuality returns clean for empty map", () => {
  assert.equal(classifyDataQuality(new Map()), "clean");
});

// ── SQL query building ──────────────────────────────────────────────

test("buildTimeSeriesQuery column entry granularity", () => {
  const metric = resolveMetric("valence")!;
  const { sql, params } = buildTimeSeriesQuery(
    metric,
    "2025-01-01",
    "2025-03-01",
    "entry"
  );
  assert.ok(sql.includes("entry_id"));
  assert.ok(sql.includes("valence AS value"));
  assert.ok(!sql.includes("GROUP BY"));
  assert.deepEqual(params, ["2025-01-01", "2025-03-01"]);
});

test("buildTimeSeriesQuery includes quality filter by default", () => {
  const metric = resolveMetric("valence")!;
  const { sql } = buildTimeSeriesQuery(
    metric,
    "2025-01-01",
    "2025-03-01",
    "entry"
  );
  assert.ok(sql.includes("data_quality = 'clean'"));
});

test("buildTimeSeriesQuery omits quality filter when includeSuspect", () => {
  const metric = resolveMetric("valence")!;
  const { sql } = buildTimeSeriesQuery(
    metric,
    "2025-01-01",
    "2025-03-01",
    "entry",
    { includeSuspect: true }
  );
  assert.ok(!sql.includes("data_quality"));
});

test("buildTimeSeriesQuery column daily granularity uses AVG", () => {
  const metric = resolveMetric("agency")!;
  const { sql } = buildTimeSeriesQuery(
    metric,
    "2025-01-01",
    "2025-03-01",
    "daily"
  );
  assert.ok(sql.includes("AVG("));
  assert.ok(sql.includes("GROUP BY"));
});

test("buildTimeSeriesQuery word_count daily uses SUM", () => {
  const metric = resolveMetric("word_count")!;
  const { sql } = buildTimeSeriesQuery(
    metric,
    "2025-01-01",
    "2025-03-01",
    "daily"
  );
  assert.ok(sql.includes("SUM("));
});

test("buildTimeSeriesQuery weekly uses DATE_TRUNC", () => {
  const metric = resolveMetric("valence")!;
  const { sql } = buildTimeSeriesQuery(
    metric,
    "2025-01-01",
    "2025-03-01",
    "weekly"
  );
  assert.ok(sql.includes("DATE_TRUNC"));
});

test("buildTimeSeriesQuery archetype entry granularity", () => {
  const metric = resolveMetric("archetype:Creator")!;
  const { sql, params } = buildTimeSeriesQuery(
    metric,
    "2025-01-01",
    "2025-03-01",
    "entry"
  );
  assert.ok(sql.includes("entry_archetypes"));
  assert.ok(sql.includes("strength AS value"));
  assert.equal(params[0], "Creator");
});

test("buildTimeSeriesQuery concept daily uses COUNT", () => {
  const metric = resolveMetric("theme:seeking clarity")!;
  const { sql, params } = buildTimeSeriesQuery(
    metric,
    "2025-01-01",
    "2025-03-01",
    "daily"
  );
  assert.ok(sql.includes("COUNT(*)"));
  assert.ok(sql.includes("concept_type = ?"));
  assert.equal(params[0], "seeking clarity");
  assert.equal(params[1], "theme");
});

// ── Operator validation ─────────────────────────────────────────────

test("isValidOperator accepts valid operators", () => {
  for (const op of [">", ">=", "<", "<=", "=", "!="]) {
    assert.ok(isValidOperator(op), `should accept ${op}`);
  }
});

test("isValidOperator rejects invalid operators", () => {
  assert.ok(!isValidOperator("LIKE"));
  assert.ok(!isValidOperator("OR 1=1"));
  assert.ok(!isValidOperator(""));
});

// ── Statistics ───────────────────────────────────────────────────────

test("pearsonPValue returns 1.0 for n <= 2", () => {
  assert.equal(pearsonPValue(0.5, 2), 1.0);
  assert.equal(pearsonPValue(0.5, 1), 1.0);
});

test("pearsonPValue returns 0.0 for r = 1", () => {
  assert.equal(pearsonPValue(1.0, 100), 0.0);
});

test("pearsonPValue returns small p for high r, large n", () => {
  const p = pearsonPValue(0.9, 100);
  assert.ok(p < 0.001, `Expected p < 0.001, got ${p}`);
});

test("pearsonPValue returns large p for low r", () => {
  const p = pearsonPValue(0.05, 30);
  assert.ok(p > 0.5, `Expected p > 0.5, got ${p}`);
});

test("interpretCorrelation produces readable output", () => {
  const msg = interpretCorrelation(0.72, 0.001);
  assert.ok(msg.includes("Very strong"), `expected 'Very strong' in: ${msg}`);
  assert.ok(msg.includes("positive"));
  assert.ok(msg.includes("0.720"));
});

test("interpretCorrelation handles negative correlation", () => {
  const msg = interpretCorrelation(-0.45, 0.02);
  assert.ok(msg.includes("Moderate"));
  assert.ok(msg.includes("negative"));
  assert.ok(msg.includes("inversely"));
});

// ── DuckDB integration (in-memory) ─────────────────────────────────

function dbAll(
  db: duckdb.Database,
  sql: string,
  params: unknown[] = []
): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    const cb = (err: Error | null, rows: Record<string, unknown>[]) => {
      if (err) reject(err);
      else resolve(rows ?? []);
    };
    const args: unknown[] = [sql, ...params, cb];
    (db.all as Function).apply(db, args);
  });
}

function dbExec(db: duckdb.Database, sql: string): Promise<void> {
  return new Promise((resolve, reject) => {
    db.exec(sql, (err: Error | null) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

test("DuckDB schema + insert + query round-trip", async () => {
  const db = new duckdb.Database(":memory:");

  await dbExec(db, `
    CREATE TABLE entries (
      entry_id TEXT PRIMARY KEY,
      entry_date DATE NOT NULL,
      word_count INTEGER,
      valence DOUBLE,
      activation DOUBLE,
      agency DOUBLE,
      certainty DOUBLE,
      relational_openness DOUBLE,
      self_trust DOUBLE,
      time_orientation DOUBLE,
      integration DOUBLE,
      data_quality TEXT NOT NULL DEFAULT 'clean'
    );
    CREATE TABLE entry_concepts (
      entry_id TEXT NOT NULL,
      concept TEXT NOT NULL,
      concept_type TEXT NOT NULL,
      weight DOUBLE DEFAULT 1.0,
      PRIMARY KEY (entry_id, concept, concept_type)
    );
  `);

  await dbExec(db, `
    INSERT INTO entries VALUES
      ('1-1-2025', '2025-01-01', 1500, 0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.1, 0.8, 'clean'),
      ('1-2-2025', '2025-01-02', 2000, -0.3, 0.8, 0.1, -0.5, 0.2, -0.1, 0.4, 0.3, 'clean'),
      ('1-3-2025', '2025-01-03', 800, 0.9, -0.2, 0.5, 0.8, 0.7, 0.9, 0.6, 0.95, 'clean'),
      ('1-4-2025', '2025-01-04', 3000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'suspect_zero');
    INSERT INTO entry_concepts VALUES
      ('1-1-2025', 'cultivating resilience', 'theme', 1.0),
      ('1-1-2025', 'John', 'person', 1.0),
      ('1-2-2025', 'cultivating resilience', 'theme', 1.0),
      ('1-3-2025', 'seeking clarity', 'theme', 1.0);
  `);

  // Test basic select (DuckDB COUNT returns BigInt)
  const rows = await dbAll(db, "SELECT COUNT(*) AS n FROM entries");
  assert.equal(Number(rows[0].n), 4);

  // Test AVG aggregation (all 4 entries including suspect)
  const [avgRow] = await dbAll(
    db,
    "SELECT AVG(valence) AS avg_val FROM entries"
  );
  const avgVal = avgRow.avg_val as number;
  assert.ok(
    Math.abs(avgVal - (0.5 + -0.3 + 0.9 + 0.0) / 4) < 0.001,
    `avg should be ~0.275, got ${avgVal}`
  );

  // Test correlation
  const [corrRow] = await dbAll(
    db,
    "SELECT CORR(valence, integration) AS r FROM entries"
  );
  assert.ok(corrRow.r != null, "correlation should be computed");

  // Test date filtering
  const filtered = await dbAll(
    db,
    "SELECT entry_id FROM entries WHERE entry_date BETWEEN ? AND ?",
    ["2025-01-01", "2025-01-02"]
  );
  assert.equal(filtered.length, 2);

  // Test concept join
  const themes = await dbAll(
    db,
    `SELECT e.entry_date::TEXT AS date, COUNT(*)::INTEGER AS freq
     FROM entries e JOIN entry_concepts c ON e.entry_id = c.entry_id
     WHERE c.concept = ? AND c.concept_type = 'theme'
     GROUP BY e.entry_date ORDER BY date`,
    ["cultivating resilience"]
  );
  assert.equal(themes.length, 2);

  // Test weekly aggregation
  const weekly = await dbAll(
    db,
    `SELECT DATE_TRUNC('week', entry_date)::DATE::TEXT AS week, AVG(valence) AS avg
     FROM entries GROUP BY DATE_TRUNC('week', entry_date)`
  );
  assert.ok(weekly.length >= 1);

  // Test data_quality filter excludes suspect entries
  const cleanOnly = await dbAll(
    db,
    "SELECT COUNT(*) AS n FROM entries WHERE data_quality = 'clean'"
  );
  assert.equal(Number(cleanOnly[0].n), 3);

  // Test suspect entries are excluded from AVG when filtered
  const [cleanAvg] = await dbAll(
    db,
    "SELECT AVG(valence) AS avg_val FROM entries WHERE data_quality = 'clean'"
  );
  const cleanAvgVal = cleanAvg.avg_val as number;
  // Without suspect: (0.5 + -0.3 + 0.9) / 3 = 0.367
  assert.ok(
    Math.abs(cleanAvgVal - (0.5 + -0.3 + 0.9) / 3) < 0.001,
    `clean avg should be ~0.367, got ${cleanAvgVal}`
  );

  db.close();
});

test("DuckDB STDDEV_SAMP and MEDIAN work", async () => {
  const db = new duckdb.Database(":memory:");
  await dbExec(db, `
    CREATE TABLE t (v DOUBLE);
    INSERT INTO t VALUES (1), (2), (3), (4), (5);
  `);

  const [row] = await dbAll(
    db,
    "SELECT STDDEV_SAMP(v) AS std, MEDIAN(v) AS med FROM t"
  );
  const std = row.std as number;
  const med = row.med as number;
  assert.ok(Math.abs(std - 1.5811) < 0.01, `std should be ~1.58, got ${std}`);
  assert.equal(med, 3);

  db.close();
});
