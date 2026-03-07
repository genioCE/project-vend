import Database from "better-sqlite3";
import { readFileSync, existsSync } from "fs";
import { join, resolve } from "path";
import { createProvider } from "./reprocess/providers.js";
import { PROMPT_VERSION } from "./reprocess/prompts.js";
import type { StateDimension } from "./reprocess/providers.js";

// ── Load .env from project root ──────────────────────────────────────

function loadEnv(envPath: string): void {
  if (!existsSync(envPath)) return;
  const lines = readFileSync(envPath, "utf-8").split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx < 0) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    const value = trimmed.slice(eqIdx + 1).trim();
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

loadEnv(resolve(__dirname, "../../.env"));

// ── CLI args ────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const suspectOnly = args.includes("--suspect-only");
const skipDone = args.includes("--skip-done");
const limitIdx = args.indexOf("--limit");
const limit = limitIdx >= 0 ? parseInt(args[limitIdx + 1], 10) : 0;
const concurrencyIdx = args.indexOf("--concurrency");
const concurrency = concurrencyIdx >= 0 ? parseInt(args[concurrencyIdx + 1], 10) : 5;
const sqlitePathIdx = args.indexOf("--sqlite-path");
const sqlitePath = sqlitePathIdx >= 0
  ? args[sqlitePathIdx + 1]
  : process.env.ANALYSIS_SQLITE_PATH ?? "../analysis.sqlite";
const corpusPathIdx = args.indexOf("--corpus-path");
const corpusPath = corpusPathIdx >= 0
  ? args[corpusPathIdx + 1]
  : process.env.CORPUS_PATH ?? "../data/2025";
const providerIdx = args.indexOf("--provider");
const providerName = providerIdx >= 0 ? args[providerIdx + 1] : undefined;

// ── Get entry IDs to reprocess ──────────────────────────────────────

async function getEntryIds(): Promise<string[]> {
  if (suspectOnly) {
    const duckdb = await import("duckdb");
    const duckdbPath = process.env.DUCKDB_PATH ?? "./data/timeseries.duckdb";
    return new Promise<string[]>((resolve, reject) => {
      const db = new duckdb.default.Database(duckdbPath);
      const cb = (err: Error | null, rows: Record<string, unknown>[]) => {
        db.close();
        if (err) reject(err);
        else resolve(rows.map((r) => r.entry_id as string));
      };
      (db.all as Function).call(
        db,
        "SELECT entry_id FROM entries WHERE data_quality = 'suspect_zero' ORDER BY entry_date",
        cb
      );
    });
  }

  const sqlite = new Database(sqlitePath, { readonly: true });
  const rows = sqlite
    .prepare("SELECT entry_id FROM entry_summaries ORDER BY entry_id")
    .all() as Array<{ entry_id: string }>;
  sqlite.close();
  return rows.map((r) => r.entry_id);
}

// ── Check if entry was already reprocessed ───────────────────────────

function isAlreadyReprocessed(
  sqlite: Database.Database,
  entryId: string
): boolean {
  const row = sqlite
    .prepare("SELECT payload_json FROM entry_summaries WHERE entry_id = ?")
    .get(entryId) as { payload_json: string } | undefined;
  if (!row) return false;
  try {
    const payload = JSON.parse(row.payload_json);
    return payload.processing?.state_prompt_version === PROMPT_VERSION;
  } catch {
    return false;
  }
}

// ── Find corpus file for an entry ID ────────────────────────────────

function findCorpusFile(entryId: string): string | null {
  const candidates = [
    join(corpusPath, `${entryId}.md`),
    join(corpusPath, `${entryId}.MD`),
  ];
  for (const path of candidates) {
    if (existsSync(path)) return path;
  }
  return null;
}

// ── Update analysis.sqlite with new state profile ───────────────────

function updateStateProfile(
  sqlite: Database.Database,
  entryId: string,
  dimensions: StateDimension[],
  providerUsed: string
): void {
  const row = sqlite
    .prepare("SELECT payload_json FROM entry_summaries WHERE entry_id = ?")
    .get(entryId) as { payload_json: string } | undefined;

  if (!row) {
    process.stderr.write(`  No existing record for ${entryId}, skipping update\n`);
    return;
  }

  const payload = JSON.parse(row.payload_json);

  // Replace state_profile dimensions
  if (!payload.state_profile) {
    payload.state_profile = { score_range: { min: -1.0, max: 1.0 }, dimensions: [] };
  }
  payload.state_profile.dimensions = dimensions.map((d) => ({
    dimension: d.dimension,
    score: d.score,
    low_anchor: getAnchor(d.dimension, "low"),
    high_anchor: getAnchor(d.dimension, "high"),
    label: d.label,
    evidence_spans: [],
  }));

  // Update processing metadata
  if (!payload.processing) payload.processing = {};
  payload.processing.state_provider = providerUsed;
  payload.processing.state_prompt_version = PROMPT_VERSION;
  payload.processing.state_reprocessed_at = new Date().toISOString();

  const updatedJson = JSON.stringify(payload);
  sqlite
    .prepare(
      "UPDATE entry_summaries SET payload_json = ?, updated_at = ? WHERE entry_id = ?"
    )
    .run(updatedJson, new Date().toISOString(), entryId);
}

const ANCHORS: Record<string, [string, string]> = {
  valence: ["heavy", "uplifted"],
  activation: ["calm", "activated"],
  agency: ["stuck", "empowered"],
  certainty: ["conflicted", "resolved"],
  relational_openness: ["guarded", "open"],
  self_trust: ["doubt", "trust"],
  time_orientation: ["past_looping", "future_building"],
  integration: ["fragmented", "coherent"],
};

function getAnchor(dim: string, end: "low" | "high"): string {
  const a = ANCHORS[dim];
  return a ? a[end === "low" ? 0 : 1] : end;
}

// ── Concurrency pool ─────────────────────────────────────────────────

async function runWithConcurrency<T>(
  items: T[],
  maxConcurrency: number,
  fn: (item: T) => Promise<void>
): Promise<void> {
  let index = 0;
  const workers = Array.from({ length: Math.min(maxConcurrency, items.length) }, async () => {
    while (index < items.length) {
      const i = index++;
      await fn(items[i]);
    }
  });
  await Promise.all(workers);
}

// ── Main ────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const provider = createProvider(providerName);
  console.log(`Provider: ${provider.name}`);
  console.log(`SQLite: ${sqlitePath}`);
  console.log(`Corpus: ${corpusPath}`);
  console.log(`Mode: ${suspectOnly ? "suspect_zero only" : "all entries"}`);
  console.log(`Concurrency: ${concurrency}`);
  if (skipDone) console.log("Skipping already-reprocessed entries");
  if (dryRun) console.log("DRY RUN — no API calls, no writes");
  if (limit) console.log(`Limit: ${limit} entries`);
  console.log();

  let entryIds = await getEntryIds();
  if (limit) entryIds = entryIds.slice(0, limit);

  console.log(`Found ${entryIds.length} entries to reprocess\n`);

  // Check corpus file availability
  let missingFiles = 0;
  for (const id of entryIds) {
    if (!findCorpusFile(id)) missingFiles++;
  }
  if (missingFiles > 0) {
    console.log(`Warning: ${missingFiles} entries have no corpus .md file\n`);
  }

  if (dryRun) {
    for (const id of entryIds) {
      const file = findCorpusFile(id);
      const status = file ? "OK" : "MISSING";
      console.log(`  ${id}: ${status}`);
    }
    console.log(`\nDry run complete. ${entryIds.length} entries would be reprocessed.`);
    return;
  }

  // Open analysis.sqlite for writing
  const sqlite = new Database(sqlitePath);

  // Filter out already-reprocessed entries if --skip-done
  if (skipDone) {
    const before = entryIds.length;
    entryIds = entryIds.filter((id) => !isAlreadyReprocessed(sqlite, id));
    const skippedDone = before - entryIds.length;
    if (skippedDone > 0) {
      console.log(`Skipped ${skippedDone} already-reprocessed entries\n`);
    }
  }

  // Build work items (entries with corpus files)
  const workItems: Array<{ entryId: string; filePath: string }> = [];
  let skipped = 0;
  for (const entryId of entryIds) {
    const filePath = findCorpusFile(entryId);
    if (!filePath) {
      skipped++;
      continue;
    }
    workItems.push({ entryId, filePath });
  }

  const total = workItems.length;
  let processed = 0;
  let errors = 0;

  await runWithConcurrency(workItems, concurrency, async ({ entryId, filePath }) => {
    try {
      const text = readFileSync(filePath, "utf-8");
      const dimensions = await provider.generate(entryId, text);

      // SQLite writes are serialized (better-sqlite3 is synchronous)
      updateStateProfile(sqlite, entryId, dimensions, provider.name);
      processed++;

      const scores = dimensions
        .map((d) => `${d.dimension.slice(0, 3)}=${d.score.toFixed(2)}`)
        .join(" ");
      console.log(`[${processed + errors}/${total}] ${entryId} ✓ ${scores}`);
    } catch (e) {
      errors++;
      process.stderr.write(`[${processed + errors}/${total}] ${entryId} ✗ ${e}\n`);
    }
  });

  sqlite.close();

  console.log(
    `\nDone: ${processed} reprocessed, ${errors} errors, ${skipped} skipped (no file)`
  );
  console.log(
    `\nNext: run 'npm run backfill -- ../analysis.sqlite ../data/2025 --full' to refresh DuckDB`
  );
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
