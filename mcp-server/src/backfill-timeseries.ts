import { backfill } from "./timeseries/etl.js";
import { close } from "./timeseries/db.js";

const args = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const full = process.argv.includes("--full");

const sqlitePath =
  args[0] ||
  process.env.ANALYSIS_SQLITE_PATH ||
  "../analysis.sqlite";
const corpusPath = args[1] || process.env.CORPUS_PATH || undefined;

console.log(`Backfilling DuckDB from: ${sqlitePath}`);
if (corpusPath) console.log(`Corpus path (word counts): ${corpusPath}`);
if (full) console.log("Mode: full (replacing all entries)");
else console.log("Mode: incremental (skipping existing)");

backfill(sqlitePath, corpusPath, { full })
  .then(async (result) => {
    console.log(
      `Done: ${result.inserted} inserted, ${result.skipped} skipped, ${result.errors} errors, ${result.qualityFlagged} flagged as suspect_zero`
    );
    await close();
    process.exit(0);
  })
  .catch(async (err) => {
    console.error("Backfill failed:", err);
    await close();
    process.exit(1);
  });
