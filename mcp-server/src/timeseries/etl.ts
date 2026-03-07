import Database from "better-sqlite3";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { ensureSchema } from "./schema.js";
import { query, run, exec } from "./db.js";
import type { Dimension, DataQuality } from "./types.js";
import { DIMENSIONS } from "./types.js";

interface PayloadDimension {
  dimension: string;
  score: number;
}

interface PayloadEntity {
  name: string;
  type: string;
}

interface EntryPayload {
  entry_id: string;
  entry_date: string | null;
  source_file: string | null;
  themes: string[];
  entities: (PayloadEntity | string)[];
  state_profile: {
    dimensions: PayloadDimension[];
  };
  processing?: {
    provider?: string;
    mock?: boolean;
  };
}

export function classifyDataQuality(
  dimScores: Map<string, number>
): DataQuality {
  const allZero = DIMENSIONS.every((d) => {
    const score = dimScores.get(d);
    return score === 0 || score === 0.0;
  });
  return allZero && dimScores.size > 0 ? "suspect_zero" : "clean";
}

function entryIdToDate(entryId: string): string | null {
  // entry_id formats: "1-1-2025", "08-18-2025 10-15 Expression @ Hix"
  const match = entryId.match(/^(\d{1,2})-(\d{1,2})-(\d{4})/);
  if (!match) return null;
  const [, m, d, y] = match;
  return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
}

function countWords(text: string): number {
  return text.split(/\s+/).filter((w) => w.length > 0).length;
}

function buildWordCountMap(corpusPath: string): Map<string, number> {
  const counts = new Map<string, number>();
  let files: string[];
  try {
    files = readdirSync(corpusPath).filter(
      (f) => f.endsWith(".md") && !f.startsWith("._")
    );
  } catch {
    return counts;
  }
  for (const file of files) {
    const stem = file.replace(/\.md$/i, "");
    try {
      const text = readFileSync(join(corpusPath, file), "utf-8");
      counts.set(stem, countWords(text));
    } catch {
      // skip unreadable files
    }
  }
  return counts;
}

export async function backfill(
  sqlitePath: string,
  corpusPath?: string,
  opts: { full?: boolean } = {}
): Promise<{ inserted: number; skipped: number; errors: number; qualityFlagged: number }> {
  await ensureSchema();

  // Load word counts from corpus if available
  const wordCounts = corpusPath ? buildWordCountMap(corpusPath) : new Map();

  // Open analysis SQLite
  const sqlite = new Database(sqlitePath, { readonly: true });
  const rows = sqlite
    .prepare("SELECT entry_id, payload_json FROM entry_summaries")
    .all() as Array<{ entry_id: string; payload_json: string }>;
  sqlite.close();

  // Get existing entry IDs (for incremental)
  let existingIds = new Set<string>();
  if (!opts.full) {
    const existing = await query("SELECT entry_id FROM entries");
    existingIds = new Set(existing.map((r) => r.entry_id as string));
  }

  let inserted = 0;
  let skipped = 0;
  let errors = 0;
  let qualityFlagged = 0;

  for (const row of rows) {
    if (!opts.full && existingIds.has(row.entry_id)) {
      skipped++;
      continue;
    }

    let payload: EntryPayload;
    try {
      payload = JSON.parse(row.payload_json);
    } catch {
      errors++;
      continue;
    }

    // Skip mock entries
    if (payload.processing?.mock) {
      skipped++;
      continue;
    }

    const entryDate =
      payload.entry_date || entryIdToDate(row.entry_id);
    if (!entryDate) {
      errors++;
      continue;
    }

    // Extract dimension scores into a map
    const dimScores = new Map<string, number>();
    if (payload.state_profile?.dimensions) {
      for (const d of payload.state_profile.dimensions) {
        dimScores.set(d.dimension, d.score);
      }
    }

    const wordCount = wordCounts.get(row.entry_id) ?? null;
    const quality = classifyDataQuality(dimScores);
    if (quality === "suspect_zero") qualityFlagged++;

    // Upsert entry
    try {
      await run(
        `INSERT OR REPLACE INTO entries (
          entry_id, entry_date, word_count,
          valence, activation, agency, certainty,
          relational_openness, self_trust, time_orientation, integration,
          data_quality
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          row.entry_id,
          entryDate,
          wordCount,
          ...DIMENSIONS.map((d: Dimension) => dimScores.get(d) ?? null),
          quality,
        ]
      );

      // Insert concepts (themes + entities)
      if (payload.themes) {
        for (const theme of payload.themes) {
          await run(
            `INSERT OR REPLACE INTO entry_concepts (entry_id, concept, concept_type, weight)
             VALUES (?, ?, 'theme', 1.0)`,
            [row.entry_id, theme]
          );
        }
      }
      if (payload.entities) {
        for (const entity of payload.entities) {
          if (typeof entity === "string" && entity.trim()) {
            await run(
              `INSERT OR REPLACE INTO entry_concepts (entry_id, concept, concept_type, weight)
               VALUES (?, ?, 'concept', 1.0)`,
              [row.entry_id, entity.trim()]
            );
          } else if (typeof entity === "object" && entity !== null && (entity as PayloadEntity).name) {
            const e = entity as PayloadEntity;
            await run(
              `INSERT OR REPLACE INTO entry_concepts (entry_id, concept, concept_type, weight)
               VALUES (?, ?, ?, 1.0)`,
              [row.entry_id, e.name, e.type || "concept"]
            );
          }
        }
      }

      inserted++;
    } catch (e) {
      process.stderr.write(
        `Error inserting ${row.entry_id}: ${e}\n`
      );
      errors++;
    }
  }

  return { inserted, skipped, errors, qualityFlagged };
}

export async function incrementalUpdate(
  entryId: string,
  payload: EntryPayload,
  wordCount?: number
): Promise<void> {
  await ensureSchema();

  const entryDate =
    payload.entry_date || entryIdToDate(entryId);
  if (!entryDate) {
    throw new Error(`Cannot determine date for entry: ${entryId}`);
  }

  const dimScores = new Map<string, number>();
  if (payload.state_profile?.dimensions) {
    for (const d of payload.state_profile.dimensions) {
      dimScores.set(d.dimension, d.score);
    }
  }

  const quality = classifyDataQuality(dimScores);

  await run(
    `INSERT OR REPLACE INTO entries (
      entry_id, entry_date, word_count,
      valence, activation, agency, certainty,
      relational_openness, self_trust, time_orientation, integration,
      data_quality
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      entryId,
      entryDate,
      wordCount ?? null,
      ...DIMENSIONS.map((d: Dimension) => dimScores.get(d) ?? null),
      quality,
    ]
  );

  // Clear old concepts for this entry, then re-insert
  await run("DELETE FROM entry_concepts WHERE entry_id = ?", [entryId]);
  if (payload.themes) {
    for (const theme of payload.themes) {
      await run(
        `INSERT INTO entry_concepts (entry_id, concept, concept_type, weight)
         VALUES (?, ?, 'theme', 1.0)`,
        [entryId, theme]
      );
    }
  }
  if (payload.entities) {
    for (const entity of payload.entities) {
      if (typeof entity === "string" && entity.trim()) {
        await run(
          `INSERT INTO entry_concepts (entry_id, concept, concept_type, weight)
           VALUES (?, ?, 'concept', 1.0)`,
          [entryId, entity.trim()]
        );
      } else if (typeof entity === "object" && entity !== null && (entity as PayloadEntity).name) {
        const e = entity as PayloadEntity;
        await run(
          `INSERT INTO entry_concepts (entry_id, concept, concept_type, weight)
           VALUES (?, ?, ?, 1.0)`,
          [entryId, e.name, e.type || "concept"]
        );
      }
    }
  }
}
