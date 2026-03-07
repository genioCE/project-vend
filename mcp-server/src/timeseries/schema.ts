import { exec } from "./db.js";

export async function ensureSchema(): Promise<void> {
  await exec(`
    CREATE TABLE IF NOT EXISTS entries (
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

    CREATE TABLE IF NOT EXISTS entry_archetypes (
      entry_id TEXT NOT NULL,
      archetype TEXT NOT NULL,
      strength DOUBLE DEFAULT 1.0,
      PRIMARY KEY (entry_id, archetype)
    );

    CREATE TABLE IF NOT EXISTS entry_concepts (
      entry_id TEXT NOT NULL,
      concept TEXT NOT NULL,
      concept_type TEXT NOT NULL,
      weight DOUBLE DEFAULT 1.0,
      PRIMARY KEY (entry_id, concept, concept_type)
    );

    CREATE INDEX IF NOT EXISTS idx_entries_date ON entries(entry_date);
    CREATE INDEX IF NOT EXISTS idx_concepts_concept ON entry_concepts(concept);
    CREATE INDEX IF NOT EXISTS idx_concepts_type ON entry_concepts(concept_type);
    CREATE INDEX IF NOT EXISTS idx_archetypes_archetype ON entry_archetypes(archetype);
  `);

  // Migration: add data_quality column to existing tables
  try {
    await exec(`ALTER TABLE entries ADD COLUMN data_quality TEXT DEFAULT 'clean'`);
    await exec(`UPDATE entries SET data_quality = 'clean' WHERE data_quality IS NULL`);
  } catch {
    // Column already exists — ignore
  }
}
