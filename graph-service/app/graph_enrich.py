"""Enrich the knowledge graph from analysis data (SQLite).

Reads from the analysis service's SQLite database and enriches
existing graph nodes with themes, typed entities, and state profiles.

Usage:
  python -m app.graph_enrich                    # incremental (skip already-enriched)
  python -m app.graph_enrich --full             # re-enrich everything
  python -m app.graph_enrich --dry-run          # report what would change
  python -m app.graph_enrich --dry-run --full   # preview full enrichment
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from itertools import combinations

from .graph import close_driver, create_indexes, get_driver, run_query

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

ANALYSIS_DB_PATH = os.environ.get("ANALYSIS_DB_PATH", "/analysis-data/analysis.sqlite")

STATE_DIMENSIONS = [
    "valence", "activation", "agency", "certainty",
    "relational_openness", "self_trust", "time_orientation", "integration",
]


# ---------------------------------------------------------------------------
# Schema extensions
# ---------------------------------------------------------------------------

def create_enrichment_indexes():
    """Create indexes for new node labels introduced by enrichment."""
    queries = [
        "CREATE INDEX IF NOT EXISTS FOR (t:Theme) ON (t.normalized_name)",
        "CREATE INDEX IF NOT EXISTS FOR (o:Organization) ON (o.normalized_name)",
        "CREATE INDEX IF NOT EXISTS FOR (s:Spiritual) ON (s.normalized_name)",
    ]
    for q in queries:
        run_query(q)
    logger.info("Enrichment indexes created")


# ---------------------------------------------------------------------------
# Analysis data loading
# ---------------------------------------------------------------------------

def load_analysis_records(db_path: str) -> list[dict]:
    """Load all entry summary records from analysis SQLite."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute(
        "SELECT entry_id, payload_json FROM entry_summaries "
        "WHERE prompt_version LIKE '%v4%' OR prompt_version LIKE '%v5%' OR prompt_version LIKE '%v6%' OR prompt_version LIKE '%claude%'"
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    for entry_id, payload_json in rows:
        try:
            payload = json.loads(payload_json)
            payload["entry_id"] = entry_id
            records.append(payload)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Skipping {entry_id}: invalid JSON — {e}")
    return records


# ---------------------------------------------------------------------------
# State profile extraction
# ---------------------------------------------------------------------------

def extract_state_scores(payload: dict) -> dict[str, float]:
    """Extract 8-dimension state scores from an analysis payload."""
    state_profile = payload.get("state_profile", {})
    dimensions = state_profile.get("dimensions", [])
    scores = {}
    for dim in dimensions:
        name = dim.get("dimension", "")
        if name in STATE_DIMENSIONS:
            scores[name] = float(dim.get("score", 0.0))
    return scores


# ---------------------------------------------------------------------------
# Entry enrichment
# ---------------------------------------------------------------------------

def enrich_entry(session, payload: dict, dry_run: bool = False) -> dict:
    """Enrich a single entry's graph data from analysis payload.

    Returns a stats dict: {themes, entities, state_set, skipped_reason}.
    """
    source_file = payload.get("source_file")
    if not source_file:
        return {"skipped_reason": "no source_file"}

    # Check entry exists in graph
    result = session.run(
        "MATCH (e:Entry {filename: $filename}) RETURN e.filename AS f",
        {"filename": source_file},
    ).data()
    if not result:
        return {"skipped_reason": "not in graph"}

    filename = source_file
    stats = {"themes": 0, "entities": 0, "state_set": False, "skipped_reason": None}

    if dry_run:
        stats["themes"] = len(payload.get("themes", []))
        stats["entities"] = len(payload.get("entities", []))
        stats["state_set"] = True
        return stats

    # --- 1. State profile properties on Entry node ---
    scores = extract_state_scores(payload)
    session.run(
        """MATCH (e:Entry {filename: $filename})
           SET e.state_valence = $valence,
               e.state_activation = $activation,
               e.state_agency = $agency,
               e.state_certainty = $certainty,
               e.state_relational_openness = $relational_openness,
               e.state_self_trust = $self_trust,
               e.state_time_orientation = $time_orientation,
               e.state_integration = $integration,
               e.state_enriched = true,
               e.short_summary = $summary""",
        {
            "filename": filename,
            "valence": scores.get("valence", 0.0),
            "activation": scores.get("activation", 0.0),
            "agency": scores.get("agency", 0.0),
            "certainty": scores.get("certainty", 0.0),
            "relational_openness": scores.get("relational_openness", 0.0),
            "self_trust": scores.get("self_trust", 0.0),
            "time_orientation": scores.get("time_orientation", 0.0),
            "integration": scores.get("integration", 0.0),
            "summary": payload.get("short_summary", ""),
        },
    ).consume()
    stats["state_set"] = True

    # --- 2. Themes ---
    themes = payload.get("themes", [])
    theme_data = []
    for theme in themes:
        if not theme or not isinstance(theme, str) or not theme.strip():
            continue
        normalized = theme.lower().strip()
        theme_data.append({"normalized": normalized, "name": theme.strip()})

    if theme_data:
        session.run(
            """MATCH (e:Entry {filename: $filename})
               UNWIND $themes AS theme
               MERGE (t:Theme {normalized_name: theme.normalized})
               SET t.name = theme.name
               MERGE (e)-[:HAS_THEME]->(t)""",
            {"filename": filename, "themes": theme_data},
        ).consume()
        stats["themes"] = len(theme_data)

        # Theme co-occurrence (alphabetically ordered to avoid duplicates)
        normalized_themes = sorted({t["normalized"] for t in theme_data})
        pairs = [
            {"a": a, "b": b}
            for a, b in combinations(normalized_themes, 2)
        ]
        if pairs:
            session.run(
                """UNWIND $pairs AS pair
                   MATCH (t1:Theme {normalized_name: pair.a})
                   MATCH (t2:Theme {normalized_name: pair.b})
                   MERGE (t1)-[r:THEME_COOCCURS]->(t2)
                     ON CREATE SET r.weight = 1
                     ON MATCH SET r.weight = r.weight + 1""",
                {"pairs": pairs},
            ).consume()

    # --- 3. Typed entities ---
    entities = payload.get("entities", [])
    valence = scores.get("valence", 0.0)
    activation = scores.get("activation", 0.0)
    agency = scores.get("agency", 0.0)

    # Group entities by type for batched UNWIND queries
    by_type: dict[str, list[dict]] = {
        "person": [], "place": [], "organization": [],
        "spiritual": [], "concept": [],
    }
    for entity in entities:
        if isinstance(entity, dict):
            name = entity.get("name", "").strip()
            etype = entity.get("type", "concept").strip()
        elif isinstance(entity, str):
            name = entity.strip()
            etype = "concept"
        else:
            continue
        if not name:
            continue
        normalized = name.lower().strip()
        if etype in by_type:
            by_type[etype].append({"normalized": normalized, "name": name})

    mention_state = {
        "filename": filename,
        "valence": valence,
        "activation": activation,
        "agency": agency,
    }

    if by_type["person"]:
        session.run(
            """MATCH (e:Entry {filename: $filename})
               UNWIND $items AS item
               MERGE (p:Person {normalized_name: item.normalized})
               SET p.name = item.name
               MERGE (e)-[r:MENTIONS]->(p)
               SET r.entry_valence = $valence,
                   r.entry_activation = $activation,
                   r.entry_agency = $agency""",
            {**mention_state, "items": by_type["person"]},
        ).consume()

    if by_type["place"]:
        session.run(
            """MATCH (e:Entry {filename: $filename})
               UNWIND $items AS item
               MERGE (p:Place {normalized_name: item.normalized})
               SET p.name = item.name
               MERGE (e)-[r:MENTIONS]->(p)
               SET r.entry_valence = $valence,
                   r.entry_activation = $activation,
                   r.entry_agency = $agency""",
            {**mention_state, "items": by_type["place"]},
        ).consume()

    if by_type["organization"]:
        session.run(
            """MATCH (e:Entry {filename: $filename})
               UNWIND $items AS item
               MERGE (o:Organization {normalized_name: item.normalized})
               SET o.name = item.name
               MERGE (e)-[r:MENTIONS]->(o)
               SET r.entry_valence = $valence,
                   r.entry_activation = $activation,
                   r.entry_agency = $agency""",
            {**mention_state, "items": by_type["organization"]},
        ).consume()

    if by_type["spiritual"]:
        session.run(
            """MATCH (e:Entry {filename: $filename})
               UNWIND $items AS item
               MERGE (s:Spiritual {normalized_name: item.normalized})
               SET s.name = item.name
               MERGE (e)-[r:MENTIONS]->(s)
               SET r.entry_valence = $valence,
                   r.entry_activation = $activation,
                   r.entry_agency = $agency""",
            {**mention_state, "items": by_type["spiritual"]},
        ).consume()

    if by_type["concept"]:
        session.run(
            """MATCH (e:Entry {filename: $filename})
               UNWIND $items AS item
               MERGE (c:Concept {normalized_name: item.normalized})
               SET c.name = item.name
               MERGE (e)-[:CONTAINS]->(c)""",
            {"filename": filename, "items": by_type["concept"]},
        ).consume()

    entity_count = sum(len(v) for v in by_type.values())
    stats["entities"] = entity_count

    # --- 4. State-weight existing MENTIONS rels from spaCy that lack state ---
    session.run(
        """MATCH (e:Entry {filename: $filename})-[r:MENTIONS]->(target)
           WHERE r.entry_valence IS NULL
           SET r.entry_valence = $valence,
               r.entry_activation = $activation,
               r.entry_agency = $agency""",
        mention_state,
    ).consume()

    return stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_enrichment(
    db_path: str,
    full: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the enrichment pipeline."""
    mode = "FULL" if full else "INCREMENTAL"
    if dry_run:
        mode = f"DRY RUN ({mode})"
    print(f"\n{'='*60}")
    print(f"Graph Enrichment — {mode}")
    print(f"{'='*60}\n")

    # Load analysis data
    print(f"Loading analysis records from {db_path}...")
    records = load_analysis_records(db_path)
    print(f"Loaded {len(records)} records")

    if not records:
        print("No records to process.")
        return

    # Create indexes (idempotent)
    if not dry_run:
        create_indexes()
        create_enrichment_indexes()

    # Determine which entries to process
    if not full and not dry_run:
        # Check which entries are already enriched
        enriched = run_query(
            "MATCH (e:Entry) WHERE e.state_enriched = true "
            "RETURN e.filename AS filename"
        )
        enriched_files = {r["filename"] for r in enriched}
        before_count = len(records)
        records = [r for r in records if r.get("source_file") not in enriched_files]
        skipped = before_count - len(records)
        if skipped:
            print(f"Skipping {skipped} already-enriched entries")

    print(f"Processing {len(records)} entries...\n")

    # Process entries
    total_themes = 0
    total_entities = 0
    total_state = 0
    skipped = 0
    errors = 0

    driver = get_driver()

    for i, payload in enumerate(records):
        entry_id = payload.get("entry_id", "unknown")
        try:
            with driver.session() as session:
                result = enrich_entry(session, payload, dry_run=dry_run)

            if result.get("skipped_reason"):
                skipped += 1
                if i < 5 or result["skipped_reason"] != "not in graph":
                    logger.debug(f"  Skipped {entry_id}: {result['skipped_reason']}")
            else:
                total_themes += result.get("themes", 0)
                total_entities += result.get("entities", 0)
                if result.get("state_set"):
                    total_state += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(records)}...")

        except Exception as e:
            errors += 1
            logger.error(f"  Error enriching {entry_id}: {e}")

    # Report
    print(f"\n{'='*60}")
    print(f"ENRICHMENT {'PREVIEW' if dry_run else 'COMPLETE'}")
    print(f"{'='*60}")
    print(f"  Entries processed:  {len(records) - skipped}")
    print(f"  Entries skipped:    {skipped}")
    print(f"  Errors:             {errors}")
    print(f"  Themes created:     {total_themes}")
    print(f"  Entities merged:    {total_entities}")
    print(f"  State profiles set: {total_state}")

    if not dry_run:
        # Print graph stats
        from .graph import get_graph_stats
        stats = get_graph_stats()
        print(f"\nGraph stats after enrichment:")
        for item in stats.get("nodes", []):
            print(f"  {item['label']}: {item['count']}")
        for item in stats.get("relationships", []):
            print(f"  {item['type']}: {item['count']}")


def main():
    parser = argparse.ArgumentParser(description="Enrich graph from analysis data")
    parser.add_argument("--full", action="store_true", help="Re-enrich all entries")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    db_path = ANALYSIS_DB_PATH
    if not os.path.exists(db_path):
        print(f"ERROR: Analysis database not found at {db_path}")
        sys.exit(1)

    try:
        run_enrichment(db_path, full=args.full, dry_run=args.dry_run)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
