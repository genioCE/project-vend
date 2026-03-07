"""Neo4j graph database operations — node creation, queries, and schema management."""

import os
import logging

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_PASSWORD is required")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def run_query(query: str, params: dict | None = None) -> list[dict]:
    with get_driver().session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def create_indexes():
    queries = [
        "CREATE INDEX IF NOT EXISTS FOR (e:Entry) ON (e.date)",
        "CREATE INDEX IF NOT EXISTS FOR (e:Entry) ON (e.filename)",
        "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.normalized_name)",
        "CREATE INDEX IF NOT EXISTS FOR (p:Place) ON (p.normalized_name)",
        "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.normalized_name)",
        "CREATE INDEX IF NOT EXISTS FOR (em:Emotion) ON (em.name)",
        "CREATE INDEX IF NOT EXISTS FOR (a:Archetype) ON (a.name)",
        # Enrichment node types
        "CREATE INDEX IF NOT EXISTS FOR (t:Theme) ON (t.normalized_name)",
        "CREATE INDEX IF NOT EXISTS FOR (o:Organization) ON (o.normalized_name)",
        "CREATE INDEX IF NOT EXISTS FOR (s:Spiritual) ON (s.normalized_name)",
    ]
    for q in queries:
        run_query(q)
    logger.info("Indexes created")


def clear_graph():
    # Delete in batches to avoid memory issues on large graphs
    while True:
        result = run_query("MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) as deleted")
        if not result or result[0].get("deleted", 0) == 0:
            break
    logger.info("Graph cleared")


def delete_entry_data(filename: str) -> None:
    """
    Surgically remove an Entry node, its owned Decision nodes,
    and all relationships. Shared nodes (Person, Place, Concept,
    Emotion, Archetype) are left intact — they may be referenced
    by other entries.
    """
    # Delete Decision nodes owned by this entry (they are 1:1 with entries)
    run_query(
        """MATCH (e:Entry {filename: $filename})-[:RECORDS]->(d:Decision)
           DETACH DELETE d""",
        {"filename": filename},
    )
    # Delete the entry node and its remaining relationships
    run_query(
        """MATCH (e:Entry {filename: $filename})
           DETACH DELETE e""",
        {"filename": filename},
    )
    logger.info(f"Deleted entry data for: {filename}")


# ---------------------------------------------------------------------------
# Node creation (all use MERGE for idempotent re-runs)
# ---------------------------------------------------------------------------

def create_entry(date: str, filename: str, word_count: int, title: str = ""):
    run_query(
        """MERGE (e:Entry {filename: $filename})
           SET e.date = $date, e.word_count = $word_count, e.title = $title""",
        {"date": date, "filename": filename, "word_count": word_count, "title": title},
    )


def create_person(name: str, entry_filename: str):
    normalized = name.lower().strip()
    run_query(
        """MERGE (p:Person {normalized_name: $normalized})
           SET p.name = $name
           WITH p
           MATCH (e:Entry {filename: $filename})
           MERGE (e)-[:MENTIONS]->(p)""",
        {"name": name, "normalized": normalized, "filename": entry_filename},
    )


def create_place(name: str, entry_filename: str):
    normalized = name.lower().strip()
    run_query(
        """MERGE (p:Place {normalized_name: $normalized})
           SET p.name = $name
           WITH p
           MATCH (e:Entry {filename: $filename})
           MERGE (e)-[:MENTIONS]->(p)""",
        {"name": name, "normalized": normalized, "filename": entry_filename},
    )


def create_concept(name: str, entry_filename: str):
    normalized = name.lower().strip()
    run_query(
        """MERGE (c:Concept {normalized_name: $normalized})
           SET c.name = $name
           WITH c
           MATCH (e:Entry {filename: $filename})
           MERGE (e)-[:CONTAINS]->(c)""",
        {"name": name, "normalized": normalized, "filename": entry_filename},
    )


def create_emotion(emotion: str, intensity: float, entry_filename: str):
    run_query(
        """MERGE (em:Emotion {name: $emotion})
           WITH em
           MATCH (e:Entry {filename: $filename})
           MERGE (e)-[r:EXPRESSES]->(em)
           SET r.intensity = $intensity""",
        {"emotion": emotion, "intensity": intensity, "filename": entry_filename},
    )


def create_decision(text: str, entry_filename: str):
    run_query(
        """MATCH (e:Entry {filename: $filename})
           MERGE (e)-[:RECORDS]->(d:Decision {text: $text})""",
        {"text": text, "filename": entry_filename},
    )


def create_archetype(name: str, strength: float, entry_filename: str):
    run_query(
        """MERGE (a:Archetype {name: $name})
           WITH a
           MATCH (e:Entry {filename: $filename})
           MERGE (e)-[r:INVOKES]->(a)
           SET r.strength = $strength""",
        {"name": name, "strength": strength, "filename": entry_filename},
    )


def create_cooccurrence(concept1: str, concept2: str, weight: int = 1):
    n1 = concept1.lower().strip()
    n2 = concept2.lower().strip()
    if n1 == n2:
        return
    # Always store in alphabetical order to avoid duplicates
    a, b = (n1, n2) if n1 < n2 else (n2, n1)
    run_query(
        """MATCH (c1:Concept {normalized_name: $a})
           MATCH (c2:Concept {normalized_name: $b})
           MERGE (c1)-[r:COOCCURS_WITH]->(c2)
           ON CREATE SET r.weight = $weight
           ON MATCH SET r.weight = r.weight + $weight""",
        {"a": a, "b": b, "weight": weight},
    )


def _dedupe_normalized_entities(values: list[str]) -> list[dict]:
    seen: dict[str, str] = {}
    for raw in values:
        normalized = raw.lower().strip()
        if not normalized:
            continue
        if normalized not in seen:
            seen[normalized] = raw.strip()
    return [{"normalized": normalized, "name": name} for normalized, name in seen.items()]


def _build_cooccurrence_pairs(concepts: list[dict]) -> list[dict]:
    normalized = sorted({c["normalized"] for c in concepts if c.get("normalized")})
    pairs: list[dict] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            pairs.append({"a": normalized[i], "b": normalized[j], "weight": 1})
    return pairs


def ingest_entry(
    date: str,
    filename: str,
    word_count: int,
    title: str,
    people: list[str],
    places: list[str],
    concepts: list[str],
    emotions: list[dict],
    decisions: list[str],
    archetypes: list[dict],
    transitions: list[dict] | None = None,
) -> None:
    people_data = _dedupe_normalized_entities(people)
    places_data = _dedupe_normalized_entities(places)
    concepts_data = _dedupe_normalized_entities(concepts)
    cooccurrences = _build_cooccurrence_pairs(concepts_data)

    payload = {
        "date": date,
        "filename": filename,
        "word_count": word_count,
        "title": title,
        "people": people_data,
        "places": places_data,
        "concepts": concepts_data,
        "emotions": [e for e in emotions if e.get("emotion")],
        "decisions": [d.strip() for d in decisions if d and d.strip()],
        "archetypes": [a for a in archetypes if a.get("archetype")],
        "cooccurrences": cooccurrences,
        "transitions": [
            {
                "from_concept": str(item.get("from_concept", "")).strip().lower(),
                "to_concept": str(item.get("to_concept", "")).strip().lower(),
                "phrase": str(item.get("phrase", "")).strip(),
                "strength": max(0.1, float(item.get("strength", 1.0))),
            }
            for item in (transitions or [])
            if isinstance(item, dict)
            and str(item.get("from_concept", "")).strip()
            and str(item.get("to_concept", "")).strip()
        ],
    }

    with get_driver().session() as session:
        # Entry node
        session.run(
            """MERGE (e:Entry {filename: $filename})
               SET e.date = $date, e.word_count = $word_count, e.title = $title""",
            payload,
        ).consume()

        if payload["people"]:
            session.run(
                """MATCH (e:Entry {filename: $filename})
                   UNWIND $people AS person
                   MERGE (p:Person {normalized_name: person.normalized})
                   SET p.name = person.name
                   MERGE (e)-[:MENTIONS]->(p)""",
                payload,
            ).consume()

        if payload["places"]:
            session.run(
                """MATCH (e:Entry {filename: $filename})
                   UNWIND $places AS place
                   MERGE (p:Place {normalized_name: place.normalized})
                   SET p.name = place.name
                   MERGE (e)-[:MENTIONS]->(p)""",
                payload,
            ).consume()

        if payload["concepts"]:
            session.run(
                """MATCH (e:Entry {filename: $filename})
                   UNWIND $concepts AS concept
                   MERGE (c:Concept {normalized_name: concept.normalized})
                   SET c.name = concept.name
                   MERGE (e)-[:CONTAINS]->(c)""",
                payload,
            ).consume()

        if payload["emotions"]:
            session.run(
                """MATCH (e:Entry {filename: $filename})
                   UNWIND $emotions AS item
                   MERGE (em:Emotion {name: item.emotion})
                   MERGE (e)-[r:EXPRESSES]->(em)
                   SET r.intensity = item.intensity""",
                payload,
            ).consume()

        if payload["decisions"]:
            session.run(
                """MATCH (e:Entry {filename: $filename})
                   UNWIND $decisions AS decision_text
                   MERGE (e)-[:RECORDS]->(d:Decision {text: decision_text})""",
                payload,
            ).consume()

        if payload["archetypes"]:
            session.run(
                """MATCH (e:Entry {filename: $filename})
                   UNWIND $archetypes AS item
                   MERGE (a:Archetype {name: item.archetype})
                   MERGE (e)-[r:INVOKES]->(a)
                   SET r.strength = item.strength""",
                payload,
            ).consume()

        if payload["cooccurrences"]:
            session.run(
                """UNWIND $cooccurrences AS pair
                   MATCH (c1:Concept {normalized_name: pair.a})
                   MATCH (c2:Concept {normalized_name: pair.b})
                   MERGE (c1)-[r:COOCCURS_WITH]->(c2)
                   ON CREATE SET r.weight = pair.weight
                   ON MATCH SET r.weight = r.weight + pair.weight""",
                payload,
            ).consume()

        if payload["transitions"]:
            session.run(
                """MATCH (e:Entry {filename: $filename})
                   UNWIND $transitions AS item
                   MERGE (from_c:Concept {normalized_name: item.from_concept})
                   SET from_c.name = coalesce(from_c.name, item.from_concept)
                   MERGE (to_c:Concept {normalized_name: item.to_concept})
                   SET to_c.name = coalesce(to_c.name, item.to_concept)
                   MERGE (e)-[:CONTAINS]->(from_c)
                   MERGE (e)-[:CONTAINS]->(to_c)
                   MERGE (from_c)-[r:FLOWS_TO]->(to_c)
                   ON CREATE SET
                     r.weight = item.strength,
                     r.first_seen = $date,
                     r.sample_phrase = item.phrase
                   ON MATCH SET
                     r.weight = r.weight + item.strength,
                     r.last_seen = $date,
                     r.sample_phrase = CASE
                       WHEN r.sample_phrase IS NULL OR r.sample_phrase = '' THEN item.phrase
                       ELSE r.sample_phrase
                     END""",
                payload,
            ).consume()


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_concept_network(concept_name: str, limit: int = 30) -> list[dict]:
    normalized = concept_name.lower().strip()
    return run_query(
        """MATCH (c:Concept {normalized_name: $name})-[r]-(connected)
           RETURN type(r) as relationship,
                  labels(connected)[0] as node_type,
                  CASE WHEN connected.name IS NOT NULL THEN connected.name
                       WHEN connected.filename IS NOT NULL THEN connected.filename
                       ELSE coalesce(connected.text, toString(id(connected)))
                  END as node_name,
                  connected.date as date,
                  properties(r) as rel_props
           ORDER BY connected.date DESC
           LIMIT $limit""",
        {"name": normalized, "limit": limit},
    )


def get_person_network(person_name: str, limit: int = 30) -> list[dict]:
    normalized = person_name.lower().strip()
    return run_query(
        """MATCH (p:Person {normalized_name: $name})<-[:MENTIONS]-(e:Entry)
           OPTIONAL MATCH (e)-[:CONTAINS]->(c:Concept)
           OPTIONAL MATCH (e)-[:EXPRESSES]->(em:Emotion)
           RETURN e.date as date, e.filename as filename,
                  collect(DISTINCT c.name) as concepts,
                  collect(DISTINCT em.name) as emotions
           ORDER BY e.date DESC
           LIMIT $limit""",
        {"name": normalized, "limit": limit},
    )


def get_concept_evolution(concept_name: str, limit: int = 20) -> list[dict]:
    normalized = concept_name.lower().strip()
    return run_query(
        """MATCH (e:Entry)-[:CONTAINS]->(c:Concept {normalized_name: $name})
           OPTIONAL MATCH (e)-[:EXPRESSES]->(em:Emotion)
           OPTIONAL MATCH (e)-[:MENTIONS]->(p:Person)
           RETURN e.date as date, e.filename as filename,
                  e.word_count as word_count,
                  collect(DISTINCT em.name) as emotions,
                  collect(DISTINCT p.name) as people
           ORDER BY e.date ASC
           LIMIT $limit""",
        {"name": normalized, "limit": limit},
    )


def compare_periods(start1: str, end1: str, start2: str, end2: str) -> dict:
    def _period_data(start: str, end: str) -> dict:
        rows = run_query(
            """MATCH (e:Entry) WHERE e.date >= $start AND e.date <= $end
               OPTIONAL MATCH (e)-[:CONTAINS]->(c:Concept)
               OPTIONAL MATCH (e)-[:EXPRESSES]->(em:Emotion)
               OPTIONAL MATCH (e)-[:INVOKES]->(a:Archetype)
               OPTIONAL MATCH (e)-[:HAS_THEME]->(t:Theme)
               RETURN collect(DISTINCT c.name) as concepts,
                      collect(DISTINCT em.name) as emotions,
                      collect(DISTINCT a.name) as archetypes,
                      collect(DISTINCT t.name) as themes,
                      count(DISTINCT e) as entry_count,
                      avg(e.state_valence) as avg_valence,
                      avg(e.state_activation) as avg_activation,
                      avg(e.state_agency) as avg_agency""",
            {"start": start, "end": end},
        )
        data = rows[0] if rows else {}
        # Round averages for readability
        for key in ("avg_valence", "avg_activation", "avg_agency"):
            if data.get(key) is not None:
                data[key] = round(data[key], 2)
        return data

    return {
        "period1": {"range": f"{start1} to {end1}", **_period_data(start1, end1)},
        "period2": {"range": f"{start2} to {end2}", **_period_data(start2, end2)},
    }


def get_decision_context(keyword: str | None = None, limit: int = 10) -> list[dict]:
    if keyword:
        return run_query(
            """MATCH (e:Entry)-[:RECORDS]->(d:Decision)
               WHERE toLower(d.text) CONTAINS toLower($keyword)
               OPTIONAL MATCH (e)-[:EXPRESSES]->(em:Emotion)
               OPTIONAL MATCH (e)-[:CONTAINS]->(c:Concept)
               RETURN d.text as decision, e.date as date, e.filename as filename,
                      collect(DISTINCT em.name) as emotions,
                      collect(DISTINCT c.name)[..5] as concepts
               ORDER BY e.date DESC
               LIMIT $limit""",
            {"keyword": keyword, "limit": limit},
        )
    return run_query(
        """MATCH (e:Entry)-[:RECORDS]->(d:Decision)
           OPTIONAL MATCH (e)-[:EXPRESSES]->(em:Emotion)
           OPTIONAL MATCH (e)-[:CONTAINS]->(c:Concept)
           RETURN d.text as decision, e.date as date, e.filename as filename,
                  collect(DISTINCT em.name) as emotions,
                  collect(DISTINCT c.name)[..5] as concepts
           ORDER BY e.date DESC
           LIMIT $limit""",
        {"limit": limit},
    )


def get_archetype_patterns(limit: int = 10) -> list[dict]:
    return run_query(
        """MATCH (e:Entry)-[r:INVOKES]->(a:Archetype)
           OPTIONAL MATCH (e)-[:EXPRESSES]->(em:Emotion)
           RETURN a.name as archetype,
                  round(avg(r.strength) * 100) / 100 as avg_strength,
                  count(DISTINCT e) as entry_count,
                  collect(DISTINCT em.name) as associated_emotions,
                  collect(DISTINCT e.date)[..5] as sample_dates
           ORDER BY entry_count DESC
           LIMIT $limit""",
        {"limit": limit},
    )


def get_concept_flows(concept_name: str, limit: int = 20) -> list[dict]:
    normalized = concept_name.lower().strip()
    rows = run_query(
        """MATCH (source:Concept)-[r]->(target:Concept)
           WHERE type(r) = 'FLOWS_TO'
             AND (source.normalized_name = $name OR target.normalized_name = $name)
           RETURN source.name as source,
                  target.name as target,
                  round(r.weight * 100) / 100 as weight,
                  properties(r) as rel_props,
                  CASE
                    WHEN source.normalized_name = $name THEN 'outgoing'
                    ELSE 'incoming'
                  END as direction
           ORDER BY weight DESC
           LIMIT $limit""",
        {"name": normalized, "limit": limit},
    )
    for row in rows:
        props = row.get("rel_props", {}) if isinstance(row.get("rel_props"), dict) else {}
        row["first_seen"] = props.get("first_seen")
        row["last_seen"] = props.get("last_seen")
        row["sample_phrase"] = props.get("sample_phrase")
        row.pop("rel_props", None)
    return rows


def get_subgraph(center: str, depth: int = 1, limit: int = 50) -> dict:
    normalized = center.lower().strip()
    try:
        bounded_depth = max(1, min(int(depth), 3))
    except (TypeError, ValueError):
        bounded_depth = 1
    try:
        bounded_limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        bounded_limit = 50
    range_pattern = f"*1..{bounded_depth}"

    nodes = run_query(
        f"""MATCH (n)
           WHERE toLower(coalesce(n.normalized_name, "")) = $name
              OR toLower(coalesce(n.name, "")) = $name
              OR toLower(coalesce(n.filename, "")) = $name
           WITH collect(DISTINCT n) AS roots
           UNWIND roots AS root
           OPTIONAL MATCH path=(root)-[{range_pattern}]-(connected)
           WITH roots, [n IN collect(DISTINCT connected) WHERE n IS NOT NULL] AS neighbors
           UNWIND (roots + neighbors) as node
           WITH DISTINCT node
           RETURN id(node) as id,
                  labels(node)[0] as type,
                  CASE WHEN node.name IS NOT NULL THEN node.name
                       WHEN node.normalized_name IS NOT NULL THEN node.normalized_name
                       WHEN node.filename IS NOT NULL THEN node.filename
                       ELSE toString(id(node))
                  END as name
           LIMIT $limit""",
        {"name": normalized, "limit": bounded_limit},
    )
    edges = run_query(
        f"""MATCH (n)
           WHERE toLower(coalesce(n.normalized_name, "")) = $name
              OR toLower(coalesce(n.name, "")) = $name
              OR toLower(coalesce(n.filename, "")) = $name
           WITH collect(DISTINCT n) AS roots
           UNWIND roots AS root
           OPTIONAL MATCH path=(root)-[{range_pattern}]-(connected)
           WITH roots, [n IN collect(DISTINCT connected) WHERE n IS NOT NULL] AS neighbors,
                [p IN collect(path) WHERE p IS NOT NULL] AS paths
           WITH (roots + neighbors) AS nodes, paths
           UNWIND paths AS p
           UNWIND relationships(p) AS r
           WITH nodes, r
           WHERE startNode(r) IN nodes AND endNode(r) IN nodes
           RETURN DISTINCT id(startNode(r)) as source,
                  id(endNode(r)) as target,
                  type(r) as relationship
           LIMIT $limit""",
        {"name": normalized, "limit": bounded_limit},
    )
    return {"nodes": nodes, "edges": edges}


def get_theme_network(theme_name: str, limit: int = 30) -> list[dict]:
    """Get entries linked to a theme with co-occurring themes and emotions."""
    normalized = theme_name.lower().strip()
    return run_query(
        """MATCH (t:Theme {normalized_name: $name})<-[:HAS_THEME]-(e:Entry)
           OPTIONAL MATCH (e)-[:HAS_THEME]->(other:Theme)
           WHERE other.normalized_name <> $name
           OPTIONAL MATCH (e)-[:EXPRESSES]->(em:Emotion)
           RETURN e.date as date, e.filename as filename,
                  e.short_summary as summary,
                  e.state_valence as valence,
                  collect(DISTINCT other.name) as co_themes,
                  collect(DISTINCT em.name) as emotions
           ORDER BY e.date DESC
           LIMIT $limit""",
        {"name": normalized, "limit": limit},
    )


def get_theme_cooccurrences(theme_name: str, limit: int = 20) -> list[dict]:
    """Get themes that most frequently co-occur with a given theme."""
    normalized = theme_name.lower().strip()
    return run_query(
        """MATCH (t:Theme {normalized_name: $name})-[r:THEME_COOCCURS]-(other:Theme)
           RETURN other.name as theme,
                  r.weight as weight
           ORDER BY r.weight DESC
           LIMIT $limit""",
        {"name": normalized, "limit": limit},
    )


def get_entries_by_state(
    dimension: str,
    min_score: float = -1.0,
    max_score: float = 1.0,
    limit: int = 20,
) -> list[dict]:
    """Find entries filtered by a psychological state dimension."""
    valid_dims = {
        "valence", "activation", "agency", "certainty",
        "relational_openness", "self_trust", "time_orientation", "integration",
    }
    if dimension not in valid_dims:
        return []
    prop = f"e.state_{dimension}"
    return run_query(
        f"""MATCH (e:Entry)
           WHERE {prop} IS NOT NULL
             AND {prop} >= $min_score
             AND {prop} <= $max_score
           OPTIONAL MATCH (e)-[:HAS_THEME]->(t:Theme)
           RETURN e.date as date, e.filename as filename,
                  e.short_summary as summary,
                  {prop} as score,
                  collect(DISTINCT t.name) as themes
           ORDER BY {prop} DESC
           LIMIT $limit""",
        {"min_score": min_score, "max_score": max_score, "limit": limit},
    )


def get_organization_network(org_name: str, limit: int = 30) -> list[dict]:
    """Get entries mentioning an organization with concepts, emotions, themes."""
    normalized = org_name.lower().strip()
    return run_query(
        """MATCH (o:Organization {normalized_name: $name})<-[:MENTIONS]-(e:Entry)
           OPTIONAL MATCH (e)-[:CONTAINS]->(c:Concept)
           OPTIONAL MATCH (e)-[:EXPRESSES]->(em:Emotion)
           OPTIONAL MATCH (e)-[:HAS_THEME]->(t:Theme)
           RETURN e.date as date, e.filename as filename,
                  collect(DISTINCT c.name) as concepts,
                  collect(DISTINCT em.name) as emotions,
                  collect(DISTINCT t.name) as themes
           ORDER BY e.date DESC
           LIMIT $limit""",
        {"name": normalized, "limit": limit},
    )


def get_graph_stats() -> dict:
    node_counts = run_query(
        """MATCH (n)
           RETURN labels(n)[0] as label, count(n) as count
           ORDER BY count DESC"""
    )
    rel_counts = run_query(
        """MATCH ()-[r]->()
           RETURN type(r) as type, count(r) as count
           ORDER BY count DESC"""
    )
    return {"nodes": node_counts, "relationships": rel_counts}
