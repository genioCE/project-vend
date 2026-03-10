"""Graph consistency checks — detect data quality issues in the knowledge graph."""

import logging
from typing import Any

from .graph import run_query

logger = logging.getLogger(__name__)


def check_orphaned_nodes() -> dict[str, Any]:
    """
    Find nodes that have no relationships to Entry nodes.
    These are "orphaned" and may indicate deleted entries or ingestion errors.
    """
    checks = {
        "Person": """
            MATCH (p:Person)
            WHERE NOT exists { MATCH (:Entry)-[:MENTIONS]->(p) }
            RETURN p.name as name, p.normalized_name as normalized_name
            LIMIT 50
        """,
        "Place": """
            MATCH (p:Place)
            WHERE NOT exists { MATCH (:Entry)-[:MENTIONS]->(p) }
            RETURN p.name as name, p.normalized_name as normalized_name
            LIMIT 50
        """,
        "Concept": """
            MATCH (c:Concept)
            WHERE NOT exists { MATCH (:Entry)-[:CONTAINS]->(c) }
            RETURN c.name as name, c.normalized_name as normalized_name
            LIMIT 50
        """,
        "Emotion": """
            MATCH (em:Emotion)
            WHERE NOT exists { MATCH (:Entry)-[:EXPRESSES]->(em) }
            RETURN em.name as name
            LIMIT 50
        """,
        "Archetype": """
            MATCH (a:Archetype)
            WHERE NOT exists { MATCH (:Entry)-[:INVOKES]->(a) }
            RETURN a.name as name
            LIMIT 50
        """,
        "Theme": """
            MATCH (t:Theme)
            WHERE NOT exists { MATCH (:Entry)-[:HAS_THEME]->(t) }
            RETURN t.name as name, t.normalized_name as normalized_name
            LIMIT 50
        """,
    }

    results = {}
    total_orphaned = 0

    for label, query in checks.items():
        try:
            rows = run_query(query)
            results[label] = {
                "count": len(rows),
                "samples": rows[:10],  # Return up to 10 samples
            }
            total_orphaned += len(rows)
        except Exception as e:
            results[label] = {"error": str(e)}

    return {
        "check": "orphaned_nodes",
        "total_orphaned": total_orphaned,
        "by_type": results,
        "status": "ok" if total_orphaned == 0 else "warning",
    }


def check_duplicate_names() -> dict[str, Any]:
    """
    Find nodes where normalized_name appears multiple times.
    This shouldn't happen if MERGE is working correctly, but worth checking.
    """
    checks = {
        "Person": """
            MATCH (p:Person)
            WITH p.normalized_name as norm, collect(p.name) as names, count(*) as cnt
            WHERE cnt > 1
            RETURN norm as normalized_name, names, cnt as count
            ORDER BY cnt DESC
            LIMIT 20
        """,
        "Place": """
            MATCH (p:Place)
            WITH p.normalized_name as norm, collect(p.name) as names, count(*) as cnt
            WHERE cnt > 1
            RETURN norm as normalized_name, names, cnt as count
            ORDER BY cnt DESC
            LIMIT 20
        """,
        "Concept": """
            MATCH (c:Concept)
            WITH c.normalized_name as norm, collect(c.name) as names, count(*) as cnt
            WHERE cnt > 1
            RETURN norm as normalized_name, names, cnt as count
            ORDER BY cnt DESC
            LIMIT 20
        """,
        "Theme": """
            MATCH (t:Theme)
            WITH t.normalized_name as norm, collect(t.name) as names, count(*) as cnt
            WHERE cnt > 1
            RETURN norm as normalized_name, names, cnt as count
            ORDER BY cnt DESC
            LIMIT 20
        """,
    }

    results = {}
    total_duplicates = 0

    for label, query in checks.items():
        try:
            rows = run_query(query)
            results[label] = {
                "count": len(rows),
                "duplicates": rows,
            }
            total_duplicates += len(rows)
        except Exception as e:
            results[label] = {"error": str(e)}

    return {
        "check": "duplicate_names",
        "total_duplicate_groups": total_duplicates,
        "by_type": results,
        "status": "ok" if total_duplicates == 0 else "warning",
    }


def check_missing_state_profiles() -> dict[str, Any]:
    """
    Find Entry nodes that are missing psychological state dimensions.
    These may not have been enriched yet.
    """
    # Check for entries missing any state dimension
    query = """
        MATCH (e:Entry)
        WHERE e.state_valence IS NULL
           OR e.state_activation IS NULL
           OR e.state_agency IS NULL
           OR e.state_certainty IS NULL
           OR e.state_relational_openness IS NULL
           OR e.state_self_trust IS NULL
           OR e.state_time_orientation IS NULL
           OR e.state_integration IS NULL
        RETURN e.filename as filename, e.date as date,
               CASE WHEN e.state_valence IS NULL THEN 'valence' ELSE null END as missing_valence,
               CASE WHEN e.state_activation IS NULL THEN 'activation' ELSE null END as missing_activation,
               CASE WHEN e.state_agency IS NULL THEN 'agency' ELSE null END as missing_agency,
               CASE WHEN e.state_certainty IS NULL THEN 'certainty' ELSE null END as missing_certainty,
               CASE WHEN e.state_relational_openness IS NULL THEN 'relational_openness' ELSE null END as missing_relational_openness,
               CASE WHEN e.state_self_trust IS NULL THEN 'self_trust' ELSE null END as missing_self_trust,
               CASE WHEN e.state_time_orientation IS NULL THEN 'time_orientation' ELSE null END as missing_time_orientation,
               CASE WHEN e.state_integration IS NULL THEN 'integration' ELSE null END as missing_integration
        ORDER BY e.date DESC
        LIMIT 100
    """

    try:
        rows = run_query(query)

        # Count total entries and entries with state profiles
        counts = run_query("""
            MATCH (e:Entry)
            RETURN count(*) as total,
                   sum(CASE WHEN e.state_valence IS NOT NULL THEN 1 ELSE 0 END) as with_state
        """)
        total = counts[0]["total"] if counts else 0
        with_state = counts[0]["with_state"] if counts else 0
        missing = total - with_state

        return {
            "check": "missing_state_profiles",
            "total_entries": total,
            "entries_with_state": with_state,
            "entries_missing_state": missing,
            "coverage_pct": round(with_state / total * 100, 1) if total > 0 else 0,
            "samples": rows[:20],
            "status": "ok" if missing == 0 else ("warning" if missing < total * 0.1 else "critical"),
        }
    except Exception as e:
        return {
            "check": "missing_state_profiles",
            "error": str(e),
            "status": "error",
        }


def check_abnormal_cooccurrence_weights() -> dict[str, Any]:
    """
    Find COOCCURS relationships with unusually high weights.
    High weights may indicate extraction errors or over-counting.
    """
    query = """
        MATCH (a:Concept)-[r:COOCCURS]-(b:Concept)
        WHERE r.weight > 50
        RETURN a.name as concept_a, b.name as concept_b, r.weight as weight
        ORDER BY r.weight DESC
        LIMIT 30
    """

    try:
        rows = run_query(query)

        # Also get distribution stats
        stats = run_query("""
            MATCH (:Concept)-[r:COOCCURS]-(:Concept)
            RETURN avg(r.weight) as avg_weight,
                   max(r.weight) as max_weight,
                   percentileCont(r.weight, 0.95) as p95_weight,
                   count(*) as total_cooccurs
        """)
        stats_row = stats[0] if stats else {}

        # Handle None values from empty result sets
        avg_weight = stats_row.get("avg_weight")
        max_weight = stats_row.get("max_weight")
        p95_weight = stats_row.get("p95_weight")

        return {
            "check": "abnormal_cooccurrence_weights",
            "high_weight_pairs": len(rows),
            "samples": rows,
            "stats": {
                "avg_weight": round(avg_weight, 2) if avg_weight is not None else 0,
                "max_weight": max_weight if max_weight is not None else 0,
                "p95_weight": round(p95_weight, 2) if p95_weight is not None else 0,
                "total_cooccurrences": stats_row.get("total_cooccurs", 0),
            },
            "status": "ok" if len(rows) == 0 else "warning",
        }
    except Exception as e:
        return {
            "check": "abnormal_cooccurrence_weights",
            "error": str(e),
            "status": "error",
        }


def check_disconnected_entries() -> dict[str, Any]:
    """
    Find Entry nodes with no relationships at all.
    These may have failed during ingestion.
    """
    query = """
        MATCH (e:Entry)
        WHERE NOT exists { MATCH (e)-[]-() }
        RETURN e.filename as filename, e.date as date, e.word_count as word_count
        ORDER BY e.date DESC
        LIMIT 50
    """

    try:
        rows = run_query(query)

        return {
            "check": "disconnected_entries",
            "count": len(rows),
            "samples": rows[:20],
            "status": "ok" if len(rows) == 0 else "warning",
        }
    except Exception as e:
        return {
            "check": "disconnected_entries",
            "error": str(e),
            "status": "error",
        }


def check_entries_without_concepts() -> dict[str, Any]:
    """
    Find entries that have no concepts extracted.
    Short entries may legitimately have none, but longer ones should.
    """
    query = """
        MATCH (e:Entry)
        WHERE NOT exists { MATCH (e)-[:CONTAINS]->(:Concept) }
          AND e.word_count > 100
        RETURN e.filename as filename, e.date as date, e.word_count as word_count
        ORDER BY e.word_count DESC
        LIMIT 30
    """

    try:
        rows = run_query(query)

        return {
            "check": "entries_without_concepts",
            "count": len(rows),
            "description": "Entries with >100 words but no concepts extracted",
            "samples": rows[:15],
            "status": "ok" if len(rows) == 0 else "warning",
        }
    except Exception as e:
        return {
            "check": "entries_without_concepts",
            "error": str(e),
            "status": "error",
        }


def cleanup_duplicate_entries() -> dict[str, Any]:
    """
    Remove duplicate Entry nodes, keeping only one per filename.
    Also cleans up the cascading duplicates this caused in Person/Place/Concept nodes.
    """
    results = {"entries_removed": 0, "relationships_removed": 0, "stages": []}

    # Stage 1: Find and delete duplicate Entry nodes (keep lowest elementId)
    query_find_duplicates = """
        MATCH (e:Entry)
        WITH e.filename as filename, collect(e) as entries, count(*) as cnt
        WHERE cnt > 1
        RETURN filename, cnt, [x IN entries | elementId(x)] as ids
    """
    duplicates = run_query(query_find_duplicates)
    results["stages"].append({
        "stage": "find_duplicate_entries",
        "duplicate_groups": len(duplicates),
    })

    if not duplicates:
        results["status"] = "ok"
        results["message"] = "No duplicate entries found"
        return results

    # For each duplicate group, delete all but the first entry
    entries_to_delete = []
    for dup in duplicates:
        # Keep the first ID, delete the rest
        ids_to_delete = dup["ids"][1:]
        entries_to_delete.extend(ids_to_delete)

    # Stage 2: Delete duplicate entries and their relationships
    if entries_to_delete:
        # Delete relationships first, then nodes
        delete_query = """
            UNWIND $ids AS id
            MATCH (e:Entry) WHERE elementId(e) = id
            OPTIONAL MATCH (e)-[r]-()
            DELETE r
            WITH e
            DELETE e
            RETURN count(*) as deleted
        """
        result = run_query(delete_query, {"ids": entries_to_delete})
        results["entries_removed"] = len(entries_to_delete)
        results["stages"].append({
            "stage": "delete_duplicate_entries",
            "entries_deleted": len(entries_to_delete),
        })

    # Stage 3: Clean up orphaned entity nodes (Person, Place, Concept, etc.)
    orphan_cleanup_queries = {
        "Person": """
            MATCH (p:Person)
            WHERE NOT exists { MATCH (:Entry)-[:MENTIONS]->(p) }
            DETACH DELETE p
            RETURN count(*) as deleted
        """,
        "Place": """
            MATCH (p:Place)
            WHERE NOT exists { MATCH (:Entry)-[:MENTIONS]->(p) }
            DETACH DELETE p
            RETURN count(*) as deleted
        """,
        "Concept": """
            MATCH (c:Concept)
            WHERE NOT exists { MATCH (:Entry)-[:CONTAINS]->(c) }
            DETACH DELETE c
            RETURN count(*) as deleted
        """,
        "Theme": """
            MATCH (t:Theme)
            WHERE NOT exists { MATCH (:Entry)-[:HAS_THEME]->(t) }
            DETACH DELETE t
            RETURN count(*) as deleted
        """,
    }

    orphans_removed = {}
    for label, query in orphan_cleanup_queries.items():
        try:
            result = run_query(query)
            deleted = result[0]["deleted"] if result else 0
            orphans_removed[label] = deleted
        except Exception as e:
            orphans_removed[label] = f"error: {e}"

    results["stages"].append({
        "stage": "cleanup_orphaned_entities",
        "removed": orphans_removed,
    })

    # Stage 4: Merge duplicate entity nodes (same normalized_name)
    merge_queries = {
        "Person": """
            MATCH (p:Person)
            WITH p.normalized_name as norm, collect(p) as nodes
            WHERE size(nodes) > 1
            WITH norm, nodes[0] as keep, nodes[1..] as duplicates
            UNWIND duplicates as dup
            MATCH (e:Entry)-[r:MENTIONS]->(dup)
            MERGE (e)-[:MENTIONS]->(keep)
            DELETE r
            WITH dup
            DETACH DELETE dup
            RETURN count(*) as merged
        """,
        "Place": """
            MATCH (p:Place)
            WITH p.normalized_name as norm, collect(p) as nodes
            WHERE size(nodes) > 1
            WITH norm, nodes[0] as keep, nodes[1..] as duplicates
            UNWIND duplicates as dup
            MATCH (e:Entry)-[r:MENTIONS]->(dup)
            MERGE (e)-[:MENTIONS]->(keep)
            DELETE r
            WITH dup
            DETACH DELETE dup
            RETURN count(*) as merged
        """,
        "Concept": """
            MATCH (c:Concept)
            WITH c.normalized_name as norm, collect(c) as nodes
            WHERE size(nodes) > 1
            WITH norm, nodes[0] as keep, nodes[1..] as duplicates
            UNWIND duplicates as dup
            MATCH (e:Entry)-[r:CONTAINS]->(dup)
            MERGE (e)-[:CONTAINS]->(keep)
            DELETE r
            WITH dup
            DETACH DELETE dup
            RETURN count(*) as merged
        """,
    }

    entities_merged = {}
    for label, query in merge_queries.items():
        try:
            result = run_query(query)
            merged = result[0]["merged"] if result else 0
            entities_merged[label] = merged
        except Exception as e:
            entities_merged[label] = f"error: {e}"

    results["stages"].append({
        "stage": "merge_duplicate_entities",
        "merged": entities_merged,
    })

    results["status"] = "completed"
    results["message"] = f"Cleaned up {len(entries_to_delete)} duplicate entries"
    return results


def run_all_consistency_checks() -> dict[str, Any]:
    """Run all consistency checks and return a summary."""
    checks = [
        check_orphaned_nodes,
        check_duplicate_names,
        check_missing_state_profiles,
        check_abnormal_cooccurrence_weights,
        check_disconnected_entries,
        check_entries_without_concepts,
    ]

    results = []
    status_counts = {"ok": 0, "warning": 0, "critical": 0, "error": 0}

    for check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
            status = result.get("status", "error")
            status_counts[status] = status_counts.get(status, 0) + 1
        except Exception as e:
            results.append({
                "check": check_fn.__name__,
                "error": str(e),
                "status": "error",
            })
            status_counts["error"] += 1

    # Overall status: worst of all checks
    if status_counts["error"] > 0:
        overall_status = "error"
    elif status_counts["critical"] > 0:
        overall_status = "critical"
    elif status_counts["warning"] > 0:
        overall_status = "warning"
    else:
        overall_status = "ok"

    return {
        "overall_status": overall_status,
        "status_counts": status_counts,
        "checks": results,
    }
