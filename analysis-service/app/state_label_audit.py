"""
Psych Scoring Audit Framework

Provides tools to:
1. Sample entries with extreme dimension scores for review
2. Record human judgments (agree/disagree with each dimension)
3. Generate reports on false positive/negative rates per dimension
4. Track signal accuracy for calibration

This helps identify which dimensions are most/least accurate and which
signals need refinement.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("analysis-service.audit")

# Dimensions to audit
DIMENSIONS = [
    "valence", "activation", "agency", "certainty",
    "relational_openness", "self_trust", "time_orientation", "integration"
]


class AuditSample(BaseModel):
    """An entry sampled for audit review."""
    entry_id: str
    entry_date: str | None
    source_file: str | None
    text_preview: str  # First 500 chars of entry
    dimension: str
    score: float
    anchor: str  # "low" or "high"
    detected_signals: list[dict[str, Any]]  # Signals that contributed to this dimension
    confidence: float


class AuditJudgment(BaseModel):
    """Human judgment on a dimension score."""
    entry_id: str
    dimension: str
    original_score: float
    judgment: Literal["agree", "disagree", "uncertain"]
    correct_direction: Literal["low", "high", "neutral"] | None = None
    notes: str | None = None


class AuditReport(BaseModel):
    """Aggregated audit statistics."""
    total_judgments: int
    by_dimension: dict[str, dict[str, Any]]
    overall_accuracy: float
    worst_dimensions: list[str]
    best_dimensions: list[str]
    signal_issues: list[dict[str, Any]]


class StateAuditStore:
    """SQLite-backed storage for audit judgments."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_judgments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL,
                    dimension TEXT NOT NULL,
                    original_score REAL NOT NULL,
                    judgment TEXT NOT NULL,
                    correct_direction TEXT,
                    notes TEXT,
                    signals_json TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(entry_id, dimension)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_dimension
                ON audit_judgments(dimension)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_judgment
                ON audit_judgments(judgment)
            """)

    def record_judgment(
        self,
        entry_id: str,
        dimension: str,
        original_score: float,
        judgment: str,
        correct_direction: str | None = None,
        notes: str | None = None,
        signals: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Record a human judgment on a dimension score."""
        now = datetime.now(timezone.utc).isoformat()
        signals_json = json.dumps(signals) if signals else None

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO audit_judgments (
                    entry_id, dimension, original_score, judgment,
                    correct_direction, notes, signals_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id, dimension) DO UPDATE SET
                    original_score = excluded.original_score,
                    judgment = excluded.judgment,
                    correct_direction = excluded.correct_direction,
                    notes = excluded.notes,
                    signals_json = excluded.signals_json,
                    created_at = excluded.created_at
            """, (
                entry_id, dimension, original_score, judgment,
                correct_direction, notes, signals_json, now
            ))

        return {
            "status": "recorded",
            "entry_id": entry_id,
            "dimension": dimension,
            "judgment": judgment,
        }

    def get_judgments(
        self,
        dimension: str | None = None,
        judgment: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve audit judgments with optional filters."""
        query = "SELECT * FROM audit_judgments WHERE 1=1"
        params: list[Any] = []

        if dimension:
            query += " AND dimension = ?"
            params.append(dimension)
        if judgment:
            query += " AND judgment = ?"
            params.append(judgment)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_report(self) -> dict[str, Any]:
        """Generate aggregate audit statistics."""
        with self._connect() as conn:
            # Total judgments
            total = conn.execute(
                "SELECT COUNT(*) FROM audit_judgments"
            ).fetchone()[0]

            if total == 0:
                return {
                    "total_judgments": 0,
                    "by_dimension": {},
                    "overall_accuracy": 0,
                    "worst_dimensions": [],
                    "best_dimensions": [],
                    "signal_issues": [],
                }

            # Per-dimension stats
            by_dimension = {}
            for dim in DIMENSIONS:
                stats = conn.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN judgment = 'agree' THEN 1 ELSE 0 END) as agree,
                        SUM(CASE WHEN judgment = 'disagree' THEN 1 ELSE 0 END) as disagree,
                        SUM(CASE WHEN judgment = 'uncertain' THEN 1 ELSE 0 END) as uncertain,
                        AVG(ABS(original_score)) as avg_abs_score
                    FROM audit_judgments
                    WHERE dimension = ?
                """, (dim,)).fetchone()

                dim_total = stats["total"] or 0
                agree = stats["agree"] or 0
                disagree = stats["disagree"] or 0

                by_dimension[dim] = {
                    "total": dim_total,
                    "agree": agree,
                    "disagree": disagree,
                    "uncertain": stats["uncertain"] or 0,
                    "accuracy": round(agree / dim_total, 3) if dim_total > 0 else None,
                    "avg_abs_score": round(stats["avg_abs_score"], 3) if stats["avg_abs_score"] else None,
                }

            # Overall accuracy
            total_agree = conn.execute(
                "SELECT COUNT(*) FROM audit_judgments WHERE judgment = 'agree'"
            ).fetchone()[0]
            total_decided = conn.execute(
                "SELECT COUNT(*) FROM audit_judgments WHERE judgment IN ('agree', 'disagree')"
            ).fetchone()[0]

            overall_accuracy = round(total_agree / total_decided, 3) if total_decided > 0 else 0

            # Rank dimensions by accuracy
            dim_accuracies = [
                (dim, stats["accuracy"])
                for dim, stats in by_dimension.items()
                if stats["accuracy"] is not None
            ]
            dim_accuracies.sort(key=lambda x: x[1])

            worst = [d[0] for d in dim_accuracies[:3] if d[1] is not None]
            best = [d[0] for d in reversed(dim_accuracies[-3:]) if d[1] is not None]

            # Find problematic signals (from disagreements)
            signal_issues = []
            disagree_rows = conn.execute("""
                SELECT dimension, signals_json, notes
                FROM audit_judgments
                WHERE judgment = 'disagree' AND signals_json IS NOT NULL
                LIMIT 50
            """).fetchall()

            # Count signal frequency in disagreements
            signal_counts: dict[str, int] = {}
            for row in disagree_rows:
                try:
                    signals = json.loads(row["signals_json"])
                    for sig in signals:
                        phrase = sig.get("phrase", sig.get("text", "unknown"))
                        key = f"{row['dimension']}:{phrase}"
                        signal_counts[key] = signal_counts.get(key, 0) + 1
                except:
                    pass

            signal_issues = [
                {"signal": k, "disagreement_count": v}
                for k, v in sorted(signal_counts.items(), key=lambda x: -x[1])[:10]
            ]

            return {
                "total_judgments": total,
                "by_dimension": by_dimension,
                "overall_accuracy": overall_accuracy,
                "worst_dimensions": worst,
                "best_dimensions": best,
                "signal_issues": signal_issues,
            }


def get_extreme_entries(
    db_path: str,
    dimension: str | None = None,
    threshold: float = 0.6,
    limit: int = 20,
    exclude_audited: bool = True,
    audit_db_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Sample entries with extreme scores (|score| > threshold) for audit.

    Returns entries sorted by absolute score descending (most extreme first).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get already-audited entry IDs if excluding
    audited_ids: set[str] = set()
    if exclude_audited and audit_db_path:
        try:
            audit_conn = sqlite3.connect(audit_db_path)
            if dimension:
                rows = audit_conn.execute(
                    "SELECT DISTINCT entry_id FROM audit_judgments WHERE dimension = ?",
                    (dimension,)
                ).fetchall()
            else:
                rows = audit_conn.execute(
                    "SELECT DISTINCT entry_id FROM audit_judgments"
                ).fetchall()
            audited_ids = {row[0] for row in rows}
            audit_conn.close()
        except:
            pass

    # Build query for entries with state labels
    results = []

    # Get state labels from the state_labels table
    rows = conn.execute("""
        SELECT entry_id, payload_json
        FROM state_labels
        ORDER BY RANDOM()
        LIMIT 500
    """).fetchall()

    for row in rows:
        entry_id = row["entry_id"]
        if entry_id in audited_ids:
            continue

        try:
            payload = json.loads(row["payload_json"])
            state_profile = payload.get("state_profile", {})
            dimensions_list = state_profile.get("dimensions", [])

            # Convert list to dict for easier lookup
            dimensions_data = {}
            if isinstance(dimensions_list, list):
                for dim_item in dimensions_list:
                    dim_name = dim_item.get("dimension")
                    if dim_name:
                        dimensions_data[dim_name] = dim_item
            elif isinstance(dimensions_list, dict):
                dimensions_data = dimensions_list

            # Check each dimension or specific one
            dims_to_check = [dimension] if dimension else DIMENSIONS

            for dim in dims_to_check:
                dim_data = dimensions_data.get(dim, {})
                score = dim_data.get("score")

                if score is None:
                    continue

                if abs(score) >= threshold:
                    # Get signals for this dimension
                    all_signals = payload.get("observed_text_signals", [])
                    dim_signals = [
                        s for s in all_signals
                        if s.get("dimension") == dim
                    ]

                    # Get anchor from data or infer from score
                    anchor = dim_data.get("label", "")
                    if not anchor or anchor.startswith("between"):
                        anchor = dim_data.get("high_anchor") if score > 0 else dim_data.get("low_anchor")

                    results.append({
                        "entry_id": entry_id,
                        "entry_date": payload.get("entry_date"),
                        "source_file": payload.get("source_file"),
                        "dimension": dim,
                        "score": round(score, 4),
                        "anchor": anchor,
                        "detected_signals": dim_signals[:10],  # Limit signals
                        "confidence": payload.get("confidence", {}).get("overall", 0),
                    })

        except Exception as e:
            logger.warning(f"Error parsing state label for {entry_id}: {e}")
            continue

    conn.close()

    # Sort by absolute score descending and limit
    results.sort(key=lambda x: -abs(x["score"]))
    return results[:limit]


def get_entry_text_preview(db_path: str, entry_id: str, max_chars: int = 800) -> str:
    """Get a text preview for an entry from the entry_summaries table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    row = conn.execute("""
        SELECT payload_json
        FROM entry_summaries
        WHERE entry_id = ?
    """, (entry_id,)).fetchone()

    conn.close()

    if not row:
        return "(Entry text not available)"

    try:
        payload = json.loads(row["payload_json"])

        # Try to get original text from provenance spans
        provenance = payload.get("provenance", {})
        spans = provenance.get("spans", []) or provenance.get("source_spans", [])
        if spans:
            text = " ".join(s.get("excerpt", "") or s.get("text", "") for s in spans)
            if text.strip():
                return text[:max_chars] + "..." if len(text) > max_chars else text

        # Fallback to detailed_summary or short_summary
        summary = (
            payload.get("detailed_summary", "") or
            payload.get("short_summary", "") or
            payload.get("summary", "")
        )
        if summary:
            return summary[:max_chars] + "..." if len(summary) > max_chars else summary

        return "(No text content available)"
    except:
        return "(Error parsing entry)"


# Singleton store instance
_audit_store: StateAuditStore | None = None


def get_audit_store(db_path: str | None = None) -> StateAuditStore:
    """Get or create the audit store singleton."""
    global _audit_store
    if _audit_store is None:
        import os
        path = db_path or os.environ.get(
            "AUDIT_DB_PATH",
            os.environ.get("ENTRY_SUMMARY_DB_PATH", "/service/data/analysis.sqlite")
        )
        _audit_store = StateAuditStore(path)
    return _audit_store
