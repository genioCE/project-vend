from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .models import EntrySummaryRecord


class EntrySummaryStore(Protocol):
    def upsert(self, record: EntrySummaryRecord) -> EntrySummaryRecord:
        """Persist summary data for an entry."""

    def get(self, entry_id: str) -> EntrySummaryRecord | None:
        """Retrieve summary data for an entry."""


class SQLiteEntrySummaryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def upsert(self, record: EntrySummaryRecord) -> EntrySummaryRecord:
        now = datetime.now(timezone.utc).isoformat()
        payload_json = record.model_dump_json()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entry_summaries (
                    entry_id,
                    payload_json,
                    model_version,
                    prompt_version,
                    schema_version,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    model_version = excluded.model_version,
                    prompt_version = excluded.prompt_version,
                    schema_version = excluded.schema_version,
                    updated_at = excluded.updated_at
                """,
                (
                    record.entry_id,
                    payload_json,
                    record.processing.model_version,
                    record.processing.prompt_version,
                    record.processing.schema_version,
                    record.processing.created_at,
                    now,
                ),
            )

        return record

    def get(self, entry_id: str) -> EntrySummaryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM entry_summaries WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()

        if not row:
            return None
        return EntrySummaryRecord.model_validate_json(row[0])

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entry_summaries (
                    entry_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_entry_summaries_updated_at
                ON entry_summaries(updated_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
