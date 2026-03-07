"""SQLite-backed ingest manifest for tracking file state and computing deltas."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .file_discovery import compute_file_hash

logger = logging.getLogger(__name__)


@dataclass
class ChangeSet:
    """Result of comparing on-disk files against the manifest."""

    new: list[Path] = field(default_factory=list)
    modified: list[Path] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # filenames only
    unchanged: list[Path] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new or self.modified or self.deleted)

    def summary(self) -> str:
        parts = []
        if self.new:
            parts.append(f"{len(self.new)} new")
        if self.modified:
            parts.append(f"{len(self.modified)} modified")
        if self.deleted:
            parts.append(f"{len(self.deleted)} deleted")
        if self.unchanged:
            parts.append(f"{len(self.unchanged)} unchanged")
        return ", ".join(parts) if parts else "empty"


class IngestManifest:
    """SQLite-backed manifest that tracks ingested files and their content hashes."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS manifest (
                filename TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                ingested_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS manifest_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def get(self, filename: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM manifest WHERE filename = ?", (filename,)
        ).fetchone()
        return dict(row) if row else None

    def all_filenames(self) -> set[str]:
        rows = self._conn.execute("SELECT filename FROM manifest").fetchall()
        return {row["filename"] for row in rows}

    def upsert(
        self,
        filename: str,
        file_path: str,
        content_hash: str,
        chunk_count: int = 0,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO manifest (filename, file_path, content_hash, chunk_count, ingested_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(filename) DO UPDATE SET
                   file_path = excluded.file_path,
                   content_hash = excluded.content_hash,
                   chunk_count = excluded.chunk_count,
                   ingested_at = excluded.ingested_at""",
            (filename, file_path, content_hash, chunk_count, now),
        )
        self._conn.commit()

    def remove(self, filename: str) -> None:
        self._conn.execute("DELETE FROM manifest WHERE filename = ?", (filename,))
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM manifest")
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM manifest").fetchone()
        return row["cnt"]

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            """INSERT INTO manifest_meta (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM manifest_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def close(self) -> None:
        self._conn.close()


def compute_changeset(files: list[Path], manifest: IngestManifest) -> ChangeSet:
    """
    Compare on-disk files against the manifest to determine what changed.

    Args:
        files: Deduplicated list of .md file paths.
        manifest: The manifest to compare against.

    Returns:
        ChangeSet with new, modified, deleted, and unchanged files.
    """
    cs = ChangeSet()
    on_disk = {}

    for path in files:
        filename = path.name
        on_disk[filename] = path
        record = manifest.get(filename)

        if record is None:
            cs.new.append(path)
        else:
            current_hash = compute_file_hash(path)
            if current_hash != record["content_hash"]:
                cs.modified.append(path)
            else:
                cs.unchanged.append(path)

    # Files in manifest but not on disk
    manifest_filenames = manifest.all_filenames()
    for filename in manifest_filenames:
        if filename not in on_disk:
            cs.deleted.append(filename)

    return cs
