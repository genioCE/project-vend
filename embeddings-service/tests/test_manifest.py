"""Unit tests for corpus_utils manifest and change detection."""

import tempfile
from pathlib import Path

from app.corpus_utils.manifest import IngestManifest, ChangeSet, compute_changeset
from app.corpus_utils.file_discovery import compute_file_hash


def _make_file(tmp: Path, name: str, content: str = "hello") -> Path:
    p = tmp / name
    p.write_text(content)
    return p


class TestIngestManifest:
    def test_upsert_and_get(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.upsert("a.md", "/corpus/a.md", "abc123", chunk_count=3)
        rec = m.get("a.md")

        assert rec is not None
        assert rec["filename"] == "a.md"
        assert rec["content_hash"] == "abc123"
        assert rec["chunk_count"] == 3
        assert rec["file_path"] == "/corpus/a.md"
        m.close()

    def test_get_missing(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        assert m.get("missing.md") is None
        m.close()

    def test_upsert_overwrites(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.upsert("a.md", "/corpus/a.md", "hash1", chunk_count=2)
        m.upsert("a.md", "/corpus/a.md", "hash2", chunk_count=5)
        rec = m.get("a.md")

        assert rec["content_hash"] == "hash2"
        assert rec["chunk_count"] == 5
        m.close()

    def test_remove(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.upsert("a.md", "/corpus/a.md", "hash1")
        m.remove("a.md")

        assert m.get("a.md") is None
        assert m.count() == 0
        m.close()

    def test_all_filenames(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.upsert("a.md", "/a.md", "h1")
        m.upsert("b.md", "/b.md", "h2")

        assert m.all_filenames() == {"a.md", "b.md"}
        m.close()

    def test_clear(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.upsert("a.md", "/a.md", "h1")
        m.upsert("b.md", "/b.md", "h2")
        m.clear()

        assert m.count() == 0
        m.close()

    def test_meta(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.set_meta("version", "1.0")
        assert m.get_meta("version") == "1.0"
        assert m.get_meta("missing") is None

        m.set_meta("version", "2.0")
        assert m.get_meta("version") == "2.0"
        m.close()


class TestComputeChangeset:
    def test_all_new(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _make_file(corpus, "a.md", "content a")
        _make_file(corpus, "b.md", "content b")

        files = sorted(corpus.glob("*.md"))
        cs = compute_changeset(files, m)

        assert len(cs.new) == 2
        assert len(cs.modified) == 0
        assert len(cs.deleted) == 0
        assert len(cs.unchanged) == 0
        assert cs.has_changes
        m.close()

    def test_unchanged(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        f = _make_file(corpus, "a.md", "content a")
        h = compute_file_hash(f)
        m.upsert("a.md", str(f), h, chunk_count=2)

        cs = compute_changeset([f], m)

        assert len(cs.new) == 0
        assert len(cs.modified) == 0
        assert len(cs.deleted) == 0
        assert len(cs.unchanged) == 1
        assert not cs.has_changes
        m.close()

    def test_modified(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        f = _make_file(corpus, "a.md", "content a")
        m.upsert("a.md", str(f), "old_hash", chunk_count=2)

        # File changed (hash won't match "old_hash")
        cs = compute_changeset([f], m)

        assert len(cs.new) == 0
        assert len(cs.modified) == 1
        assert cs.modified[0].name == "a.md"
        assert len(cs.deleted) == 0
        assert cs.has_changes
        m.close()

    def test_deleted(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.upsert("removed.md", "/corpus/removed.md", "hash1", chunk_count=3)

        # No files on disk
        cs = compute_changeset([], m)

        assert len(cs.new) == 0
        assert len(cs.modified) == 0
        assert len(cs.deleted) == 1
        assert cs.deleted[0] == "removed.md"
        assert cs.has_changes
        m.close()

    def test_mixed_changes(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        corpus = tmp_path / "corpus"
        corpus.mkdir()

        # unchanged
        f1 = _make_file(corpus, "unchanged.md", "same")
        h1 = compute_file_hash(f1)
        m.upsert("unchanged.md", str(f1), h1)

        # modified
        f2 = _make_file(corpus, "modified.md", "new content")
        m.upsert("modified.md", str(f2), "old_hash")

        # new
        f3 = _make_file(corpus, "new.md", "brand new")

        # deleted (in manifest, not on disk)
        m.upsert("deleted.md", "/corpus/deleted.md", "hash")

        cs = compute_changeset([f1, f2, f3], m)

        assert len(cs.new) == 1
        assert len(cs.modified) == 1
        assert len(cs.deleted) == 1
        assert len(cs.unchanged) == 1
        assert cs.has_changes
        m.close()


class TestChangeSetSummary:
    def test_summary_with_changes(self):
        cs = ChangeSet(
            new=[Path("a.md")],
            modified=[Path("b.md"), Path("c.md")],
            deleted=["d.md"],
            unchanged=[Path("e.md")],
        )
        s = cs.summary()
        assert "1 new" in s
        assert "2 modified" in s
        assert "1 deleted" in s
        assert "1 unchanged" in s

    def test_summary_empty(self):
        cs = ChangeSet()
        assert cs.summary() == "empty"
        assert not cs.has_changes


class TestComputeFileHash:
    def test_deterministic(self, tmp_path):
        f = _make_file(tmp_path, "test.md", "hello world")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content(self, tmp_path):
        f1 = _make_file(tmp_path, "a.md", "content a")
        f2 = _make_file(tmp_path, "b.md", "content b")
        assert compute_file_hash(f1) != compute_file_hash(f2)
