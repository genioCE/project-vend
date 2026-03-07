"""Unit tests for corpus_utils manifest and change detection (graph-service copy)."""

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
        m.close()

    def test_get_missing(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))
        assert m.get("missing.md") is None
        m.close()

    def test_remove(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.upsert("a.md", "/corpus/a.md", "hash1")
        m.remove("a.md")

        assert m.get("a.md") is None
        assert m.count() == 0
        m.close()

    def test_clear(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        m.upsert("a.md", "/a.md", "h1")
        m.upsert("b.md", "/b.md", "h2")
        m.clear()

        assert m.count() == 0
        m.close()


class TestComputeChangeset:
    def test_all_new(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _make_file(corpus, "a.md", "content a")

        cs = compute_changeset(list(corpus.glob("*.md")), m)

        assert len(cs.new) == 1
        assert cs.has_changes
        m.close()

    def test_deleted(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))
        m.upsert("gone.md", "/corpus/gone.md", "hash")

        cs = compute_changeset([], m)

        assert len(cs.deleted) == 1
        assert cs.deleted[0] == "gone.md"
        m.close()

    def test_unchanged(self, tmp_path):
        db = tmp_path / "test.sqlite"
        m = IngestManifest(str(db))

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        f = _make_file(corpus, "a.md", "same content")
        h = compute_file_hash(f)
        m.upsert("a.md", str(f), h)

        cs = compute_changeset([f], m)

        assert len(cs.unchanged) == 1
        assert not cs.has_changes
        m.close()
