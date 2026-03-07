"""Integration tests for incremental vector ingest.

These tests mock the embedding and ChromaDB layers to validate
the incremental logic without requiring real models or databases.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.corpus_utils.file_discovery import compute_file_hash
from app.ingest import ingest_corpus_full, ingest_corpus_incremental


def _make_corpus(tmp_path, files: dict[str, str]) -> Path:
    """Create a temporary corpus directory with the given files."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    for name, content in files.items():
        (corpus / name).write_text(content)
    return corpus


def _mock_embed_texts(texts):
    """Return fake embeddings of the right shape."""
    return [[0.1] * 384 for _ in texts]


class TestIncrementalVectorIngest:
    @patch("app.ingest.embed_texts", side_effect=_mock_embed_texts)
    @patch("app.ingest.delete_collection")
    @patch("app.ingest.get_or_create_collection")
    def test_full_ingest_creates_manifest(self, mock_collection_fn, mock_delete, mock_embed, tmp_path):
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nThis is a test journal entry with enough words.",
            "1-2-2025.md": "# Entry\n\nAnother journal entry for testing purposes.",
        })
        db_path = str(tmp_path / "manifest.sqlite")

        result = ingest_corpus_full(str(corpus), db_path)

        assert result["files_processed"] == 2
        assert result["chunks_indexed"] > 0
        mock_delete.assert_called_once()  # full wipes collection

    @patch("app.ingest.embed_texts", side_effect=_mock_embed_texts)
    @patch("app.ingest.delete_collection")
    @patch("app.ingest.get_or_create_collection")
    def test_incremental_no_changes(self, mock_collection_fn, mock_delete, mock_embed, tmp_path):
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nThis is a test journal entry with enough words.",
        })
        db_path = str(tmp_path / "manifest.sqlite")

        # First: full ingest to populate manifest
        ingest_corpus_full(str(corpus), db_path)

        # Reset mocks
        mock_collection.reset_mock()
        mock_delete.reset_mock()
        mock_embed.reset_mock()

        # Second: incremental — should detect no changes
        result = ingest_corpus_incremental(str(corpus), db_path)

        assert result["new"] == 0
        assert result["modified"] == 0
        assert result["deleted"] == 0
        assert result["unchanged"] == 1
        mock_embed.assert_not_called()  # no embedding work

    @patch("app.ingest.embed_texts", side_effect=_mock_embed_texts)
    @patch("app.ingest.delete_collection")
    @patch("app.ingest.get_or_create_collection")
    def test_incremental_detects_new_file(self, mock_collection_fn, mock_delete, mock_embed, tmp_path):
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nFirst journal entry.",
        })
        db_path = str(tmp_path / "manifest.sqlite")

        ingest_corpus_full(str(corpus), db_path)
        mock_collection.reset_mock()

        # Add a new file
        (corpus / "1-2-2025.md").write_text("# New\n\nBrand new journal entry.")

        result = ingest_corpus_incremental(str(corpus), db_path)

        assert result["new"] == 1
        assert result["unchanged"] == 1
        assert result["chunks_indexed"] > 0

    @patch("app.ingest.embed_texts", side_effect=_mock_embed_texts)
    @patch("app.ingest.delete_collection")
    @patch("app.ingest.get_or_create_collection")
    def test_incremental_detects_modified_file(self, mock_collection_fn, mock_delete, mock_embed, tmp_path):
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nOriginal content here.",
        })
        db_path = str(tmp_path / "manifest.sqlite")

        ingest_corpus_full(str(corpus), db_path)
        mock_collection.reset_mock()

        # Modify the file
        (corpus / "1-1-2025.md").write_text("# Entry\n\nModified content that is different.")

        result = ingest_corpus_incremental(str(corpus), db_path)

        assert result["modified"] == 1
        assert result["new"] == 0
        # Old chunks should be deleted
        mock_collection.delete.assert_called_once()

    @patch("app.ingest.embed_texts", side_effect=_mock_embed_texts)
    @patch("app.ingest.delete_collection")
    @patch("app.ingest.get_or_create_collection")
    def test_incremental_detects_deleted_file(self, mock_collection_fn, mock_delete, mock_embed, tmp_path):
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nContent to delete.",
            "1-2-2025.md": "# Entry\n\nContent to keep.",
        })
        db_path = str(tmp_path / "manifest.sqlite")

        ingest_corpus_full(str(corpus), db_path)
        mock_collection.reset_mock()

        # Delete a file
        (corpus / "1-1-2025.md").unlink()

        result = ingest_corpus_incremental(str(corpus), db_path)

        assert result["deleted"] == 1
        assert result["unchanged"] == 1
        mock_collection.delete.assert_called_once()

    @patch("app.ingest.embed_texts", side_effect=_mock_embed_texts)
    @patch("app.ingest.delete_collection")
    @patch("app.ingest.get_or_create_collection")
    def test_dry_run_does_not_modify(self, mock_collection_fn, mock_delete, mock_embed, tmp_path):
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nFirst entry.",
        })
        db_path = str(tmp_path / "manifest.sqlite")

        ingest_corpus_full(str(corpus), db_path)
        mock_collection.reset_mock()
        mock_embed.reset_mock()

        # Add a new file
        (corpus / "1-2-2025.md").write_text("# New\n\nNew entry.")

        result = ingest_corpus_incremental(str(corpus), db_path, dry_run=True)

        assert result["new"] == 1
        assert result.get("dry_run") is True
        mock_embed.assert_not_called()  # nothing was actually processed
        mock_collection.add.assert_not_called()
