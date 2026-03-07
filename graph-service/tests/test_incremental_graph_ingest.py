"""Integration tests for incremental graph ingest.

These tests mock the Neo4j and LLM extraction layers to validate
the incremental logic without requiring real databases.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.corpus_utils.file_discovery import compute_file_hash
from app.corpus_utils.manifest import IngestManifest


def _make_corpus(tmp_path, files: dict[str, str]) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    for name, content in files.items():
        (corpus / name).write_text(content)
    return corpus


MOCK_EXTRACTION = {
    "people": ["Alice"],
    "places": ["Home"],
    "concepts": ["testing"],
    "emotions": [{"emotion": "calm", "intensity": 0.5}],
    "decisions": ["write tests"],
    "archetypes": [{"archetype": "Creator", "strength": 0.7}],
    "transitions": [],
}


class TestIncrementalGraphIngest:
    """Test the incremental graph ingest change detection and dispatch."""

    @patch("app.graph_ingest.close_driver")
    @patch("app.graph_ingest.get_graph_stats", return_value={"nodes": [], "relationships": []})
    @patch("app.graph_ingest.ingest_entry")
    @patch("app.graph_ingest.delete_entry_data")
    @patch("app.graph_ingest.create_indexes")
    @patch("app.graph_ingest.clear_graph")
    @patch("app.graph_ingest.extract_entities", return_value=MOCK_EXTRACTION)
    def test_full_ingest_creates_manifest(
        self, mock_extract, mock_clear, mock_indexes,
        mock_delete_entry, mock_ingest, mock_stats, mock_close, tmp_path, monkeypatch,
    ):
        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nJournal entry content.",
        })
        db_path = str(tmp_path / "manifest.sqlite")
        monkeypatch.setattr("app.graph_ingest.CORPUS_PATH", str(corpus))

        from app.graph_ingest import run_ingest_full
        run_ingest_full(db_path)

        mock_clear.assert_called_once()
        mock_ingest.assert_called_once()

        # Verify manifest was populated
        m = IngestManifest(db_path)
        assert m.count() == 1
        assert m.get("1-1-2025.md") is not None
        m.close()

    @patch("app.graph_ingest.close_driver")
    @patch("app.graph_ingest.get_graph_stats", return_value={"nodes": [], "relationships": []})
    @patch("app.graph_ingest.ingest_entry")
    @patch("app.graph_ingest.delete_entry_data")
    @patch("app.graph_ingest.create_indexes")
    @patch("app.graph_ingest.clear_graph")
    @patch("app.graph_ingest.extract_entities", return_value=MOCK_EXTRACTION)
    def test_incremental_no_changes(
        self, mock_extract, mock_clear, mock_indexes,
        mock_delete_entry, mock_ingest, mock_stats, mock_close, tmp_path, monkeypatch,
    ):
        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nContent.",
        })
        db_path = str(tmp_path / "manifest.sqlite")
        monkeypatch.setattr("app.graph_ingest.CORPUS_PATH", str(corpus))

        from app.graph_ingest import run_ingest_full, run_ingest_incremental

        run_ingest_full(db_path)
        mock_ingest.reset_mock()
        mock_clear.reset_mock()

        run_ingest_incremental(db_path)

        mock_clear.assert_not_called()  # incremental never clears
        mock_ingest.assert_not_called()  # nothing to ingest

    @patch("app.graph_ingest.close_driver")
    @patch("app.graph_ingest.get_graph_stats", return_value={"nodes": [], "relationships": []})
    @patch("app.graph_ingest.ingest_entry")
    @patch("app.graph_ingest.delete_entry_data")
    @patch("app.graph_ingest.create_indexes")
    @patch("app.graph_ingest.clear_graph")
    @patch("app.graph_ingest.extract_entities", return_value=MOCK_EXTRACTION)
    def test_incremental_detects_new_file(
        self, mock_extract, mock_clear, mock_indexes,
        mock_delete_entry, mock_ingest, mock_stats, mock_close, tmp_path, monkeypatch,
    ):
        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nFirst.",
        })
        db_path = str(tmp_path / "manifest.sqlite")
        monkeypatch.setattr("app.graph_ingest.CORPUS_PATH", str(corpus))

        from app.graph_ingest import run_ingest_full, run_ingest_incremental

        run_ingest_full(db_path)
        mock_ingest.reset_mock()

        # Add new file
        (corpus / "1-2-2025.md").write_text("# New\n\nNew entry.")

        run_ingest_incremental(db_path)

        # Only the new file should be ingested
        assert mock_ingest.call_count == 1
        mock_delete_entry.assert_not_called()  # no deletions

    @patch("app.graph_ingest.close_driver")
    @patch("app.graph_ingest.get_graph_stats", return_value={"nodes": [], "relationships": []})
    @patch("app.graph_ingest.ingest_entry")
    @patch("app.graph_ingest.delete_entry_data")
    @patch("app.graph_ingest.create_indexes")
    @patch("app.graph_ingest.clear_graph")
    @patch("app.graph_ingest.extract_entities", return_value=MOCK_EXTRACTION)
    def test_incremental_detects_deleted_file(
        self, mock_extract, mock_clear, mock_indexes,
        mock_delete_entry, mock_ingest, mock_stats, mock_close, tmp_path, monkeypatch,
    ):
        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nTo delete.",
            "1-2-2025.md": "# Entry\n\nTo keep.",
        })
        db_path = str(tmp_path / "manifest.sqlite")
        monkeypatch.setattr("app.graph_ingest.CORPUS_PATH", str(corpus))

        from app.graph_ingest import run_ingest_full, run_ingest_incremental

        run_ingest_full(db_path)
        mock_ingest.reset_mock()
        mock_delete_entry.reset_mock()

        # Delete a file
        (corpus / "1-1-2025.md").unlink()

        run_ingest_incremental(db_path)

        mock_delete_entry.assert_called_once_with("1-1-2025.md")
        mock_ingest.assert_not_called()  # no new files to ingest

    @patch("app.graph_ingest.close_driver")
    @patch("app.graph_ingest.get_graph_stats", return_value={"nodes": [], "relationships": []})
    @patch("app.graph_ingest.ingest_entry")
    @patch("app.graph_ingest.delete_entry_data")
    @patch("app.graph_ingest.create_indexes")
    @patch("app.graph_ingest.clear_graph")
    @patch("app.graph_ingest.extract_entities", return_value=MOCK_EXTRACTION)
    def test_incremental_dry_run(
        self, mock_extract, mock_clear, mock_indexes,
        mock_delete_entry, mock_ingest, mock_stats, mock_close, tmp_path, monkeypatch,
    ):
        corpus = _make_corpus(tmp_path, {
            "1-1-2025.md": "# Entry\n\nContent.",
        })
        db_path = str(tmp_path / "manifest.sqlite")
        monkeypatch.setattr("app.graph_ingest.CORPUS_PATH", str(corpus))

        from app.graph_ingest import run_ingest_full, run_ingest_incremental

        run_ingest_full(db_path)
        mock_ingest.reset_mock()

        # Add new file
        (corpus / "1-2-2025.md").write_text("# New\n\nNew entry.")

        run_ingest_incremental(db_path, dry_run=True)

        mock_ingest.assert_not_called()  # dry run, no actual processing
        mock_delete_entry.assert_not_called()
