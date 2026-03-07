"""Unit tests for feedback auto-tuning, aggregation, and health."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from app.feedback import FeedbackStore, _empty_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path, state: dict | None = None) -> FeedbackStore:
    """Create a FeedbackStore backed by a file in tmp_path."""
    path = tmp_path / "feedback.json"
    if state is not None:
        path.write_text(json.dumps(state), encoding="utf-8")
    return FeedbackStore(str(path))


def _store_with_concepts(tmp_path: Path, concept_weights: dict[str, float]) -> FeedbackStore:
    """Shortcut: store whose only non-trivial data is concept weights."""
    state = _empty_store()
    state["weights"]["concept"] = dict(concept_weights)
    return _make_store(tmp_path, state)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# compute_tuning
# ---------------------------------------------------------------------------

class TestComputeTuning:
    def test_empty_weights(self, tmp_path):
        store = _make_store(tmp_path)
        result = store.compute_tuning()
        assert result["graduated_count"] == 0
        assert result["concept_boosts"] == {}
        assert result["skipped"] == []

    def test_below_threshold_skipped(self, tmp_path):
        store = _store_with_concepts(tmp_path, {"mild": 0.4, "weak": -0.3})
        result = store.compute_tuning(stability_threshold=0.8)
        assert result["graduated_count"] == 0
        assert set(result["skipped"]) == {"mild", "weak"}

    def test_above_threshold_graduated(self, tmp_path):
        store = _store_with_concepts(tmp_path, {"strong": 1.2, "negative": -1.0, "weak": 0.3})
        result = store.compute_tuning(stability_threshold=0.8, scale=0.25)
        assert result["graduated_count"] == 2
        assert "strong" in result["concept_boosts"]
        assert "negative" in result["concept_boosts"]
        assert "weak" not in result["concept_boosts"]

    def test_scale_factor(self, tmp_path):
        store = _store_with_concepts(tmp_path, {"topic": 2.0})
        result = store.compute_tuning(stability_threshold=0.5, scale=0.5)
        assert result["concept_boosts"]["topic"] == 1.0

    def test_custom_threshold(self, tmp_path):
        store = _store_with_concepts(tmp_path, {"a": 0.5, "b": 1.5})
        result = store.compute_tuning(stability_threshold=0.4)
        assert result["graduated_count"] == 2


# ---------------------------------------------------------------------------
# apply_tuning
# ---------------------------------------------------------------------------

class TestApplyTuning:
    def test_writes_valid_json(self, tmp_path):
        store = _store_with_concepts(tmp_path, {"stable": 1.2})
        tuning_path = str(tmp_path / "tuning.json")
        result = store.apply_tuning(tuning_path)

        assert result["ok"] is True
        assert result["graduated_count"] == 1
        written = json.loads(Path(tuning_path).read_text(encoding="utf-8"))
        assert "concept_boosts" in written
        assert "stable" in written["concept_boosts"]
        assert "last_tuning" in written

    def test_merges_with_existing(self, tmp_path):
        tuning_path = tmp_path / "tuning.json"
        tuning_path.write_text(
            json.dumps({"concept_boosts": {"old_term": 0.5}}),
            encoding="utf-8",
        )
        store = _store_with_concepts(tmp_path, {"new_term": 1.0})
        store.apply_tuning(str(tuning_path))

        written = json.loads(tuning_path.read_text(encoding="utf-8"))
        assert "old_term" in written["concept_boosts"]
        assert "new_term" in written["concept_boosts"]

    def test_atomic_write_no_partial(self, tmp_path):
        store = _store_with_concepts(tmp_path, {"term": 1.0})
        tuning_path = str(tmp_path / "tuning.json")
        store.apply_tuning(tuning_path)
        # No .tmp file should remain
        assert not Path(tuning_path + ".tmp").exists()

    def test_records_metadata(self, tmp_path):
        store = _store_with_concepts(tmp_path, {"term": 1.0})
        tuning_path = str(tmp_path / "tuning.json")
        store.apply_tuning(tuning_path)
        assert store._state.get("last_tuning") is not None

    def test_no_graduates_still_writes(self, tmp_path):
        store = _store_with_concepts(tmp_path, {"weak": 0.1})
        tuning_path = str(tmp_path / "tuning.json")
        result = store.apply_tuning(tuning_path)
        assert result["graduated_count"] == 0
        assert Path(tuning_path).exists()


# ---------------------------------------------------------------------------
# aggregate_events
# ---------------------------------------------------------------------------

class TestAggregateEvents:
    def _make_store_with_events(self, tmp_path: Path, n: int) -> FeedbackStore:
        state = _empty_store()
        state["events"] = n
        for i in range(n):
            state["event_log"].append({
                "id": i + 1,
                "timestamp": _now_iso(),
                "signal": "up" if i % 2 == 0 else "down",
                "query": f"q{i}",
                "note": None,
                "concepts": [],
                "people": [],
                "places": [],
                "sources": [],
            })
        return _make_store(tmp_path, state)

    def test_archives_correct_count(self, tmp_path):
        store = self._make_store_with_events(tmp_path, 50)
        result = store.aggregate_events(
            keep_recent=20,
            archive_path=str(tmp_path / "archive.jsonl"),
        )
        assert result["archived"] == 30
        assert result["remaining"] == 20

    def test_jsonl_format(self, tmp_path):
        store = self._make_store_with_events(tmp_path, 10)
        archive = tmp_path / "archive.jsonl"
        store.aggregate_events(keep_recent=5, archive_path=str(archive))
        lines = archive.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert "id" in parsed

    def test_remaining_events_preserved(self, tmp_path):
        store = self._make_store_with_events(tmp_path, 20)
        store.aggregate_events(
            keep_recent=10,
            archive_path=str(tmp_path / "archive.jsonl"),
        )
        assert len(store._state["event_log"]) == 10
        # Remaining should be the last 10 (newest)
        ids = [e["id"] for e in store._state["event_log"]]
        assert ids == list(range(11, 21))

    def test_no_op_when_under_limit(self, tmp_path):
        store = self._make_store_with_events(tmp_path, 5)
        result = store.aggregate_events(
            keep_recent=10,
            archive_path=str(tmp_path / "archive.jsonl"),
        )
        assert result["archived"] == 0
        assert result["remaining"] == 5

    def test_records_aggregation_timestamp(self, tmp_path):
        store = self._make_store_with_events(tmp_path, 30)
        store.aggregate_events(
            keep_recent=10,
            archive_path=str(tmp_path / "archive.jsonl"),
        )
        assert store._state.get("last_aggregation") is not None


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_correct_counts(self, tmp_path):
        state = _empty_store()
        state["events"] = 10
        for i in range(10):
            state["event_log"].append({
                "id": i + 1,
                "timestamp": _now_iso(),
                "signal": "up" if i < 6 else "down",
                "query": "",
                "note": None,
                "concepts": [],
                "people": [],
                "places": [],
                "sources": [],
            })
        state["weights"]["concept"]["clarity"] = 1.2
        state["weights"]["concept"]["noise"] = -0.8
        store = _make_store(tmp_path, state)

        h = store.health()
        assert h["total_events"] == 10
        assert h["active_events"] == 10
        assert h["up_count"] == 6
        assert h["down_count"] == 4
        assert h["concepts_affected"] == 2

    def test_velocity_calculation(self, tmp_path):
        state = _empty_store()
        state["events"] = 7
        now = datetime.now(timezone.utc)
        for i in range(7):
            state["event_log"].append({
                "id": i + 1,
                "timestamp": (now - timedelta(days=i)).isoformat(),
                "signal": "up",
                "query": "",
                "note": None,
                "concepts": [],
                "people": [],
                "places": [],
                "sources": [],
            })
        store = _make_store(tmp_path, state)
        h = store.health()
        assert h["velocity_7d"] == 1.0  # 7 events / 7 days

    def test_zero_events(self, tmp_path):
        store = _make_store(tmp_path)
        h = store.health()
        assert h["total_events"] == 0
        assert h["active_events"] == 0
        assert h["velocity_7d"] == 0.0
        assert h["last_tuning"] is None


# ---------------------------------------------------------------------------
# _maybe_auto_tune
# ---------------------------------------------------------------------------

class TestMaybeAutoTune:
    def test_triggers_at_interval(self, tmp_path):
        import app.feedback as fb

        store = _store_with_concepts(tmp_path, {"test": 1.5})
        store._state["events"] = 10  # 10 % 5 == 0

        tuning_path = str(tmp_path / "tuning.json")
        with patch.object(fb, "GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL", 5):
            with patch.object(fb, "GRAPH_AUTO_TUNING_PATH", tuning_path):
                with patch.object(store, "apply_tuning") as mock_apply:
                    store._maybe_auto_tune()
                    mock_apply.assert_called_once_with(tuning_path)

    def test_skips_when_not_at_interval(self, tmp_path):
        import app.feedback as fb

        store = _store_with_concepts(tmp_path, {"test": 1.5})
        store._state["events"] = 7  # 7 % 5 != 0

        with patch.object(fb, "GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL", 5):
            with patch.object(store, "apply_tuning") as mock_apply:
                store._maybe_auto_tune()
                mock_apply.assert_not_called()

    def test_disabled_when_zero(self, tmp_path):
        import app.feedback as fb

        store = _store_with_concepts(tmp_path, {"test": 1.5})
        store._state["events"] = 50

        with patch.object(fb, "GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL", 0):
            with patch.object(store, "apply_tuning") as mock_apply:
                store._maybe_auto_tune()
                mock_apply.assert_not_called()

    def test_calls_reload_callback(self, tmp_path):
        import app.feedback as fb

        store = _store_with_concepts(tmp_path, {"test": 1.5})
        store._state["events"] = 10
        callback_called = []
        store.set_reload_callback(lambda: callback_called.append(True))

        tuning_path = str(tmp_path / "tuning.json")
        with patch.object(fb, "GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL", 5):
            with patch.object(fb, "GRAPH_AUTO_TUNING_PATH", tuning_path):
                store._maybe_auto_tune()
        assert len(callback_called) == 1


# ---------------------------------------------------------------------------
# Style profile merge
# ---------------------------------------------------------------------------

class TestStyleProfileMerge:
    def test_manual_boosts_preserved(self, tmp_path):
        tuning = {"concept_boosts": {"auto_term": 0.5, "shared": 0.3}}
        tuning_path = tmp_path / "tuning.json"
        tuning_path.write_text(json.dumps(tuning), encoding="utf-8")

        with patch("app.style_profile.AUTO_TUNING_PATH", str(tuning_path)):
            with patch("app.style_profile.STYLE_PROFILE_JSON", json.dumps({
                "concept_boosts": {"shared": 0.9, "manual_only": 0.7}
            })):
                from app.style_profile import get_style_profile
                get_style_profile.cache_clear()
                profile = get_style_profile()

                # Manual value wins for 'shared'
                assert profile["concept_boosts"]["shared"] == 0.9
                # Auto-tuned value present
                assert profile["concept_boosts"]["auto_term"] == 0.5
                # Manual-only present
                assert profile["concept_boosts"]["manual_only"] == 0.7

                get_style_profile.cache_clear()

    def test_missing_tuning_file_no_op(self, tmp_path):
        with patch("app.style_profile.AUTO_TUNING_PATH", str(tmp_path / "nonexistent.json")):
            with patch("app.style_profile.STYLE_PROFILE_JSON", None):
                with patch("app.style_profile.STYLE_PROFILE_PATH", str(tmp_path / "nope.json")):
                    from app.style_profile import get_style_profile
                    get_style_profile.cache_clear()
                    profile = get_style_profile()
                    assert profile["concept_boosts"] == {}
                    get_style_profile.cache_clear()

    def test_corrupt_tuning_file_handled(self, tmp_path):
        tuning_path = tmp_path / "tuning.json"
        tuning_path.write_text("NOT VALID JSON", encoding="utf-8")

        with patch("app.style_profile.AUTO_TUNING_PATH", str(tuning_path)):
            with patch("app.style_profile.STYLE_PROFILE_JSON", None):
                with patch("app.style_profile.STYLE_PROFILE_PATH", str(tmp_path / "nope.json")):
                    from app.style_profile import get_style_profile
                    get_style_profile.cache_clear()
                    profile = get_style_profile()
                    # Should not crash, just no auto boosts
                    assert profile["concept_boosts"] == {}
                    get_style_profile.cache_clear()


# ---------------------------------------------------------------------------
# _append_event overflow archiving
# ---------------------------------------------------------------------------

class TestAppendEventArchive:
    def test_overflow_archived_not_dropped(self, tmp_path):
        state = _empty_store()
        store = _make_store(tmp_path, state)
        archive_path = tmp_path / "archive.jsonl"

        with patch("app.feedback.GRAPH_FEEDBACK_MAX_EVENTS", 5):
            with patch("app.feedback.GRAPH_FEEDBACK_ARCHIVE_PATH", str(archive_path)):
                # Add 7 events
                for i in range(7):
                    store._append_event({
                        "id": i + 1,
                        "timestamp": _now_iso(),
                        "signal": "up",
                    })

        # Should have 5 events in log (max)
        assert len(store._state["event_log"]) == 5
        # Overflow should be archived
        assert archive_path.exists()
        lines = archive_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2  # 7 - 5 = 2 archived


# ---------------------------------------------------------------------------
# reload_style_config
# ---------------------------------------------------------------------------

class TestReloadStyleConfig:
    def test_clears_cache(self, tmp_path):
        from app.style_profile import get_style_profile, reload_style_config

        # Call to populate cache
        get_style_profile.cache_clear()

        with patch("app.style_profile.STYLE_PROFILE_JSON", None):
            with patch("app.style_profile.STYLE_PROFILE_PATH", str(tmp_path / "nope.json")):
                with patch("app.style_profile.AUTO_TUNING_PATH", str(tmp_path / "nope2.json")):
                    p1 = get_style_profile()
                    # Should be cached
                    p2 = get_style_profile()
                    assert p1 is p2

                    # After reload, should be a new object
                    reload_style_config()
                    p3 = get_style_profile()
                    assert p3 is not p1

                    get_style_profile.cache_clear()
