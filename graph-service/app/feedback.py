"""Feedback capture and reranking for GraphRAG retrieval."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

GRAPH_FEEDBACK_STORE_PATH = os.environ.get(
    "GRAPH_FEEDBACK_STORE_PATH", "/service/data/graphrag_feedback.json"
)
GRAPH_FEEDBACK_MAX_TERMS = max(1, int(os.environ.get("GRAPH_FEEDBACK_MAX_TERMS", "2000")))
GRAPH_FEEDBACK_MAX_EVENTS = max(10, int(os.environ.get("GRAPH_FEEDBACK_MAX_EVENTS", "1500")))
GRAPH_FEEDBACK_TERM_DELTA = float(os.environ.get("GRAPH_FEEDBACK_TERM_DELTA", "0.4"))
GRAPH_FEEDBACK_QUERY_DELTA = float(os.environ.get("GRAPH_FEEDBACK_QUERY_DELTA", "0.2"))
GRAPH_AUTO_TUNING_PATH = os.environ.get(
    "GRAPH_AUTO_TUNING_PATH", "/service/data/feedback_tuning.json"
)
GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL = int(os.environ.get("GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL", "50"))
GRAPH_FEEDBACK_ARCHIVE_PATH = os.environ.get(
    "GRAPH_FEEDBACK_ARCHIVE_PATH", "/service/data/feedback_archive.jsonl"
)

CATEGORIES = ("concept", "person", "place", "source")
TOKEN_RE = re.compile(r"[a-z0-9']+")


def _empty_store() -> dict[str, Any]:
    return {
        "version": 2,
        "events": 0,
        "weights": {
            "concept": {},
            "person": {},
            "place": {},
            "source": {},
        },
        "event_log": [],
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_term(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalize_terms(values: list[str] | None) -> list[str]:
    if not values:
        return []

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        term = _normalize_term(value)
        if not term or term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def _normalize_note(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    return cleaned[:2000]


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _contains_term(text_normalized: str, tokens: set[str], term: str) -> bool:
    if " " in term:
        return term in text_normalized
    return term in tokens


class FeedbackStore:
    def __init__(self, path_value: str):
        self.path = Path(path_value)
        self._lock = Lock()
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_store()

        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            logger.warning("Failed to read feedback store %s: %s", self.path, err)
            return _empty_store()

        if not isinstance(parsed, dict):
            return _empty_store()

        state = _empty_store()
        state["events"] = int(parsed.get("events", 0))

        raw_weights = parsed.get("weights", {})
        if isinstance(raw_weights, dict):
            for category in CATEGORIES:
                raw_category = raw_weights.get(category, {})
                if not isinstance(raw_category, dict):
                    continue
                normalized: dict[str, float] = {}
                for raw_key, raw_value in raw_category.items():
                    if not isinstance(raw_key, str):
                        continue
                    key = _normalize_term(raw_key)
                    if not key:
                        continue
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    normalized[key] = _clamp(value, -3.0, 3.0)
                state["weights"][category] = normalized

        raw_events = parsed.get("event_log", [])
        if isinstance(raw_events, list):
            normalized_events: list[dict[str, Any]] = []
            for entry in raw_events[-GRAPH_FEEDBACK_MAX_EVENTS:]:
                if not isinstance(entry, dict):
                    continue
                signal = str(entry.get("signal", "")).strip().lower()
                if signal not in {"up", "down"}:
                    continue
                normalized_events.append(
                    {
                        "id": int(entry.get("id", 0) or 0),
                        "timestamp": str(entry.get("timestamp", "")),
                        "signal": signal,
                        "query": str(entry.get("query", "") or ""),
                        "note": _normalize_note(entry.get("note") if isinstance(entry.get("note"), str) else None),
                        "concepts": _normalize_terms(entry.get("concepts") if isinstance(entry.get("concepts"), list) else None),
                        "people": _normalize_terms(entry.get("people") if isinstance(entry.get("people"), list) else None),
                        "places": _normalize_terms(entry.get("places") if isinstance(entry.get("places"), list) else None),
                        "sources": _normalize_terms(entry.get("sources") if isinstance(entry.get("sources"), list) else None),
                    }
                )
            state["event_log"] = normalized_events

        return state

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def _trim_category(self, category: str) -> None:
        weights = self._state["weights"][category]
        if len(weights) <= GRAPH_FEEDBACK_MAX_TERMS:
            return

        sorted_terms = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
        kept = dict(sorted_terms[:GRAPH_FEEDBACK_MAX_TERMS])
        self._state["weights"][category] = kept

    def _update_terms(self, category: str, terms: list[str], delta: float) -> None:
        if not terms:
            return

        weights = self._state["weights"][category]
        for term in terms:
            current = float(weights.get(term, 0.0))
            updated = _clamp(current + delta, -3.0, 3.0)
            if abs(updated) < 0.01:
                weights.pop(term, None)
            else:
                weights[term] = round(updated, 4)

        self._trim_category(category)

    def _append_event(self, event: dict[str, Any]) -> None:
        event_log = self._state.setdefault("event_log", [])
        if not isinstance(event_log, list):
            event_log = []
            self._state["event_log"] = event_log
        event_log.append(event)
        if len(event_log) > GRAPH_FEEDBACK_MAX_EVENTS:
            overflow = event_log[: len(event_log) - GRAPH_FEEDBACK_MAX_EVENTS]
            self._archive_events(overflow)
            del event_log[: len(event_log) - GRAPH_FEEDBACK_MAX_EVENTS]

    def _archive_events(self, events: list[dict[str, Any]], archive_path_str: str | None = None) -> None:
        """Append overflow events to the JSONL archive file."""
        if not events:
            return
        archive_path = Path(archive_path_str or GRAPH_FEEDBACK_ARCHIVE_PATH)
        try:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with open(archive_path, "a", encoding="utf-8") as fh:
                for event in events:
                    fh.write(json.dumps(event, sort_keys=True) + "\n")
            archived_total = int(self._state.get("archived_events", 0))
            self._state["archived_events"] = archived_total + len(events)
        except OSError as err:
            logger.warning("Failed to write feedback archive %s: %s", archive_path, err)

    def record_feedback(
        self,
        *,
        signal: str,
        query: str | None = None,
        note: str | None = None,
        concepts: list[str] | None = None,
        people: list[str] | None = None,
        places: list[str] | None = None,
        sources: list[str] | None = None,
        query_entities: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        direction = signal.lower().strip()
        if direction not in {"up", "down"}:
            raise ValueError("signal must be 'up' or 'down'")

        delta = GRAPH_FEEDBACK_TERM_DELTA if direction == "up" else -GRAPH_FEEDBACK_TERM_DELTA
        query_delta = GRAPH_FEEDBACK_QUERY_DELTA if direction == "up" else -GRAPH_FEEDBACK_QUERY_DELTA

        normalized = {
            "concept": _normalize_terms(concepts),
            "person": _normalize_terms(people),
            "place": _normalize_terms(places),
            "source": _normalize_terms(sources),
        }

        query_entities = query_entities or {}
        query_terms = {
            "concept": _normalize_terms(query_entities.get("concepts")),
            "person": _normalize_terms(query_entities.get("people")),
            "place": _normalize_terms(query_entities.get("places")),
        }

        normalized_query = " ".join(query.strip().split()) if isinstance(query, str) else ""
        normalized_note = _normalize_note(note)

        with self._lock:
            for category, terms in normalized.items():
                self._update_terms(category, terms, delta)

            for category, terms in query_terms.items():
                self._update_terms(category, terms, query_delta)

            self._state["events"] = int(self._state.get("events", 0)) + 1
            event_id = self._state["events"]
            self._append_event(
                {
                    "id": event_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signal": direction,
                    "query": normalized_query,
                    "note": normalized_note,
                    "concepts": normalized["concept"],
                    "people": normalized["person"],
                    "places": normalized["place"],
                    "sources": normalized["source"],
                }
            )

            self._save()
            self._maybe_auto_tune()

            applied_terms = {
                category: terms
                for category, terms in normalized.items()
                if terms
            }

            return {
                "ok": True,
                "signal": direction,
                "events": self._state["events"],
                "event_id": event_id,
                "applied": applied_terms,
                "note_saved": normalized_note is not None,
            }

    def get_event_log(
        self,
        *,
        limit: int = 200,
        signal: str | None = None,
        notes_only: bool = False,
    ) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), GRAPH_FEEDBACK_MAX_EVENTS))

        with self._lock:
            event_log = self._state.get("event_log", [])
            if not isinstance(event_log, list):
                return []

            selected = event_log
            if signal in {"up", "down"}:
                selected = [e for e in selected if e.get("signal") == signal]
            if notes_only:
                selected = [e for e in selected if isinstance(e.get("note"), str) and e.get("note")]

            result = list(selected[-bounded:])
            result.reverse()
            return [dict(item) for item in result]

    def profile(self, top_n: int = 15) -> dict[str, Any]:
        bounded = max(1, min(int(top_n), 100))

        with self._lock:
            weights = self._state["weights"]
            event_log = self._state.get("event_log", [])
            if not isinstance(event_log, list):
                event_log = []

            off_target = sum(1 for item in event_log if item.get("signal") == "down")
            helpful = sum(1 for item in event_log if item.get("signal") == "up")
            with_notes = sum(
                1
                for item in event_log
                if item.get("signal") == "down" and isinstance(item.get("note"), str) and item.get("note")
            )

            profile: dict[str, Any] = {
                "events": self._state["events"],
                "summary": {
                    "helpful": helpful,
                    "off_target": off_target,
                    "off_target_with_notes": with_notes,
                },
                "top_positive": {},
                "top_negative": {},
            }
            for category in CATEGORIES:
                entries = list(weights[category].items())
                positive = [item for item in entries if item[1] > 0]
                negative = [item for item in entries if item[1] < 0]
                positive.sort(key=lambda item: item[1], reverse=True)
                negative.sort(key=lambda item: item[1])
                profile["top_positive"][category] = [
                    {"term": term, "weight": weight}
                    for term, weight in positive[:bounded]
                ]
                profile["top_negative"][category] = [
                    {"term": term, "weight": weight}
                    for term, weight in negative[:bounded]
                ]
            return profile

    def rerank_vector_results(
        self,
        results: list[dict[str, Any]],
        query_entities: dict[str, list[str]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not results:
            return results, {"applied": False, "reason": "no_results"}

        query_entities = query_entities or {}

        with self._lock:
            weights = self._state["weights"]
            no_weights = all(len(weights[category]) == 0 for category in CATEGORIES)
            if no_weights:
                return results, {"applied": False, "reason": "empty_profile"}

            reranked: list[dict[str, Any]] = []
            for item in results:
                scored = dict(item)
                base = float(item.get("relevance_score") or 0.0)
                source = _normalize_term(str(item.get("source_file", "")))
                text_norm = _normalize_term(str(item.get("text", "")))
                tokens = _tokenize(text_norm)

                adjustment = 0.0

                for term, weight in weights["concept"].items():
                    if _contains_term(text_norm, tokens, term):
                        adjustment += 0.07 * weight

                for term, weight in weights["person"].items():
                    if _contains_term(text_norm, tokens, term):
                        adjustment += 0.08 * weight

                for term, weight in weights["place"].items():
                    if _contains_term(text_norm, tokens, term):
                        adjustment += 0.06 * weight

                if source:
                    for term, weight in weights["source"].items():
                        if term in source:
                            adjustment += 0.1 * weight

                for term in _normalize_terms(query_entities.get("concepts")):
                    adjustment += 0.04 * float(weights["concept"].get(term, 0.0))

                for term in _normalize_terms(query_entities.get("people")):
                    adjustment += 0.05 * float(weights["person"].get(term, 0.0))

                for term in _normalize_terms(query_entities.get("places")):
                    adjustment += 0.04 * float(weights["place"].get(term, 0.0))

                clipped = _clamp(adjustment, -0.55, 0.55)
                tuned = max(0.0, base + clipped)

                scored["base_relevance_score"] = round(base, 4)
                scored["feedback_adjustment"] = round(clipped, 4)
                scored["relevance_score"] = round(tuned, 4)
                reranked.append(scored)

            reranked.sort(
                key=lambda item: (
                    float(item.get("relevance_score", 0.0)),
                    str(item.get("date", "")),
                ),
                reverse=True,
            )

            return reranked, {
                "applied": True,
                "events": self._state["events"],
                "profile_terms": {
                    category: len(weights[category]) for category in CATEGORIES
                },
            }

    def rerank_graph_context(self, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not context:
            return context

        with self._lock:
            weights = self._state["weights"]

            scored: list[tuple[float, int, dict[str, Any]]] = []
            for idx, item in enumerate(context):
                entity = _normalize_term(str(item.get("entity", "")))
                ctx_type = str(item.get("type", ""))

                score = 0.0
                if ctx_type == "person_context":
                    score += float(weights["person"].get(entity, 0.0))
                elif ctx_type == "concept_context":
                    score += float(weights["concept"].get(entity, 0.0))

                score += 0.2 * len(item.get("data", []))
                scored.append((score, idx, item))

            scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
            return [row[2] for row in scored]

    # --- Auto-tuning ---

    def compute_tuning(
        self,
        *,
        stability_threshold: float = 0.8,
        scale: float = 0.25,
    ) -> dict[str, Any]:
        """Identify stable concept weights and convert to concept_boosts."""
        with self._lock:
            concept_weights = self._state["weights"].get("concept", {})
            graduated: dict[str, float] = {}
            skipped: list[str] = []

            for term, weight in concept_weights.items():
                if abs(weight) >= stability_threshold:
                    graduated[term] = round(weight * scale, 4)
                else:
                    skipped.append(term)

            return {
                "concept_boosts": graduated,
                "graduated_count": len(graduated),
                "skipped": skipped,
            }

    def apply_tuning(self, tuning_path: str) -> dict[str, Any]:
        """Write graduated concept_boosts to the auto-tuning file."""
        tuning = self.compute_tuning()
        graduated = tuning["concept_boosts"]

        path = Path(tuning_path)
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except (OSError, json.JSONDecodeError):
                existing = {}

        existing_boosts = existing.get("concept_boosts", {})
        if not isinstance(existing_boosts, dict):
            existing_boosts = {}
        existing_boosts.update(graduated)
        existing["concept_boosts"] = existing_boosts

        now_iso = datetime.now(timezone.utc).isoformat()
        existing["last_tuning"] = now_iso
        existing["graduated_count"] = tuning["graduated_count"]

        # Atomic write
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp_path.replace(path)

        # Record metadata in feedback store
        with self._lock:
            self._state["last_tuning"] = now_iso
            self._save()

        return {
            "ok": True,
            "graduated_count": tuning["graduated_count"],
            "concept_boosts": graduated,
            "tuning_path": str(path),
        }

    def aggregate_events(
        self,
        *,
        keep_recent: int = 1000,
        archive_path: str | None = None,
    ) -> dict[str, Any]:
        """Archive oldest events to JSONL, keep recent ones in active log."""
        resolved_path = archive_path or GRAPH_FEEDBACK_ARCHIVE_PATH

        with self._lock:
            event_log = self._state.get("event_log", [])
            if not isinstance(event_log, list):
                return {"archived": 0, "remaining": 0}

            if len(event_log) <= keep_recent:
                return {"archived": 0, "remaining": len(event_log)}

            to_archive = event_log[: len(event_log) - keep_recent]
            self._archive_events(to_archive, resolved_path)
            del event_log[: len(event_log) - keep_recent]

            self._state["last_aggregation"] = datetime.now(timezone.utc).isoformat()
            self._save()

            return {"archived": len(to_archive), "remaining": len(event_log)}

    def health(self) -> dict[str, Any]:
        """Return feedback health metrics."""
        with self._lock:
            event_log = self._state.get("event_log", [])
            if not isinstance(event_log, list):
                event_log = []

            total_events = int(self._state.get("events", 0))
            active_events = len(event_log)
            archived_events = int(self._state.get("archived_events", 0))
            up_count = sum(1 for e in event_log if e.get("signal") == "up")
            down_count = sum(1 for e in event_log if e.get("signal") == "down")

            concepts_affected = sum(
                1
                for weight in self._state["weights"].get("concept", {}).values()
                if abs(weight) > 0.01
            )

            # Velocity: events per day over last 7 days
            velocity_7d = 0.0
            now = datetime.now(timezone.utc)
            seven_days_ago = now.timestamp() - (7 * 86400)
            recent_count = 0
            for event in event_log:
                ts_str = event.get("timestamp", "")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts.timestamp() >= seven_days_ago:
                        recent_count += 1
                except (ValueError, TypeError):
                    continue
            if recent_count > 0:
                velocity_7d = round(recent_count / 7.0, 2)

            return {
                "total_events": total_events,
                "active_events": active_events,
                "archived_events": archived_events,
                "up_count": up_count,
                "down_count": down_count,
                "concepts_affected": concepts_affected,
                "last_tuning": self._state.get("last_tuning"),
                "last_aggregation": self._state.get("last_aggregation"),
                "velocity_7d": velocity_7d,
            }

    def _maybe_auto_tune(self) -> None:
        """Auto-apply tuning every GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL events."""
        if GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL <= 0:
            return
        total = int(self._state.get("events", 0))
        if total == 0 or total % GRAPH_FEEDBACK_AUTO_TUNE_INTERVAL != 0:
            return
        try:
            self.apply_tuning(GRAPH_AUTO_TUNING_PATH)
            if self._reload_callback is not None:
                self._reload_callback()
        except Exception as err:
            logger.warning("Auto-tune failed at event %d: %s", total, err)

    _reload_callback: Any = None

    def set_reload_callback(self, callback: Any) -> None:
        """Register a callback to invoke after auto-tuning applies."""
        self._reload_callback = callback


_store: FeedbackStore | None = None


def get_feedback_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore(GRAPH_FEEDBACK_STORE_PATH)
    return _store
