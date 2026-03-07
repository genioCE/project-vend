"""Generate actionable language-improvement prompts from GraphRAG feedback notes."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


TAG_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "redundancy",
        re.compile(r"\b(redundan|repeat|repetit|same point|restate|duplicate|looping?)\b", re.IGNORECASE),
    ),
    (
        "evidence_dates",
        re.compile(r"\b(date|dated|timeline|timestamp|evidence|proof|cite|citation|source)\b", re.IGNORECASE),
    ),
    (
        "specificity",
        re.compile(r"\b(vague|generic|broad|specific|concrete|detail|shallow|surface)\b", re.IGNORECASE),
    ),
    (
        "voice_alignment",
        re.compile(r"\b(voice|tone|sound|style|language|flow|cadence|feels like me|my writing)\b", re.IGNORECASE),
    ),
    (
        "graph_focus",
        re.compile(r"\b(graph|node|edge|relationship|connected|entity|concept map|network)\b", re.IGNORECASE),
    ),
    (
        "brevity",
        re.compile(r"\b(too long|wordy|verbose|concise|shorter|trim|tighten)\b", re.IGNORECASE),
    ),
    (
        "contradictions",
        re.compile(r"\b(contradict|tension|conflict|inconsisten|ambivalen|uncertain|mixed)\b", re.IGNORECASE),
    ),
]

PROMPT_LIBRARY: dict[str, dict[str, str]] = {
    "redundancy": {
        "title": "Reduce semantic repetition",
        "instructions": (
            "Update GraphRAG answer generation to collapse overlapping claims into one evidence point. "
            "Tighten anti-redundancy language in `mcp-server/src/agent.ts` (`GRAPHRAG_SYSTEM_PROMPT`) and add a "
            "dedupe pass for repeated evidence blocks in `graph-service/app/graphrag.py` before `formatted_context` is returned."
        ),
    },
    "evidence_dates": {
        "title": "Increase dated evidence density",
        "instructions": (
            "Improve answer grounding so claims are tied to concrete dates and sources whenever available. "
            "Adjust GraphRAG formatting and prompt guidance to prefer date-anchored bullets and avoid ungrounded summaries. "
            "Touch `graph-service/app/graphrag.py` and `mcp-server/src/agent.ts`."
        ),
    },
    "specificity": {
        "title": "Make interpretations more specific",
        "instructions": (
            "Reduce generic language and require concrete, non-overlapping observations tied to retrieved entries. "
            "Strengthen prompt rules in `mcp-server/src/agent.ts` and, if needed, enrich `formatted_context` in "
            "`graph-service/app/graphrag.py` with sharper evidence labels."
        ),
    },
    "voice_alignment": {
        "title": "Align output language with writer voice",
        "instructions": (
            "Tune response style so phrasing better matches the user’s writing cadence while preserving evidence fidelity. "
            "Use `graph-service/config/style_profile.json` and extraction/scoring hooks in `graph-service/app/extractor.py`; "
            "refine response style guidance in `mcp-server/src/agent.ts`."
        ),
    },
    "graph_focus": {
        "title": "Improve graph-aware reasoning",
        "instructions": (
            "Increase use of explicit graph relationships in final answers (people/concepts/emotions/flows), and reduce purely "
            "vector-only paraphrasing. Update tool-selection and output guidance in `mcp-server/src/agent.ts` and ensure "
            "graph context is surfaced clearly in `graph-service/app/graphrag.py`."
        ),
    },
    "brevity": {
        "title": "Tighten response length",
        "instructions": (
            "Make GraphRAG responses shorter and higher-signal by capping repetitive prose and favoring compact evidence bullets. "
            "Update style constraints in `mcp-server/src/agent.ts` and trim context verbosity in `graph-service/app/graphrag.py`."
        ),
    },
    "contradictions": {
        "title": "Handle tensions/contradictions explicitly",
        "instructions": (
            "When entries conflict, call out the tension directly instead of averaging into a generic summary. "
            "Refine output rules in `mcp-server/src/agent.ts` and optionally add contradiction cues in graph context formatting "
            "within `graph-service/app/graphrag.py`."
        ),
    },
    "general_language": {
        "title": "General language quality refinement",
        "instructions": (
            "Improve GraphRAG language quality based on user feedback. Focus on clarity, non-redundancy, and stronger evidence anchoring. "
            "Apply targeted updates to `mcp-server/src/agent.ts` and `graph-service/app/graphrag.py`."
        ),
    },
}


def _normalize_note(note: str) -> str:
    return " ".join(note.strip().split())


def infer_tags_from_note(note: str) -> list[str]:
    normalized = _normalize_note(note)
    if not normalized:
        return ["general_language"]

    tags: list[str] = []
    for tag, pattern in TAG_RULES:
        if pattern.search(normalized):
            tags.append(tag)

    if not tags:
        tags.append("general_language")

    return tags


def _render_prompt(tag: str, count: int, sample_notes: list[str]) -> str:
    spec = PROMPT_LIBRARY.get(tag, PROMPT_LIBRARY["general_language"])
    samples = "\n".join(f"- {note}" for note in sample_notes[:3]) or "- (no sample note available)"

    return (
        "Task: Improve GraphRAG language quality from feedback backlog.\n"
        f"Theme: {spec['title']} ({count} related reports).\n"
        "\n"
        "What to change:\n"
        f"{spec['instructions']}\n"
        "\n"
        "Representative user notes:\n"
        f"{samples}\n"
        "\n"
        "Acceptance criteria:\n"
        "- Responses avoid repeated paraphrases.\n"
        "- Evidence bullets are specific and non-overlapping.\n"
        "- Date/source grounding appears when available.\n"
        "- Behavior remains stable in both standard and GraphRAG modes."
    )


def suggest_prompts_from_note(note: str, limit: int = 3) -> list[dict[str, Any]]:
    tags = infer_tags_from_note(note)
    bounded = max(1, min(int(limit), 10))

    prompts: list[dict[str, Any]] = []
    for tag in tags[:bounded]:
        spec = PROMPT_LIBRARY.get(tag, PROMPT_LIBRARY["general_language"])
        prompts.append(
            {
                "tag": tag,
                "title": spec["title"],
                "prompt": _render_prompt(tag, 1, [_normalize_note(note)]),
            }
        )

    return prompts


def build_feedback_review(
    events: list[dict[str, Any]],
    *,
    top_n: int = 15,
    recent_limit: int = 40,
) -> dict[str, Any]:
    bounded_top = max(1, min(int(top_n), 50))
    bounded_recent = max(1, min(int(recent_limit), 200))

    total_events = len(events)
    helpful_count = sum(1 for event in events if event.get("signal") == "up")
    off_target_count = sum(1 for event in events if event.get("signal") == "down")

    note_events = [
        event
        for event in events
        if event.get("signal") == "down" and isinstance(event.get("note"), str) and event.get("note")
    ]

    tag_counts: Counter[str] = Counter()
    tag_latest: dict[str, str] = {}
    tag_samples: defaultdict[str, list[str]] = defaultdict(list)

    for event in note_events:
        note = _normalize_note(str(event.get("note", "")))
        tags = infer_tags_from_note(note)
        timestamp = str(event.get("timestamp", ""))

        for tag in tags:
            tag_counts[tag] += 1
            if len(tag_samples[tag]) < 3 and note not in tag_samples[tag]:
                tag_samples[tag].append(note)
            prev_latest = tag_latest.get(tag, "")
            if timestamp > prev_latest:
                tag_latest[tag] = timestamp

    ranked_tags = tag_counts.most_common(bounded_top)
    prompt_backlog: list[dict[str, Any]] = []
    for tag, count in ranked_tags:
        spec = PROMPT_LIBRARY.get(tag, PROMPT_LIBRARY["general_language"])
        samples = tag_samples.get(tag, [])
        prompt_backlog.append(
            {
                "id": tag,
                "tag": tag,
                "title": spec["title"],
                "count": count,
                "latest_at": tag_latest.get(tag, ""),
                "sample_notes": samples,
                "prompt": _render_prompt(tag, count, samples),
            }
        )

    recent_off_target = []
    for event in note_events[:bounded_recent]:
        recent_off_target.append(
            {
                "id": event.get("id"),
                "timestamp": event.get("timestamp"),
                "query": event.get("query", ""),
                "note": event.get("note", ""),
                "concepts": event.get("concepts", []),
                "people": event.get("people", []),
                "places": event.get("places", []),
                "sources": event.get("sources", []),
            }
        )

    return {
        "summary": {
            "total_events": total_events,
            "helpful_count": helpful_count,
            "off_target_count": off_target_count,
            "off_target_with_notes": len(note_events),
        },
        "prompt_backlog": prompt_backlog,
        "recent_off_target": recent_off_target,
    }
