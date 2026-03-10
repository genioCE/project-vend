from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import httpx

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from .models import EntryChunk


logger = logging.getLogger("entry-summary-provider")


@dataclass(frozen=True)
class EntrySummaryGeneration:
    short_summary: str
    detailed_summary: str
    themes: list[str]
    entities: list[dict[str, str]]
    decisions_actions: list[str]
    model_version: str
    prompt_version: str
    provider: str
    mock: bool


class EntrySummaryProvider(Protocol):
    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunks: list[EntryChunk],
    ) -> EntrySummaryGeneration:
        """Generate summary fields for a single entry."""


class MockEntrySummaryProvider:
    """Deterministic fallback provider for offline/local-first operation."""

    model_version = "deterministic-summary-mock-v1"
    prompt_version = "entry-summary-prompt-mock-v1"

    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunks: list[EntryChunk],
    ) -> EntrySummaryGeneration:
        normalized = _normalize_whitespace(entry_text)
        sentences = _split_sentences(normalized)

        short_summary = _truncate_words(sentences[0] if sentences else normalized, 30)
        detailed_summary = " ".join(sentences[:4]).strip() if sentences else _truncate_words(normalized, 120)

        themes = _extract_key_terms(normalized, max_terms=6)
        raw_entities = _extract_entities(normalized, max_entities=12)
        entities = [{"name": e, "type": "concept"} for e in raw_entities]
        decisions_actions = _extract_decisions(sentences)

        return EntrySummaryGeneration(
            short_summary=short_summary or "No summary available.",
            detailed_summary=detailed_summary or short_summary or "No detailed summary available.",
            themes=themes,
            entities=entities,
            decisions_actions=decisions_actions,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
            provider="mock",
            mock=True,
        )


class LocalEntrySummaryProvider:
    """Local-only provider using rule-based extraction. No API calls.

    Uses hardened regex patterns from the gravity decomposer for entity detection,
    shared keyword lexicons for emotions/decisions/archetypes, and frequency-based
    theme extraction with emotion-proximity boosting.

    When a finetuned theme provider is available, uses neural theme predictions
    instead of rule-based extraction for higher quality theme labels.
    """

    model_version = "local-extractor-v1"
    prompt_version = "entry-summary-local-v1"

    def __init__(self, theme_provider=None, entity_provider=None, decision_provider=None):
        self._theme_provider = theme_provider
        self._entity_provider = entity_provider
        self._decision_provider = decision_provider
        if theme_provider is not None or entity_provider is not None or decision_provider is not None:
            self.prompt_version = "entry-summary-local-finetuned-v1"

    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunks: list[EntryChunk],
    ) -> EntrySummaryGeneration:
        from .local_extractor import (
            extract_decisions_local,
            extract_entities_local,
            extract_themes_local,
        )

        normalized = _normalize_whitespace(entry_text)
        sentences = _split_sentences(normalized)

        short_summary = _truncate_words(sentences[0] if sentences else normalized, 30)
        detailed_summary = " ".join(sentences[:4]).strip() if sentences else _truncate_words(normalized, 120)

        # Entity extraction: regex spans + optional finetuned type classification
        entities = extract_entities_local(entry_text)
        if self._entity_provider is not None:
            try:
                entities = self._entity_provider.classify_entities(entry_text, candidates=entities)
            except Exception:
                logger.warning("finetuned_entity_fallback", exc_info=True)

        # Decision extraction: finetuned sentence classifier or regex fallback
        if self._decision_provider is not None:
            try:
                decisions = self._decision_provider.extract_decisions(entry_text)
            except Exception:
                logger.warning("finetuned_decision_fallback", exc_info=True)
                decisions = extract_decisions_local(entry_text)
        else:
            decisions = extract_decisions_local(entry_text)

        # Use finetuned theme classifier when available, else rule-based
        if self._theme_provider is not None:
            try:
                themes = self._theme_provider.predict_themes(entry_text)
            except Exception:
                logger.warning("finetuned_theme_fallback", exc_info=True)
                themes = extract_themes_local(entry_text)
        else:
            themes = extract_themes_local(entry_text)

        return EntrySummaryGeneration(
            short_summary=short_summary or "No summary available.",
            detailed_summary=detailed_summary or short_summary or "No detailed summary available.",
            themes=themes,
            entities=entities,
            decisions_actions=decisions,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
            provider="local",
            mock=False,
        )


# ---------------------------------------------------------------------------
# Hybrid provider — slim Claude prompt for summaries + themes only
# ---------------------------------------------------------------------------

_HYBRID_SYSTEM_PROMPT = (
    "You analyze and summarize personal journal entries. Read carefully for both surface content and underlying emotional/psychological currents.\n\n"
    "Return strict JSON with these keys:\n\n"
    "- **short_summary**: 1-2 sentences capturing the emotional and thematic core. What is this entry *really about* beneath the surface events?\n\n"
    "- **detailed_summary**: 3-5 sentences covering events, emotions, transitions, and decisions. Trace the arc of the entry — where did the writer start, what shifted, where did they land?\n\n"
    "- **themes**: 3-8 abstract psychological or behavioral themes. Themes are patterns of meaning, NOT topic labels. "
    "Keep each theme SHORT: 2-4 words only. Do NOT use 'through' as a connector. Do NOT pad with prepositional phrases.\n"
    "  Good: 'reclaiming agency', 'solitude vs connection', 'creative mastery', 'processing grief', 'embodied presence'\n"
    "  Bad (too vague): 'work', 'coding', 'feelings'\n"
    "  Bad (too long): 'reclaiming agency through creative work', 'building confidence through technical problem-solving'\n\n"
    "## Example\n\n"
    "Entry: \"Spent the morning rebuilding the search pipeline. Frustrating at first — kept hitting dead ends. But once I switched approaches everything clicked. Called Mom after. She was proud. That meant a lot.\"\n\n"
    "```json\n"
    "{\n"
    '  "short_summary": "A day of technical breakthrough followed by meaningful connection — the writer moved from frustration to mastery to vulnerability.",\n'
    '  "detailed_summary": "The writer spent the morning debugging, hitting dead ends before finding a working approach. The afternoon brought real results. They called their mother and shared their work, finding emotional value in being seen.",\n'
    '  "themes": ["technical confidence", "persistence and breakthrough", "seeking connection", "being witnessed"]\n'
    "}\n"
    "```\n\n"
    "Do not return markdown. Return strict JSON only."
)


class HybridEntrySummaryProvider:
    """Hybrid provider: local extraction for entities/decisions + ONE Claude call for summaries/themes.

    Cuts API cost by ~71% compared to full Claude (eliminates entity/decision extraction
    from Claude prompt AND eliminates the separate state label API call entirely).
    """

    prompt_version = "entry-summary-hybrid-v1"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001", timeout_seconds: float = 120.0):
        if anthropic is None:
            raise ImportError("anthropic package is required for HybridEntrySummaryProvider")
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunks: list[EntryChunk],
    ) -> EntrySummaryGeneration:
        from .local_extractor import (
            extract_decisions_local,
            extract_entities_local,
        )

        # Local extraction (free, instant)
        entities = extract_entities_local(entry_text)
        decisions = extract_decisions_local(entry_text)

        # ONE Claude call for summaries + themes only (slim prompt)
        user_prompt = f"Entry ID: {entry_id}\nEntry text:\n{entry_text}\n"
        parsed = self._call_claude(_HYBRID_SYSTEM_PROMPT, user_prompt)

        short_summary = _normalize_whitespace(str(
            parsed.get("short_summary", "")
            or parsed.get("summary", "")
        ))
        detailed_summary = _normalize_whitespace(str(
            parsed.get("detailed_summary", "")
            or parsed.get("detail_summary", "")
        ))
        themes = _sanitize_list(parsed.get("themes", []), max_items=10)
        themes = [t[0].lower() + t[1:] if t else t for t in themes]
        themes = _dedupe_themes_by_overlap(themes)

        if not short_summary:
            short_summary = _truncate_words(entry_text, 30)
        if not detailed_summary:
            detailed_summary = _truncate_words(entry_text, 140)

        return EntrySummaryGeneration(
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            themes=themes,
            entities=entities,
            decisions_actions=decisions,
            model_version=self.model,
            prompt_version=self.prompt_version,
            provider="hybrid",
            mock=False,
        )

    def _call_claude(self, system_prompt: str, user_prompt: str) -> dict:
        """Make a single Claude API call and return parsed JSON."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return json.loads(content)


# ---------------------------------------------------------------------------
# Ollama prompts
# ---------------------------------------------------------------------------

_FULL_SYSTEM_PROMPT = (
    "You analyze and summarize personal journal entries. Read carefully for both surface content and underlying emotional/psychological currents.\n\n"
    "Return strict JSON with these keys:\n\n"
    "- **short_summary**: 1-2 sentences capturing the emotional and thematic core. What is this entry *really about* beneath the surface events?\n\n"
    "- **detailed_summary**: 3-5 sentences covering events, emotions, transitions, and decisions. Trace the arc of the entry — where did the writer start, what shifted, where did they land?\n\n"
    "- **themes**: 3-8 abstract psychological or behavioral themes. Themes are patterns of meaning, NOT topic labels. "
    "Keep each theme SHORT: 2-4 words only. Do NOT use 'through' as a connector. Do NOT pad with prepositional phrases.\n"
    "  Good: 'reclaiming agency', 'solitude vs connection', 'creative mastery', 'processing grief', 'embodied presence'\n"
    "  Bad (too vague): 'work', 'coding', 'feelings'\n"
    "  Bad (too long): 'reclaiming agency through creative work', 'building confidence through technical problem-solving'\n\n"
    "- **entities**: People, places, organizations, and concepts the writer engages with. Return as a list of objects with \"name\" and \"type\" fields.\n"
    "  Types: \"person\" (humans by name or role), \"place\" (locations, venues), \"organization\" (companies, groups), \"concept\" (ideas, tools, projects), \"spiritual\" (God, Higher Power).\n"
    "  Example: [{\"name\": \"Mom\", \"type\": \"person\"}, {\"name\": \"search pipeline\", \"type\": \"concept\"}, {\"name\": \"God\", \"type\": \"spiritual\"}]\n\n"
    "- **decisions_actions**: Explicit decisions, commitments, realizations, or next steps. Include both stated intentions ('I'm going to...') and implicit commitments revealed by the entry's direction.\n\n"
    "## Example\n\n"
    "Entry: \"Spent the morning rebuilding the search pipeline. Frustrating at first — kept hitting dead ends with the chunking logic. But once I switched to sentence-level splits everything clicked. By afternoon I had semantic search returning real results. Called Mom after. She sounded tired but we had a good talk. Told her about the project. She didn't fully get it but she was proud. That meant a lot.\"\n\n"
    "```json\n"
    "{\n"
    "  \"short_summary\": \"A day of technical breakthrough followed by meaningful connection — the writer moved from frustration to mastery to vulnerability, ending grounded.\",\n"
    "  \"detailed_summary\": \"The writer spent the morning debugging a search pipeline, hitting dead ends before finding a working approach with sentence-level chunking. The afternoon brought real results and a sense of accomplishment. They then called their mother and shared their work, finding emotional value in being seen even without full understanding. The entry traces an arc from frustration through mastery to relational openness.\",\n"
    "  \"themes\": [\"technical confidence\", \"persistence and breakthrough\", \"seeking connection\", \"being witnessed\"],\n"
    "  \"entities\": [{\"name\": \"Mom\", \"type\": \"person\"}, {\"name\": \"search pipeline\", \"type\": \"concept\"}, {\"name\": \"semantic search\", \"type\": \"concept\"}, {\"name\": \"chunking logic\", \"type\": \"concept\"}],\n"
    "  \"decisions_actions\": [\"Switched chunking strategy to sentence-level splits\", \"Reached out to Mom to share progress\", \"Chose to be vulnerable about creative work\"]\n"
    "}\n"
    "```\n\n"
    "Do not return markdown. Return strict JSON only."
)

_CHUNK_EXTRACT_SYSTEM_PROMPT = (
    "You extract structured data from a passage of a personal journal entry. This is one section of a longer entry.\n\n"
    "Return strict JSON with these keys:\n\n"
    "- **themes**: 2-5 abstract psychological or behavioral themes present in THIS passage. Themes are patterns of meaning, not topic labels. "
    "Keep each theme SHORT: 2-4 words only. Do NOT use 'through' as a connector.\n"
    "  Good: 'reclaiming agency', 'solitude vs connection', 'creative mastery'\n"
    "  Bad: 'reclaiming agency through craft', 'building confidence through problem-solving'\n\n"
    "- **entities**: People, places, organizations, and concepts in THIS passage. Return as a list of objects with \"name\" and \"type\" fields.\n"
    "  Types: \"person\", \"place\", \"organization\", \"concept\", \"spiritual\".\n"
    "  Example: [{\"name\": \"Kyle\", \"type\": \"person\"}, {\"name\": \"Hix\", \"type\": \"place\"}]\n\n"
    "- **decisions_actions**: Explicit decisions, commitments, realizations, or next steps found in THIS passage.\n\n"
    "Do not return markdown. Return strict JSON only."
)

_SYNTHESIS_SYSTEM_PROMPT = (
    "You write summaries for personal journal entries. You have been given the opening of a long journal entry "
    "plus themes, entities, and decisions extracted from the FULL entry (including parts you cannot see directly).\n\n"
    "Return strict JSON with these keys:\n\n"
    "- **short_summary**: 1-2 sentences capturing the emotional and thematic core of the FULL entry.\n\n"
    "- **detailed_summary**: 3-5 sentences covering events, emotions, transitions, and decisions across the FULL entry. "
    "Use the extracted themes and entities to ensure you cover content from later sections even though you only see the opening directly.\n\n"
    "Do not return markdown. Return strict JSON only."
)


class OllamaEntrySummaryProvider:
    """Ollama-backed provider with chunk-and-merge for long entries.

    Short entries (<=1500 words): single Ollama call — fast and reliable.
    Long entries (>1500 words): per-chunk extraction + synthesis — full coverage.
    """

    prompt_version = "entry-summary-prompt-ollama-v6"

    # Threshold in words. Below this, single call. Above, chunk-and-merge.
    SINGLE_CALL_MAX_WORDS = 1500

    def __init__(self, ollama_url: str, model: str, timeout_seconds: float = 240.0):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public interface (unchanged signature)
    # ------------------------------------------------------------------

    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunks: list[EntryChunk],
    ) -> EntrySummaryGeneration:
        word_count = len(entry_text.split())
        if word_count <= self.SINGLE_CALL_MAX_WORDS or len(chunks) <= 1:
            return self._generate_single(entry_id, entry_text)
        logger.info(
            "chunk_and_merge_start",
            extra={"entry_id": entry_id, "word_count": word_count, "num_chunks": len(chunks)},
        )
        return self._generate_chunked(entry_id, entry_text, chunks)

    # ------------------------------------------------------------------
    # Single-call path (short entries)
    # ------------------------------------------------------------------

    def _generate_single(self, entry_id: str, entry_text: str) -> EntrySummaryGeneration:
        """Analyze a short entry in one Ollama call."""
        user_prompt = f"Entry ID: {entry_id}\nEntry text:\n{entry_text}\n"

        parsed = self._call_ollama(_FULL_SYSTEM_PROMPT, user_prompt)
        return self._parse_full_response(parsed, entry_id, entry_text)

    # ------------------------------------------------------------------
    # Chunk-and-merge path (long entries)
    # ------------------------------------------------------------------

    def _generate_chunked(
        self,
        entry_id: str,
        entry_text: str,
        chunks: list[EntryChunk],
    ) -> EntrySummaryGeneration:
        """Analyze a long entry by extracting from each chunk, then synthesizing."""

        # Phase 1: Extract themes/entities/decisions from each chunk
        all_themes: list[str] = []
        all_entities: list[dict[str, str]] = []
        all_decisions: list[str] = []

        for i, chunk in enumerate(chunks):
            logger.info(
                "chunk_extract",
                extra={"entry_id": entry_id, "chunk": f"{i + 1}/{len(chunks)}"},
            )
            extracted = self._extract_from_chunk(entry_id, chunk, i, len(chunks))
            all_themes.extend(extracted.get("themes", []))
            all_entities.extend(extracted.get("entities", []))
            all_decisions.extend(extracted.get("decisions_actions", []))

        # Dedupe and limit
        themes = _dedupe_preserve_order(all_themes, max_items=10)
        entities = _dedupe_typed_entities(all_entities, max_items=20)
        decisions_actions = _dedupe_preserve_order(all_decisions, max_items=20)

        # Normalize theme casing and dedup near-duplicates
        themes = [t[0].lower() + t[1:] if t else t for t in themes]
        themes = _dedupe_themes_by_overlap(themes)

        # Phase 2: Synthesize summaries using opening text + extracted data
        logger.info(
            "chunk_synthesize",
            extra={
                "entry_id": entry_id,
                "themes": len(themes),
                "entities": len(entities),
                "decisions": len(decisions_actions),
            },
        )
        summaries = self._synthesize_summaries(
            entry_id, entry_text, themes, entities, decisions_actions,
        )

        short_summary = summaries.get("short_summary", "")
        detailed_summary = summaries.get("detailed_summary", "")

        # Safety fallbacks
        if not short_summary:
            short_summary = _truncate_words(entry_text, 30)
        if not detailed_summary:
            detailed_summary = _truncate_words(entry_text, 140)

        return EntrySummaryGeneration(
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            themes=themes,
            entities=entities,
            decisions_actions=decisions_actions,
            model_version=self.model,
            prompt_version=self.prompt_version,
            provider="ollama",
            mock=False,
        )

    def _extract_from_chunk(
        self,
        entry_id: str,
        chunk: EntryChunk,
        chunk_index: int,
        total_chunks: int,
    ) -> dict:
        """Extract themes/entities/decisions from a single chunk."""
        user_prompt = (
            f"Entry ID: {entry_id}\n"
            f"Passage {chunk_index + 1} of {total_chunks}:\n{chunk.text}\n"
        )

        parsed = self._call_ollama(_CHUNK_EXTRACT_SYSTEM_PROMPT, user_prompt)

        return {
            "themes": _sanitize_list(parsed.get("themes", []), max_items=8),
            "entities": _sanitize_typed_entities(parsed.get("entities", []), max_items=15),
            "decisions_actions": _sanitize_list(
                parsed.get("decisions_actions", [])
                or parsed.get("decisions", [])
                or parsed.get("actions", []),
                max_items=10,
            ),
        }

    def _synthesize_summaries(
        self,
        entry_id: str,
        entry_text: str,
        themes: list[str],
        entities: list[dict[str, str]],
        decisions_actions: list[str],
    ) -> dict:
        """Generate summaries from opening text + extracted data from full entry."""
        # Give the model the first 1500 words for narrative voice/arc
        opening = _truncate_words(entry_text, self.SINGLE_CALL_MAX_WORDS)
        total_words = len(entry_text.split())

        # Flatten entity names for the synthesis prompt (model just needs names for context)
        entity_names = [e.get("name", "") for e in entities if isinstance(e, dict)]

        user_prompt = (
            f"Entry ID: {entry_id}\n"
            f"Entry opening (first {self.SINGLE_CALL_MAX_WORDS} of {total_words} words):\n{opening}\n\n"
            f"== Data extracted from the FULL entry ({total_words} words) ==\n"
            f"Themes: {json.dumps(themes)}\n"
            f"Entities: {json.dumps(entity_names)}\n"
            f"Decisions/Actions: {json.dumps(decisions_actions)}\n\n"
            f"Write short_summary and detailed_summary covering the FULL entry, not just the opening."
        )

        parsed = self._call_ollama(_SYNTHESIS_SYSTEM_PROMPT, user_prompt)

        short_summary = _normalize_whitespace(str(
            parsed.get("short_summary", "")
            or parsed.get("summary", "")
            or parsed.get("text", "")
        ))
        detailed_summary = _normalize_whitespace(str(
            parsed.get("detailed_summary", "")
            or parsed.get("detail_summary", "")
            or parsed.get("detailed", "")
        ))

        return {
            "short_summary": short_summary,
            "detailed_summary": detailed_summary,
        }

    # ------------------------------------------------------------------
    # Ollama HTTP call (shared by all paths)
    # ------------------------------------------------------------------

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> dict:
        """Make a single Ollama chat API call and return parsed JSON."""
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0.2,
            },
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        body = response.json()
        content = (
            body.get("message", {})
            .get("content", "")
            .strip()
        )
        return json.loads(content)

    # ------------------------------------------------------------------
    # Response parsing (single-call path)
    # ------------------------------------------------------------------

    def _parse_full_response(
        self, parsed: dict, entry_id: str, entry_text: str,
    ) -> EntrySummaryGeneration:
        """Parse a full single-call Ollama response into EntrySummaryGeneration."""
        # Handle alternate key names the model sometimes uses
        short_summary = _normalize_whitespace(str(
            parsed.get("short_summary", "")
            or parsed.get("summary", "")
            or parsed.get("text", "")
        ))
        detailed_summary = _normalize_whitespace(str(
            parsed.get("detailed_summary", "")
            or parsed.get("detail_summary", "")
            or parsed.get("detailed", "")
        ))
        themes = _sanitize_list(parsed.get("themes", []), max_items=10)
        entities = _sanitize_typed_entities(parsed.get("entities", []), max_items=20)
        decisions_actions = _sanitize_list(
            parsed.get("decisions_actions", [])
            or parsed.get("decisions", [])
            or parsed.get("actions", []),
            max_items=20,
        )
        # Normalize theme casing and dedup near-duplicates
        themes = [t[0].lower() + t[1:] if t else t for t in themes]
        themes = _dedupe_themes_by_overlap(themes)

        if not short_summary:
            short_summary = _truncate_words(entry_text, 30)
        if not detailed_summary:
            detailed_summary = _truncate_words(entry_text, 140)

        return EntrySummaryGeneration(
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            themes=themes,
            entities=entities,
            decisions_actions=decisions_actions,
            model_version=self.model,
            prompt_version=self.prompt_version,
            provider="ollama",
            mock=False,
        )


# ---------------------------------------------------------------------------
# Claude / Anthropic provider (no chunk-and-merge needed)
# ---------------------------------------------------------------------------


class ClaudeEntrySummaryProvider:
    """Anthropic Claude-backed provider. Handles full entries in a single call."""

    prompt_version = "entry-summary-prompt-claude-v1"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001", timeout_seconds: float = 120.0):
        if anthropic is None:
            raise ImportError("anthropic package is required for ClaudeEntrySummaryProvider")
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunks: list[EntryChunk],
    ) -> EntrySummaryGeneration:
        user_prompt = f"Entry ID: {entry_id}\nEntry text:\n{entry_text}\n"
        parsed = self._call_claude(_FULL_SYSTEM_PROMPT, user_prompt)
        return self._parse_response(parsed, entry_id, entry_text)

    def _call_claude(self, system_prompt: str, user_prompt: str) -> dict:
        """Make a single Claude API call and return parsed JSON."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text.strip()
        # Strip markdown fencing if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return json.loads(content)

    def _parse_response(
        self, parsed: dict, entry_id: str, entry_text: str,
    ) -> EntrySummaryGeneration:
        """Parse Claude response into EntrySummaryGeneration."""
        short_summary = _normalize_whitespace(str(
            parsed.get("short_summary", "")
            or parsed.get("summary", "")
        ))
        detailed_summary = _normalize_whitespace(str(
            parsed.get("detailed_summary", "")
            or parsed.get("detail_summary", "")
        ))
        themes = _sanitize_list(parsed.get("themes", []), max_items=10)
        entities = _sanitize_typed_entities(parsed.get("entities", []), max_items=20)
        decisions_actions = _sanitize_list(
            parsed.get("decisions_actions", [])
            or parsed.get("decisions", [])
            or parsed.get("actions", []),
            max_items=20,
        )
        themes = [t[0].lower() + t[1:] if t else t for t in themes]
        themes = _dedupe_themes_by_overlap(themes)

        if not short_summary:
            short_summary = _truncate_words(entry_text, 30)
        if not detailed_summary:
            detailed_summary = _truncate_words(entry_text, 140)

        return EntrySummaryGeneration(
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            themes=themes,
            entities=entities,
            decisions_actions=decisions_actions,
            model_version=self.model,
            prompt_version=self.prompt_version,
            provider="anthropic",
            mock=False,
        )


# ---------------------------------------------------------------------------
# Utility functions (unchanged)
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]).strip()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _extract_key_terms(text: str, max_terms: int) -> list[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "had", "has", "have", "i", "if", "in", "into", "is", "it", "its", "me",
        "my", "of", "on", "or", "so", "that", "the", "their", "them", "there",
        "this", "to", "was", "were", "will", "with", "you", "your",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower())
    counts = Counter(word for word in words if word not in stop)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:max_terms]]


def _extract_entities(text: str, max_entities: int) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return _dedupe_preserve_order(candidates, max_entities)


def _extract_decisions(sentences: list[str]) -> list[str]:
    keywords = (
        "decide",
        "decided",
        "choice",
        "choose",
        "action",
        "plan",
        "planned",
        "commit",
        "will",
        "next step",
    )
    picked: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            picked.append(_truncate_words(sentence, 28))
    return _dedupe_preserve_order(picked, max_items=12)


def _dedupe_typed_entities(entities: list[dict[str, str]], max_items: int) -> list[dict[str, str]]:
    """Deduplicate typed entity dicts by lowercase name, preserving order."""
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(ent)
        if len(result) >= max_items:
            break
    return result


def _dedupe_preserve_order(values: list[str], max_items: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_whitespace(value)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if len(result) >= max_items:
            break
    return result


def _dedupe_themes_by_overlap(themes: list[str], threshold: float = 0.8) -> list[str]:
    """Remove near-duplicate themes by word overlap.
    Uses intersection/min(len_a, len_b) — catches subset relationships.
    'reclaiming agency' vs 'reclaiming agency through creative work' → keep shorter.
    """
    if not themes:
        return themes
    result: list[str] = []
    for theme in themes:
        theme_words = set(theme.lower().split())
        if not theme_words:
            continue
        is_dup = False
        for i, existing in enumerate(result):
            existing_words = set(existing.lower().split())
            if not existing_words:
                continue
            intersection = theme_words & existing_words
            smaller = min(len(theme_words), len(existing_words))
            if smaller > 0 and len(intersection) / smaller >= threshold:
                if len(theme) < len(existing):
                    result[i] = theme  # Replace with shorter variant
                is_dup = True
                break
        if not is_dup:
            result.append(theme)
    return result


def _extract_string_from_item(item: object) -> str:
    """Extract clean string from item that might be a dict.
    "Jerry" → "Jerry"
    {"name": "Jerry", "type": "person"} → "Jerry"
    {"entity": "pipeline", "category": "project"} → "pipeline"
    """
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("name", "entity", "theme", "decision", "action", "text", "value"):
            val = item.get(key)
            if val and isinstance(val, str):
                return val.strip()
        string_vals = [v for v in item.values() if isinstance(v, str) and v.strip()]
        if string_vals:
            return string_vals[0].strip()
        return ""
    return str(item).strip() if item is not None else ""


VALID_ENTITY_TYPES = {"person", "place", "organization", "concept", "spiritual"}


def _sanitize_typed_entities(value: object, max_items: int) -> list[dict[str, str]]:
    """Parse and sanitize a list of typed entity objects from LLM output.

    Handles:
      - Proper dicts: {"name": "X", "type": "person"} → kept as-is
      - Flat strings: "X" → {"name": "X", "type": "concept"}
      - Nested dicts: {"people": [...], "places": [...]} → flattened
      - Invalid types → fallback to "concept"
    """
    items: list[object] = []
    if isinstance(value, dict):
        # Flatten nested dicts like {"people": [...], "places": [...]}
        for v in value.values():
            if isinstance(v, list):
                items.extend(v)
            else:
                items.append(v)
    elif isinstance(value, list):
        items = list(value)
    else:
        return []

    result: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in items:
        if isinstance(item, dict) and "name" in item:
            name = str(item["name"]).strip()[:256]
            etype = str(item.get("type", "concept")).lower().strip()
            if etype not in VALID_ENTITY_TYPES:
                etype = "concept"
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append({"name": name, "type": etype})
        elif isinstance(item, str) and item.strip():
            name = item.strip()[:256]
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append({"name": name, "type": "concept"})
        elif isinstance(item, dict):
            # Dict without "name" key — try to extract a string
            name = _extract_string_from_item(item)
            if name:
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                result.append({"name": name, "type": "concept"})

        if len(result) >= max_items:
            break

    return result


def _sanitize_list(value: object, max_items: int) -> list[str]:
    if isinstance(value, dict):
        # Flatten nested dicts like {"people": [...], "places": [...]} into a flat list
        items: list[str] = []
        for v in value.values():
            if isinstance(v, list):
                items.extend(_extract_string_from_item(i) for i in v)
            else:
                items.append(_extract_string_from_item(v))
        items = [i for i in items if i]
        return _dedupe_preserve_order(items, max_items)
    if not isinstance(value, list):
        return []
    items = [_extract_string_from_item(item) for item in value]
    items = [i for i in items if i]
    return _dedupe_preserve_order(items, max_items)
