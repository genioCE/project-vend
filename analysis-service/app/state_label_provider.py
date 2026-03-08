from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from .models import StateLabelRequest
from .state_engine import DeterministicStateLabeler

logger = logging.getLogger("state-label-provider")

ALL_DIMENSIONS = (
    "valence",
    "activation",
    "agency",
    "certainty",
    "relational_openness",
    "self_trust",
    "time_orientation",
    "integration",
)

DIMENSION_ANCHORS = {
    "valence": ("heavy", "uplifted"),
    "activation": ("calm", "activated"),
    "agency": ("stuck", "empowered"),
    "certainty": ("conflicted", "resolved"),
    "relational_openness": ("guarded", "open"),
    "self_trust": ("doubt", "trust"),
    "time_orientation": ("past_looping", "future_building"),
    "integration": ("fragmented", "coherent"),
}


@dataclass(frozen=True)
class StateLabelGeneration:
    dimensions: list[dict]
    observed_signals: list[dict]
    model_version: str
    prompt_version: str
    provider: str
    mock: bool


class StateLabelProvider(Protocol):
    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunk_ids: list[str],
        source_file: str | None,
    ) -> StateLabelGeneration:
        """Generate state labels for a single entry."""


class LocalStateLabelProvider:
    """Rule-based state profiler using 192 hand-crafted signal rules. No API calls."""

    def __init__(self) -> None:
        self._labeler = DeterministicStateLabeler()

    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunk_ids: list[str],
        source_file: str | None,
    ) -> StateLabelGeneration:
        request = StateLabelRequest(
            entry_id=entry_id,
            text=entry_text,
            chunk_ids=chunk_ids,
            source_file=source_file,
        )
        response = self._labeler.label(request)

        dimensions = [
            {
                "dimension": dim.dimension,
                "score": dim.score,
                "label": dim.label,
                "rationale": f"Deterministic rule-based: {dim.label}",
            }
            for dim in response.inferred_state_labels
        ]

        signals = [
            {
                "signal_id": sig.signal_id,
                "signal": sig.signal,
                "category": sig.category,
                "direction": sig.direction,
                "dimensions": list(sig.dimensions),
                "weight": sig.weight,
            }
            for sig in response.observed_text_signals
        ]

        return StateLabelGeneration(
            dimensions=dimensions,
            observed_signals=signals,
            model_version=response.version.model_version,
            prompt_version=response.version.prompt_version,
            provider="local",
            mock=False,
        )


# Backward compat alias
MockStateLabelProvider = LocalStateLabelProvider


# ---------------------------------------------------------------------------
# Ollama prompts
# ---------------------------------------------------------------------------

_FULL_SYSTEM_PROMPT = """\
You are a psychological state profiler for personal journal entries. You assess the writer's internal state across 8 dimensions by reading both explicit emotional language AND implicit signals — what the person is doing, how they describe their actions, what their word choices and sentence energy imply.

Score based on the overall feel of the entry, not just keyword spotting. Someone writing excitedly about building a project is activated and agentic even if they never use the word "empowered."

## Dimensions (score from -1.0 to 1.0):

1. **valence** [-1: heavy, +1: uplifted] — Emotional tone. Consider mood words, but also whether the writer sounds burdened or buoyant. Excitement, pride, and satisfaction are high valence. Exhaustion, frustration, and numbness are low.

2. **activation** [-1: calm, +1: activated] — Energy and arousal level. Rapid-fire writing about many activities, urgency, or intensity = high. Stillness, reflection, winding down = low. Neither is better.

3. **agency** [-1: stuck, +1: empowered] — Sense of authorship over their life. Are they making things happen (built, created, decided, figured out, shipped) or feeling acted upon (waiting, blocked, helpless)? Action verbs in past tense about things *they did* are strong agency signals.

4. **certainty** [-1: conflicted, +1: resolved] — Clarity of mind. Are they settled on a direction or wrestling with ambiguity? "I know what I need to do" vs "I keep going back and forth."

5. **relational_openness** [-1: guarded, +1: open] — Orientation toward others. Are they reaching out, sharing, talking about relationships? Or withdrawing, isolating, needing space?

6. **self_trust** [-1: doubt, +1: trust] — Confidence in themselves. Self-criticism, imposter feelings, "am I good enough" = low. Pride in work, trusting instincts, "I figured it out" = high.

7. **time_orientation** [-1: past_looping, +1: future_building] — Temporal focus. Ruminating on the past, regret, nostalgia = low. Planning, envisioning, "next steps", building toward something = high.

8. **integration** [-1: fragmented, +1: coherent] — Internal coherence. Does the entry feel like the writer's thoughts, feelings, and actions are aligned? Or scattered, contradictory, pulled apart?

## Example

Entry: "Got the vector database working today. Spent hours debugging the embedding pipeline but finally cracked it. The semantic search is actually returning relevant results now. I can see how this could become something real. Already thinking about what to build next."

```json
{
  "dimensions": [
    {"dimension": "valence", "score": 0.7, "label": "uplifted", "rationale": "Satisfaction from solving a hard problem, excitement about potential. 'Finally cracked it' and 'something real' convey pride and optimism."},
    {"dimension": "activation", "score": 0.8, "label": "activated", "rationale": "High energy — spent hours working, already thinking about next steps. Momentum-driven writing."},
    {"dimension": "agency", "score": 0.9, "label": "empowered", "rationale": "Strong maker energy — 'got it working', 'cracked it', 'build next'. Every sentence describes something the writer did or plans to do."},
    {"dimension": "certainty", "score": 0.6, "label": "resolved", "rationale": "'Actually returning relevant results' shows confidence in the outcome. 'Could become something real' has slight hedging but overall direction is clear."},
    {"dimension": "relational_openness", "score": 0.0, "label": "between guarded and open", "rationale": "No relational content in this entry — focused entirely on solo technical work."},
    {"dimension": "self_trust", "score": 0.7, "label": "trust", "rationale": "Trusting their ability to solve hard problems. 'Finally cracked it' implies persistence and confidence. No self-doubt present."},
    {"dimension": "time_orientation", "score": 0.8, "label": "future_building", "rationale": "'Already thinking about what to build next' — forward-leaning, generative orientation."},
    {"dimension": "integration", "score": 0.7, "label": "coherent", "rationale": "Thoughts, actions, and feelings are aligned. Working on something meaningful, feeling good about it, planning more. Coherent narrative."}
  ],
  "observed_signals": [
    {"signal": "got it working", "category": "pattern", "direction": "high", "dimensions": ["agency"], "weight": 0.85},
    {"signal": "finally cracked it", "category": "pattern", "direction": "high", "dimensions": ["agency", "valence", "self_trust"], "weight": 0.9},
    {"signal": "something real", "category": "pattern", "direction": "high", "dimensions": ["valence", "certainty"], "weight": 0.75},
    {"signal": "what to build next", "category": "temporal", "direction": "high", "dimensions": ["time_orientation", "agency"], "weight": 0.85},
    {"signal": "spent hours", "category": "lexical", "direction": "high", "dimensions": ["activation"], "weight": 0.7}
  ]
}
```

## Output format

Return strict JSON only with two keys:
- "dimensions": array of exactly 8 objects, each with "dimension" (string), "score" (float -1.0 to 1.0), "label" (string), "rationale" (string explaining your reading)
- "observed_signals": array of objects, each with "signal" (quoted text or phrase from the entry), "category" (one of "lexical","pattern","modal","temporal","relational","structural"), "direction" ("low" or "high"), "dimensions" (array of dimension names this signal informs), "weight" (float 0-1)

All 8 dimensions MUST appear. If a dimension has no signal, score it 0.0 with label "between [low] and [high]". Do not return markdown."""

_CHUNK_SIGNALS_SYSTEM_PROMPT = """\
You extract psychological signals from a passage of a personal journal entry. This is one section of a longer entry.

Look for BOTH explicit emotional language AND implicit signals — what the person is doing, how they describe their actions, what their word choices and sentence energy imply.

The 8 dimensions you are looking for signals about:
- valence (heavy ↔ uplifted), activation (calm ↔ activated), agency (stuck ↔ empowered)
- certainty (conflicted ↔ resolved), relational_openness (guarded ↔ open), self_trust (doubt ↔ trust)
- time_orientation (past_looping ↔ future_building), integration (fragmented ↔ coherent)

Return strict JSON with one key:
- "observed_signals": array of objects, each with "signal" (quoted text or phrase), "category" (one of "lexical","pattern","modal","temporal","relational","structural"), "direction" ("low" or "high"), "dimensions" (array of dimension names), "weight" (float 0-1)

Do not return markdown. Return strict JSON only."""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a psychological state profiler for personal journal entries. You have been given the opening of a long journal entry plus observed psychological signals extracted from the FULL entry (including later sections you cannot see directly).

Score the writer's internal state across all 8 dimensions, considering ALL the provided signals — not just what you read in the opening.

## Dimensions (score from -1.0 to 1.0):
1. valence [-1: heavy, +1: uplifted]
2. activation [-1: calm, +1: activated]
3. agency [-1: stuck, +1: empowered]
4. certainty [-1: conflicted, +1: resolved]
5. relational_openness [-1: guarded, +1: open]
6. self_trust [-1: doubt, +1: trust]
7. time_orientation [-1: past_looping, +1: future_building]
8. integration [-1: fragmented, +1: coherent]

Return strict JSON with two keys:
- "dimensions": array of exactly 8 objects, each with "dimension", "score" (float -1 to 1), "label", "rationale"
- "observed_signals": curated array of the most important signals (select from those provided + any you notice in the opening)

All 8 dimensions MUST appear. Do not return markdown."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]).strip()


def _normalize_dimension_name(name: str) -> str:
    """'Relational Openness' → 'relational_openness', 'Self-Trust' → 'self_trust'"""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class OllamaStateLabelProvider:
    """Ollama-backed state label provider with chunk-and-merge for long entries."""

    prompt_version = "state-label-prompt-ollama-v4"

    # Same threshold as entry summary provider
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
        chunk_ids: list[str],
        source_file: str | None,
    ) -> StateLabelGeneration:
        word_count = len(entry_text.split())
        if word_count <= self.SINGLE_CALL_MAX_WORDS:
            return self._generate_single(entry_id, entry_text)
        logger.info(
            "state_chunk_and_merge_start",
            extra={"entry_id": entry_id, "word_count": word_count},
        )
        return self._generate_chunked(entry_id, entry_text, chunk_ids)

    # ------------------------------------------------------------------
    # Single-call path (short entries)
    # ------------------------------------------------------------------

    def _generate_single(self, entry_id: str, entry_text: str) -> StateLabelGeneration:
        """Analyze a short entry in one Ollama call."""
        user_prompt = f"Entry ID: {entry_id}\nEntry text:\n{entry_text}"
        parsed = self._call_ollama(_FULL_SYSTEM_PROMPT, user_prompt)
        return self._parse_full_response(parsed)

    # ------------------------------------------------------------------
    # Chunk-and-merge path (long entries)
    # ------------------------------------------------------------------

    def _generate_chunked(
        self,
        entry_id: str,
        entry_text: str,
        chunk_ids: list[str],
    ) -> StateLabelGeneration:
        """Extract signals per-chunk, then synthesize final dimensions."""

        # Split entry_text back into chunks at the same ~500 word boundaries.
        # We use the entry_text directly since chunk texts were joined with \n\n.
        text_chunks = self._split_into_chunks(entry_text)

        # Phase 1: Extract signals from each chunk
        all_signals: list[dict] = []
        for i, chunk_text in enumerate(text_chunks):
            logger.info(
                "state_chunk_extract",
                extra={"entry_id": entry_id, "chunk": f"{i + 1}/{len(text_chunks)}"},
            )
            signals = self._extract_signals_from_chunk(entry_id, chunk_text, i, len(text_chunks))
            all_signals.extend(signals)

        # Dedupe signals by signal text (case-insensitive)
        seen: set[str] = set()
        unique_signals: list[dict] = []
        for sig in all_signals:
            key = sig.get("signal", "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique_signals.append(sig)

        # Phase 2: Synthesize dimensions using opening text + all signals
        logger.info(
            "state_chunk_synthesize",
            extra={"entry_id": entry_id, "total_signals": len(unique_signals)},
        )
        return self._synthesize_state(entry_id, entry_text, unique_signals)

    def _split_into_chunks(self, entry_text: str, max_words: int = 500) -> list[str]:
        """Split entry text into ~500 word chunks at paragraph boundaries."""
        paragraphs = re.split(r"\n\s*\n", entry_text)
        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_words = len(para.split())
            if current_words + para_words > max_words and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words

        if current:
            chunks.append("\n\n".join(current))

        return chunks if chunks else [entry_text]

    def _extract_signals_from_chunk(
        self,
        entry_id: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> list[dict]:
        """Extract psychological signals from a single chunk."""
        user_prompt = (
            f"Entry ID: {entry_id}\n"
            f"Passage {chunk_index + 1} of {total_chunks}:\n{chunk_text}\n"
        )

        parsed = self._call_ollama(_CHUNK_SIGNALS_SYSTEM_PROMPT, user_prompt)

        raw_signals = parsed.get("observed_signals", [])
        signals: list[dict] = []
        for sig in raw_signals:
            signals.append({
                "signal": str(sig.get("signal", "")),
                "category": str(sig.get("category", "lexical")),
                "direction": str(sig.get("direction", "neutral")),
                "dimensions": list(sig.get("dimensions", [])),
                "weight": max(0.0, min(1.0, float(sig.get("weight", 0.5)))),
            })
        return signals

    def _synthesize_state(
        self,
        entry_id: str,
        entry_text: str,
        all_signals: list[dict],
    ) -> StateLabelGeneration:
        """Synthesize final dimensions from opening text + extracted signals."""
        opening = _truncate_words(entry_text, self.SINGLE_CALL_MAX_WORDS)
        total_words = len(entry_text.split())

        # Format signals for the prompt
        signals_text = json.dumps(all_signals, indent=None)

        user_prompt = (
            f"Entry ID: {entry_id}\n"
            f"Entry opening (first {self.SINGLE_CALL_MAX_WORDS} of {total_words} words):\n{opening}\n\n"
            f"== Signals extracted from the FULL entry ({total_words} words) ==\n"
            f"{signals_text}\n\n"
            f"Score all 8 dimensions considering ALL these signals from the full entry, not just the opening."
        )

        parsed = self._call_ollama(_SYNTHESIS_SYSTEM_PROMPT, user_prompt)
        return self._parse_full_response(parsed)

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
        content = body.get("message", {}).get("content", "").strip()
        return json.loads(content)

    # ------------------------------------------------------------------
    # Response parsing (shared by single-call and synthesis)
    # ------------------------------------------------------------------

    def _parse_full_response(self, parsed: dict) -> StateLabelGeneration:
        """Parse an Ollama response with dimensions + signals."""
        # Alternate top-level key fallbacks for dimensions
        raw_dims = (
            parsed.get("dimensions")
            or parsed.get("dimension_scores")
            or parsed.get("scores")
            or parsed.get("dimension_profiles")
            or []
        )

        # Handle nested dict format: {"valence": {...}, "activation": {...}}
        if isinstance(raw_dims, dict):
            raw_dims = list(raw_dims.values())

        dims_by_name: dict[str, dict] = {}
        for item in raw_dims:
            if not isinstance(item, dict):
                continue
            # Try alternate keys for the dimension name
            raw_name = (
                item.get("dimension")
                or item.get("name")
                or item.get("dim")
                or ""
            )
            name = _normalize_dimension_name(str(raw_name))
            if name in DIMENSION_ANCHORS:
                score = max(-1.0, min(1.0, float(item.get("score", 0.0))))
                dims_by_name[name] = {
                    "dimension": name,
                    "score": score,
                    "label": str(item.get("label", "")),
                    "rationale": str(item.get("rationale", "")),
                }

        # Fallback: flat dict format {"agency": 0.7, ...} or {"agency": {"score": 0.7, ...}, ...}
        if not dims_by_name:
            for key, val in parsed.items():
                norm_key = _normalize_dimension_name(key)
                if norm_key not in DIMENSION_ANCHORS:
                    continue
                if isinstance(val, (int, float)):
                    dims_by_name[norm_key] = {
                        "dimension": norm_key,
                        "score": max(-1.0, min(1.0, float(val))),
                        "label": "",
                        "rationale": "",
                    }
                elif isinstance(val, dict):
                    score = val.get("score", val.get("value", 0.0))
                    dims_by_name[norm_key] = {
                        "dimension": norm_key,
                        "score": max(-1.0, min(1.0, float(score))),
                        "label": str(val.get("label", "")),
                        "rationale": str(val.get("rationale", "")),
                    }
            if dims_by_name:
                logger.info(
                    "state_label_flat_dict_fallback: found %d/%d from top-level keys",
                    len(dims_by_name), len(ALL_DIMENSIONS),
                )

        found_count = len(dims_by_name)
        defaulted_count = len(ALL_DIMENSIONS) - found_count

        if found_count == 0:
            logger.warning(
                "state_label_all_zero: 0/%d dimensions found, raw keys=%s",
                len(ALL_DIMENSIONS),
                list(parsed.keys()),
            )
            raise ValueError(
                f"All {len(ALL_DIMENSIONS)} dimensions defaulted — "
                f"response structure unrecognizable (keys: {list(parsed.keys())})"
            )

        if defaulted_count > 0:
            missing = [d for d in ALL_DIMENSIONS if d not in dims_by_name]
            logger.warning(
                "state_label_partial: %d/%d found, %d defaulted, "
                "found=%s, missing=%s",
                found_count, len(ALL_DIMENSIONS), defaulted_count,
                list(dims_by_name.keys()), missing,
            )

        dimensions: list[dict] = []
        for dim_name in ALL_DIMENSIONS:
            if dim_name in dims_by_name:
                dimensions.append(dims_by_name[dim_name])
            else:
                low, high = DIMENSION_ANCHORS[dim_name]
                dimensions.append({
                    "dimension": dim_name,
                    "score": 0.0,
                    "label": f"between {low} and {high}",
                    "rationale": "No signal detected by model.",
                })

        # Alternate key fallback for signals
        raw_signals = parsed.get("observed_signals") or parsed.get("signals") or []
        signals: list[dict] = []
        for sig in raw_signals:
            if not isinstance(sig, dict):
                continue
            signals.append({
                "signal": str(sig.get("signal", "")),
                "category": str(sig.get("category", "lexical")),
                "direction": str(sig.get("direction", "neutral")),
                "dimensions": list(sig.get("dimensions", [])),
                "weight": max(0.0, min(1.0, float(sig.get("weight", 0.5)))),
            })

        return StateLabelGeneration(
            dimensions=dimensions,
            observed_signals=signals,
            model_version=self.model,
            prompt_version=self.prompt_version,
            provider="ollama",
            mock=False,
        )


# ---------------------------------------------------------------------------
# Claude / Anthropic provider (no chunk-and-merge needed)
# ---------------------------------------------------------------------------


class ClaudeStateLabelProvider:
    """Anthropic Claude-backed state label provider. Single call per entry."""

    prompt_version = "state-label-prompt-claude-v1"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001", timeout_seconds: float = 120.0):
        if anthropic is None:
            raise ImportError("anthropic package is required for ClaudeStateLabelProvider")
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        entry_id: str,
        entry_text: str,
        chunk_ids: list[str],
        source_file: str | None,
    ) -> StateLabelGeneration:
        user_prompt = f"Entry ID: {entry_id}\nEntry text:\n{entry_text}"
        parsed = self._call_claude(_FULL_SYSTEM_PROMPT, user_prompt)
        return self._parse_response(parsed)

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

    def _parse_response(self, parsed: dict) -> StateLabelGeneration:
        """Parse Claude response — reuses OllamaStateLabelProvider._parse_full_response logic."""
        raw_dims = (
            parsed.get("dimensions")
            or parsed.get("dimension_scores")
            or parsed.get("scores")
            or []
        )

        if isinstance(raw_dims, dict):
            raw_dims = list(raw_dims.values())

        dims_by_name: dict[str, dict] = {}
        for item in raw_dims:
            if not isinstance(item, dict):
                continue
            raw_name = item.get("dimension") or item.get("name") or ""
            name = _normalize_dimension_name(str(raw_name))
            if name in DIMENSION_ANCHORS:
                score = max(-1.0, min(1.0, float(item.get("score", 0.0))))
                dims_by_name[name] = {
                    "dimension": name,
                    "score": score,
                    "label": str(item.get("label", "")),
                    "rationale": str(item.get("rationale", "")),
                }

        if not dims_by_name:
            raise ValueError(
                f"All {len(ALL_DIMENSIONS)} dimensions defaulted — "
                f"response structure unrecognizable (keys: {list(parsed.keys())})"
            )

        dimensions: list[dict] = []
        for dim_name in ALL_DIMENSIONS:
            if dim_name in dims_by_name:
                dimensions.append(dims_by_name[dim_name])
            else:
                low, high = DIMENSION_ANCHORS[dim_name]
                dimensions.append({
                    "dimension": dim_name,
                    "score": 0.0,
                    "label": f"between {low} and {high}",
                    "rationale": "No signal detected by model.",
                })

        raw_signals = parsed.get("observed_signals") or parsed.get("signals") or []
        signals: list[dict] = []
        for sig in raw_signals:
            if not isinstance(sig, dict):
                continue
            signals.append({
                "signal": str(sig.get("signal", "")),
                "category": str(sig.get("category", "lexical")),
                "direction": str(sig.get("direction", "neutral")),
                "dimensions": list(sig.get("dimensions", [])),
                "weight": max(0.0, min(1.0, float(sig.get("weight", 0.5)))),
            })

        return StateLabelGeneration(
            dimensions=dimensions,
            observed_signals=signals,
            model_version=self.model,
            prompt_version=self.prompt_version,
            provider="anthropic",
            mock=False,
        )
