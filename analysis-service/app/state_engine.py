from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Protocol

logger = logging.getLogger("analysis-service.state-engine")

from .models import (
    AnalysisVersion,
    DimensionConfidence,
    InferredStateLabel,
    ObservedTextSignal,
    Provenance,
    SourceSpan,
    StateConfidence,
    StateDimension,
    StateDimensionProfile,
    StateLabelRequest,
    StateLabelResponse,
    StateProfile,
)

STATE_SCHEMA_VERSION = "state-profile-v1"
STATE_PROMPT_VERSION = "state-engine-local-v2"
STATE_MODEL_VERSION = "deterministic-ruleset-v2"

# ─── Scoring constants ──────────────────────────────────────────────────────
# Bayesian smoothing: score = (high - low) / (total + K)
# K=3 dampens extreme scores while staying sensitive:
#   1 hit  → ±0.25 (at threshold — registers as directional)
#   2 hits → ±0.40 (clear signal)
#   5 hits → ±0.63
#  10 hits → ±0.77
#  20 hits → ±0.87
SCORE_SMOOTHING_K = 3
SCORE_CAP = 0.95  # never output ±1.0

# ─── Negation detection ──────────────────────────────────────────────────────
_NEGATION_WORDS: set[str] = {
    "not", "no", "don't", "doesn't", "didn't", "won't", "wouldn't",
    "can't", "cannot", "couldn't", "shouldn't", "isn't", "aren't",
    "wasn't", "weren't", "never", "neither", "nor", "hardly", "barely",
    "don't", "doesn't", "didn't",  # unicode curly apostrophes
}
_NEGATION_WINDOW = 3  # words before match to check

SignalCategory = Literal[
    "lexical",
    "pattern",
    "modal",
    "temporal",
    "relational",
    "structural",
]
SignalDirection = Literal["low", "high"]


@dataclass(frozen=True)
class SignalRule:
    phrase: str
    category: SignalCategory


@dataclass(frozen=True)
class DimensionSpec:
    dimension: StateDimension
    low_anchor: str
    high_anchor: str
    low_rules: tuple[SignalRule, ...]
    high_rules: tuple[SignalRule, ...]


STATE_DIMENSION_SPECS: tuple[DimensionSpec, ...] = (
    DimensionSpec(
        dimension="valence",
        low_anchor="heavy",
        high_anchor="uplifted",
        low_rules=(
            SignalRule("heavy", "lexical"),
            SignalRule("drained", "lexical"),
            SignalRule("sad", "lexical"),
            SignalRule("grief", "lexical"),
            SignalRule("numb", "lexical"),
            SignalRule("ashamed", "lexical"),
            SignalRule("exhausted", "lexical"),
            SignalRule("overwhelmed", "lexical"),
            SignalRule("frustrated", "lexical"),
            SignalRule("angry", "lexical"),
            SignalRule("anxious", "lexical"),
            SignalRule("flat", "lexical"),
            SignalRule("empty", "lexical"),
            SignalRule("dark", "lexical"),
            SignalRule("depressed", "lexical"),
            SignalRule("hopeless", "lexical"),
            SignalRule("discouraged", "lexical"),
            SignalRule("miserable", "lexical"),
            SignalRule("hurt", "lexical"),
            SignalRule("lonely", "lexical"),
            SignalRule("weighed down", "pattern"),
            SignalRule("hard day", "pattern"),
            SignalRule("rough day", "pattern"),
            SignalRule("feeling low", "pattern"),
        ),
        high_rules=(
            SignalRule("uplifted", "lexical"),
            SignalRule("grateful", "lexical"),
            SignalRule("joy", "lexical"),
            SignalRule("light", "lexical"),
            SignalRule("hopeful", "lexical"),
            SignalRule("relief", "lexical"),
            SignalRule("excited", "lexical"),
            SignalRule("alive", "lexical"),
            SignalRule("energized", "lexical"),
            SignalRule("happy", "lexical"),
            SignalRule("warm", "lexical"),
            SignalRule("proud", "lexical"),
            SignalRule("satisfied", "lexical"),
            SignalRule("content", "lexical"),
            SignalRule("peaceful", "lexical"),
            SignalRule("thrilled", "lexical"),
            SignalRule("inspired", "lexical"),
            SignalRule("encouraged", "lexical"),
            SignalRule("optimistic", "lexical"),
            SignalRule("good day", "pattern"),
            SignalRule("great day", "pattern"),
            SignalRule("feels good", "pattern"),
            SignalRule("feeling good", "pattern"),
            SignalRule("love this", "pattern"),
            SignalRule("so cool", "pattern"),
        ),
    ),
    DimensionSpec(
        dimension="activation",
        low_anchor="calm",
        high_anchor="activated",
        low_rules=(
            SignalRule("calm", "lexical"),
            SignalRule("still", "lexical"),  # borderline but smoothing handles over-counting
            SignalRule("slow", "lexical"),
            # "rest" removed — "the rest of" is NOT calm
            SignalRule("steady", "lexical"),
            SignalRule("quiet", "lexical"),
            SignalRule("settled", "lexical"),
            SignalRule("relaxed", "lexical"),
            SignalRule("peaceful", "lexical"),
            SignalRule("stillness", "lexical"),
            SignalRule("grounded", "lexical"),
            SignalRule("eased", "lexical"),
            SignalRule("unwinding", "lexical"),
            SignalRule("resting", "lexical"),
            SignalRule("slowed down", "pattern"),
            SignalRule("taking it easy", "pattern"),
            SignalRule("nothing much", "pattern"),
        ),
        high_rules=(
            SignalRule("activated", "lexical"),
            SignalRule("urgent", "lexical"),
            SignalRule("racing", "lexical"),
            SignalRule("intense", "lexical"),
            SignalRule("wired", "lexical"),
            SignalRule("buzzing", "lexical"),
            SignalRule("momentum", "lexical"),
            SignalRule("driven", "lexical"),
            SignalRule("restless", "lexical"),
            SignalRule("pushing", "lexical"),
            SignalRule("moving", "lexical"),  # borderline but more often activated than not
            SignalRule("cranking", "lexical"),
            SignalRule("productive", "lexical"),
            SignalRule("busy", "lexical"),
            SignalRule("energized", "lexical"),
            SignalRule("nonstop", "lexical"),
            SignalRule("fired up", "pattern"),
            SignalRule("on a roll", "pattern"),
            SignalRule("deep into", "pattern"),
            SignalRule("couldn't stop", "pattern"),
            SignalRule("so much to do", "pattern"),
            SignalRule("heads down", "pattern"),
            SignalRule("in the zone", "pattern"),
            SignalRule("locked in", "pattern"),
        ),
    ),
    DimensionSpec(
        dimension="agency",
        low_anchor="stuck",
        high_anchor="empowered",
        low_rules=(
            SignalRule("stuck", "lexical"),
            SignalRule("trapped", "lexical"),
            SignalRule("helpless", "lexical"),
            SignalRule("cant", "modal"),
            SignalRule("can't", "modal"),
            SignalRule("cannot", "modal"),
            SignalRule("blocked", "lexical"),
            SignalRule("waiting", "lexical"),
            SignalRule("stalled", "lexical"),
            SignalRule("frozen", "lexical"),
            SignalRule("paralyzed", "lexical"),
            SignalRule("lost", "lexical"),
            SignalRule("powerless", "lexical"),
            SignalRule("spinning", "lexical"),
            SignalRule("don't know what to do", "pattern"),
            SignalRule("out of my hands", "pattern"),
            SignalRule("no control", "pattern"),
            SignalRule("nothing i can do", "pattern"),
        ),
        high_rules=(
            SignalRule("empowered", "lexical"),
            SignalRule("choose", "modal"),
            SignalRule("decide", "modal"),
            SignalRule("build", "modal"),
            # "can" removed — too generic (7758 matches, fires on "I can see", "you can tell")
            SignalRule("built", "modal"),
            SignalRule("created", "modal"),
            # "made" removed — too generic ("made dinner", "made a mess")
            SignalRule("shipped", "modal"),
            SignalRule("solved", "modal"),
            SignalRule("launched", "modal"),
            SignalRule("implemented", "modal"),
            SignalRule("designed", "modal"),
            SignalRule("finished", "modal"),
            SignalRule("accomplished", "lexical"),
            SignalRule("capable", "lexical"),
            SignalRule("took action", "pattern"),
            SignalRule("figured out", "pattern"),
            SignalRule("got it working", "pattern"),
            SignalRule("working on", "pattern"),
            SignalRule("making progress", "pattern"),
            SignalRule("took the step", "pattern"),
            SignalRule("pulled it off", "pattern"),
            SignalRule("i did it", "pattern"),
            SignalRule("i made it", "pattern"),
            SignalRule("made it happen", "pattern"),
            SignalRule("took control", "pattern"),
            SignalRule("stepped up", "pattern"),
        ),
    ),
    DimensionSpec(
        dimension="certainty",
        low_anchor="conflicted",
        high_anchor="resolved",
        low_rules=(
            SignalRule("conflicted", "lexical"),
            SignalRule("unsure", "lexical"),
            SignalRule("uncertain", "lexical"),
            SignalRule("torn", "lexical"),
            SignalRule("doubt", "lexical"),
            SignalRule("maybe", "modal"),
            SignalRule("confused", "lexical"),
            SignalRule("questioning", "lexical"),
            SignalRule("struggling", "lexical"),
            SignalRule("wondering", "lexical"),
            SignalRule("ambivalent", "lexical"),
            SignalRule("hesitant", "lexical"),
            SignalRule("don't know", "pattern"),
            SignalRule("not sure", "pattern"),
            SignalRule("hard to tell", "pattern"),
            SignalRule("on the fence", "pattern"),
            SignalRule("back and forth", "pattern"),
            SignalRule("can't decide", "pattern"),
        ),
        high_rules=(
            SignalRule("resolved", "lexical"),
            SignalRule("clear", "lexical"),
            SignalRule("certain", "lexical"),
            SignalRule("committed", "lexical"),
            SignalRule("decided", "modal"),
            SignalRule("aligned", "structural"),
            SignalRule("sure", "lexical"),  # borderline; smoothing handles
            SignalRule("confident", "lexical"),
            SignalRule("obvious", "lexical"),
            SignalRule("settled", "lexical"),
            SignalRule("convinced", "lexical"),
            SignalRule("know what", "pattern"),  # borderline; smoothing handles
            SignalRule("figured out", "pattern"),
            SignalRule("landed on", "pattern"),
            SignalRule("no question", "pattern"),
            SignalRule("this is it", "pattern"),
            SignalRule("the answer", "pattern"),
            SignalRule("makes sense", "pattern"),
        ),
    ),
    DimensionSpec(
        dimension="relational_openness",
        low_anchor="guarded",
        high_anchor="open",
        low_rules=(
            SignalRule("guarded", "relational"),
            SignalRule("withdrawn", "relational"),
            SignalRule("isolated", "relational"),
            SignalRule("defensive", "relational"),
            SignalRule("closed off", "relational"),
            SignalRule("alone", "relational"),
            SignalRule("hiding", "relational"),
            SignalRule("shut down", "relational"),
            SignalRule("pulling away", "relational"),
            SignalRule("walls up", "relational"),
            SignalRule("distant", "relational"),
            SignalRule("avoiding", "relational"),
            SignalRule("kept to myself", "pattern"),
            SignalRule("don't want to talk", "pattern"),
            SignalRule("need space", "pattern"),
            SignalRule("pushed away", "pattern"),
        ),
        high_rules=(
            SignalRule("open", "relational"),  # borderline; smoothing prevents extreme
            SignalRule("vulnerable", "relational"),
            SignalRule("honest", "relational"),  # borderline; smoothing prevents extreme
            SignalRule("shared", "relational"),
            SignalRule("connected", "relational"),
            SignalRule("told", "relational"),
            SignalRule("talked", "relational"),
            SignalRule("showed", "relational"),
            SignalRule("sharing", "relational"),
            SignalRule("reached out", "relational"),
            SignalRule("together", "relational"),
            SignalRule("trusted", "relational"),
            SignalRule("let them in", "pattern"),
            SignalRule("opened up", "pattern"),
            SignalRule("real conversation", "pattern"),
            SignalRule("felt seen", "pattern"),
            SignalRule("felt heard", "pattern"),
            SignalRule("deep conversation", "pattern"),
            SignalRule("heart to heart", "pattern"),
        ),
    ),
    DimensionSpec(
        dimension="self_trust",
        low_anchor="doubt",
        high_anchor="trust",
        low_rules=(
            SignalRule("self-doubt", "lexical"),
            SignalRule("second guess", "pattern"),
            SignalRule("insecure", "lexical"),
            SignalRule("inner critic", "pattern"),
            SignalRule("not enough", "pattern"),
            SignalRule("fraud", "lexical"),
            SignalRule("imposter", "lexical"),
            SignalRule("failing", "lexical"),
            SignalRule("incompetent", "lexical"),
            SignalRule("inadequate", "lexical"),
            SignalRule("not good enough", "pattern"),
            SignalRule("worried i", "pattern"),
            SignalRule("what if i", "pattern"),
            SignalRule("can't handle", "pattern"),
            SignalRule("who am i to", "pattern"),
            SignalRule("don't deserve", "pattern"),
            SignalRule("out of my depth", "pattern"),
        ),
        high_rules=(
            SignalRule("self-trust", "lexical"),
            SignalRule("grounded", "lexical"),
            SignalRule("confident", "lexical"),
            SignalRule("self belief", "pattern"),
            SignalRule("i trust", "pattern"),
            SignalRule("capable", "lexical"),
            SignalRule("solid", "lexical"),  # borderline; smoothing handles
            SignalRule("strong", "lexical"),  # borderline; smoothing handles
            SignalRule("competent", "lexical"),
            SignalRule("worthy", "lexical"),
            SignalRule("proud of myself", "pattern"),
            # "i can" removed — 3660 matches, too generic ("I can see", "I can tell")
            SignalRule("figured it out", "pattern"),
            SignalRule("trust myself", "pattern"),
            SignalRule("i know what i", "pattern"),
            SignalRule("i've got this", "pattern"),
            SignalRule("believe in myself", "pattern"),
            SignalRule("i can do this", "pattern"),
            SignalRule("i can handle", "pattern"),
        ),
    ),
    DimensionSpec(
        dimension="time_orientation",
        low_anchor="past_looping",
        high_anchor="future_building",
        low_rules=(
            SignalRule("again", "temporal"),  # borderline; smoothing handles
            SignalRule("replay", "temporal"),
            SignalRule("regret", "temporal"),
            SignalRule("should have", "temporal"),
            # "yesterday" removed — neutral time marker, not past_looping
            SignalRule("used to", "temporal"),
            SignalRule("back then", "temporal"),
            SignalRule("haunted", "temporal"),
            SignalRule("ruminating", "temporal"),
            SignalRule("dwelling", "temporal"),
            SignalRule("nostalgic", "temporal"),
            SignalRule("keep thinking about", "pattern"),
            SignalRule("wish i had", "pattern"),
            SignalRule("can't let go", "pattern"),
            SignalRule("same old", "pattern"),
            SignalRule("over and over", "pattern"),
            SignalRule("years ago", "temporal"),
        ),
        high_rules=(
            SignalRule("next", "temporal"),
            SignalRule("plan", "temporal"),
            SignalRule("tomorrow", "temporal"),
            SignalRule("build", "temporal"),
            SignalRule("forward", "temporal"),
            SignalRule("going to", "temporal"),
            SignalRule("starting", "temporal"),
            SignalRule("soon", "temporal"),
            SignalRule("vision", "temporal"),
            SignalRule("future", "temporal"),
            SignalRule("goal", "temporal"),
            SignalRule("roadmap", "temporal"),
            SignalRule("working toward", "pattern"),
            SignalRule("excited about", "pattern"),
            SignalRule("can't wait", "pattern"),
            SignalRule("looking forward", "pattern"),
            SignalRule("down the road", "pattern"),
            SignalRule("next step", "pattern"),
            SignalRule("want to build", "pattern"),
        ),
    ),
    DimensionSpec(
        dimension="integration",
        low_anchor="fragmented",
        high_anchor="coherent",
        low_rules=(
            SignalRule("fragmented", "structural"),
            SignalRule("scattered", "structural"),
            SignalRule("split", "structural"),
            SignalRule("chaotic", "structural"),
            SignalRule("disjointed", "structural"),
            SignalRule("disconnected", "structural"),
            SignalRule("messy", "structural"),
            SignalRule("confused", "structural"),
            SignalRule("conflicting", "structural"),
            SignalRule("all over the place", "pattern"),
            SignalRule("can't focus", "pattern"),
            SignalRule("pulled in", "pattern"),
            SignalRule("contradicting", "pattern"),
            SignalRule("doesn't add up", "pattern"),
            SignalRule("falling apart", "pattern"),
            SignalRule("nothing makes sense", "pattern"),
        ),
        high_rules=(
            SignalRule("coherent", "structural"),
            SignalRule("integrated", "structural"),
            SignalRule("whole", "structural"),  # borderline; smoothing handles
            SignalRule("consistent", "structural"),
            SignalRule("centered", "structural"),
            SignalRule("aligned", "structural"),
            SignalRule("flow", "structural"),  # borderline; smoothing handles
            SignalRule("harmony", "structural"),
            SignalRule("unified", "structural"),
            SignalRule("balanced", "structural"),
            SignalRule("making sense", "pattern"),
            SignalRule("clicking", "pattern"),
            SignalRule("coming together", "pattern"),
            SignalRule("everything fits", "pattern"),
            SignalRule("pieces falling into place", "pattern"),
            SignalRule("all connects", "pattern"),
            SignalRule("feels right", "pattern"),
        ),
    ),
)


class StateLabeler(Protocol):
    def label(self, request: StateLabelRequest) -> StateLabelResponse:
        """Compute state labels for a single entry."""


@dataclass
class _DimensionAccumulator:
    low_hits: int = 0
    high_hits: int = 0
    signal_ids: list[str] | None = None
    evidence_spans: list[SourceSpan] | None = None


class DeterministicStateLabeler:
    """Deterministic v1 labeler.

    TODO: replace with local model-backed classifier while preserving
    schema compatibility and provenance behavior.
    """

    def label(self, request: StateLabelRequest) -> StateLabelResponse:
        normalized_text = _normalize_whitespace(request.text)
        lowered = normalized_text.lower()
        fallback_chunk_id = request.chunk_ids[0] if request.chunk_ids else f"{request.entry_id}::chunk-000"

        accumulators: dict[StateDimension, _DimensionAccumulator] = {
            spec.dimension: _DimensionAccumulator(signal_ids=[], evidence_spans=[])
            for spec in STATE_DIMENSION_SPECS
        }

        observed_signals: list[ObservedTextSignal] = []
        signal_keys_seen: set[tuple[StateDimension, str, int, int]] = set()

        for spec in STATE_DIMENSION_SPECS:
            self._collect_signals(
                text=lowered,
                original_text=normalized_text,
                spec=spec,
                direction="low",
                rules=spec.low_rules,
                fallback_chunk_id=fallback_chunk_id,
                source_file=request.source_file,
                accumulator=accumulators[spec.dimension],
                observed_signals=observed_signals,
                signal_keys_seen=signal_keys_seen,
            )
            self._collect_signals(
                text=lowered,
                original_text=normalized_text,
                spec=spec,
                direction="high",
                rules=spec.high_rules,
                fallback_chunk_id=fallback_chunk_id,
                source_file=request.source_file,
                accumulator=accumulators[spec.dimension],
                observed_signals=observed_signals,
                signal_keys_seen=signal_keys_seen,
            )

        observed_signals = sorted(observed_signals, key=lambda item: item.signal_id)

        dimension_profiles: list[StateDimensionProfile] = []
        inferred_labels: list[InferredStateLabel] = []
        confidence_rows: list[DimensionConfidence] = []

        for spec in STATE_DIMENSION_SPECS:
            acc = accumulators[spec.dimension]
            low_hits = acc.low_hits
            high_hits = acc.high_hits
            total_hits = low_hits + high_hits
            # Bayesian smoothing: prevents ±1.0 from a single signal hit
            raw_score = (high_hits - low_hits) / (total_hits + SCORE_SMOOTHING_K)
            score = round(max(-SCORE_CAP, min(SCORE_CAP, raw_score)), 4)

            if score <= -0.25:
                label = spec.low_anchor
            elif score >= 0.25:
                label = spec.high_anchor
            else:
                label = f"between {spec.low_anchor} and {spec.high_anchor}"

            confidence = _dimension_confidence(total_hits)
            rationale = (
                f"Detected {low_hits} low-anchor and {high_hits} high-anchor cues "
                f"for {spec.dimension.replace('_', ' ')}."
            )

            evidence_spans = (acc.evidence_spans or [])[:12]
            signal_ids = (acc.signal_ids or [])[:20]

            dimension_profiles.append(
                StateDimensionProfile(
                    dimension=spec.dimension,
                    score=score,
                    low_anchor=spec.low_anchor,
                    high_anchor=spec.high_anchor,
                    label=label,
                    evidence_spans=evidence_spans,
                )
            )
            inferred_labels.append(
                InferredStateLabel(
                    dimension=spec.dimension,
                    label=label,
                    score=score,
                    rationale=rationale,
                    supporting_signal_ids=signal_ids,
                    confidence=confidence,
                )
            )
            confidence_rows.append(
                DimensionConfidence(dimension=spec.dimension, value=confidence)
            )

        overall_confidence = round(
            sum(row.value for row in confidence_rows) / max(len(confidence_rows), 1),
            4,
        )

        provenance_spans = []
        for signal in observed_signals:
            provenance_spans.extend(signal.evidence_spans)

        provenance = Provenance(
            chunk_ids=_unique(request.chunk_ids) or [fallback_chunk_id],
            spans=provenance_spans[:200],
        )

        analysis_id = _stable_id(
            "state",
            {
                "entry_id": request.entry_id,
                "text": normalized_text,
                "chunk_ids": request.chunk_ids,
                "schema": STATE_SCHEMA_VERSION,
            },
        )

        return StateLabelResponse(
            analysis_id=analysis_id,
            entry_id=request.entry_id,
            state_profile=StateProfile(dimensions=dimension_profiles),
            observed_text_signals=observed_signals,
            inferred_state_labels=inferred_labels,
            confidence=StateConfidence(overall=overall_confidence, by_dimension=confidence_rows),
            provenance=provenance,
            version=AnalysisVersion(
                schema_version=STATE_SCHEMA_VERSION,
                prompt_version=STATE_PROMPT_VERSION,
                model_version=STATE_MODEL_VERSION,
                mock=False,
            ),
        )

    def _collect_signals(
        self,
        text: str,
        original_text: str,
        spec: DimensionSpec,
        direction: SignalDirection,
        rules: tuple[SignalRule, ...],
        fallback_chunk_id: str,
        source_file: str | None,
        accumulator: _DimensionAccumulator,
        observed_signals: list[ObservedTextSignal],
        signal_keys_seen: set[tuple[StateDimension, str, int, int]],
    ) -> None:
        for rule in rules:
            pattern = re.compile(rf"(?<!\\w){re.escape(rule.phrase.lower())}(?!\\w)")
            for match in pattern.finditer(text):
                start, end = match.span()
                dedupe_key = (spec.dimension, rule.phrase, start, end)
                if dedupe_key in signal_keys_seen:
                    continue

                # Negation check: skip signal if preceded by negation word
                if _is_negated(text, start):
                    continue

                signal_keys_seen.add(dedupe_key)

                span = SourceSpan(
                    chunk_id=fallback_chunk_id,
                    source_file=source_file,
                    start_char=start,
                    end_char=end,
                    excerpt=original_text[start:min(len(original_text), end + 80)],
                )

                signal_id = _stable_id(
                    "sig",
                    {
                        "dimension": spec.dimension,
                        "phrase": rule.phrase,
                        "direction": direction,
                        "start": start,
                        "end": end,
                    },
                    digest_len=10,
                )

                weight = _signal_weight(rule.category)
                observed_signals.append(
                    ObservedTextSignal(
                        signal_id=signal_id,
                        signal=rule.phrase,
                        category=rule.category,
                        direction=direction,
                        dimensions=[spec.dimension],
                        weight=weight,
                        evidence_spans=[span],
                    )
                )

                if direction == "low":
                    accumulator.low_hits += 1
                else:
                    accumulator.high_hits += 1

                assert accumulator.signal_ids is not None
                assert accumulator.evidence_spans is not None
                accumulator.signal_ids.append(signal_id)
                accumulator.evidence_spans.append(span)


def create_state_labeler() -> StateLabeler:
    """Factory for pluggable state labeler backends.

    TODO: Load backend from env/config and support local model classifier.
    """

    return DeterministicStateLabeler()


_STATE_LABELER: StateLabeler = create_state_labeler()


def label_state(request: StateLabelRequest) -> StateLabelResponse:
    from .provider_registry import get_provider_registry
    from .oneshot_providers import label_state_via_provider

    registry = get_provider_registry()
    if registry.default_state_label != "mock":
        provider = registry.state_label_providers.get(registry.default_state_label)
        if provider is not None:
            try:
                return label_state_via_provider(request, provider)
            except Exception:
                logger.warning(
                    "oneshot_state_label_provider_failed",
                    extra={
                        "event": "oneshot_state_label_provider_failed",
                        "provider": registry.default_state_label,
                        "entry_id": request.entry_id,
                    },
                    exc_info=True,
                )

    return _STATE_LABELER.label(request)


def _dimension_confidence(total_hits: int) -> float:
    if total_hits <= 0:
        return 0.35
    return round(min(0.95, 0.45 + min(total_hits, 8) * 0.06), 4)


def _signal_weight(category: SignalCategory) -> float:
    if category in {"modal", "temporal"}:
        return 0.82
    if category in {"relational", "structural"}:
        return 0.76
    if category == "pattern":
        return 0.74
    return 0.7


def _is_negated(text: str, match_start: int) -> bool:
    """Check if a signal match is preceded by a negation word within a small window.

    Prevents "I can't do this" from matching "can" as high-agency, or
    "not happy" from matching "happy" as high-valence.
    """
    prefix = text[max(0, match_start - 50):match_start]
    words = prefix.split()
    check_words = words[-_NEGATION_WINDOW:] if len(words) >= _NEGATION_WINDOW else words
    for w in check_words:
        # Normalize curly/straight apostrophes and strip punctuation
        normalized = w.replace("\u2019", "'").replace("\u2018", "'").lower().strip(".,;:!?\"()")
        if normalized in _NEGATION_WORDS:
            return True
    return False


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _stable_id(prefix: str, payload: dict, digest_len: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:digest_len]
    return f"{prefix}-{digest}"
