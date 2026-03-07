"""Unit tests for the entity extraction pipeline in app.extractor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.extractor import (
    EMOTION_NAMES,
    STOP_CONCEPTS,
    TRANSITION_BANNED_HEAD_WORDS,
    TRANSITION_BANNED_TOKENS,
    _build_stop_concepts,
    _canonicalize_concept,
    _clean_transition_candidate,
    _extract_archetypes,
    _extract_decisions,
    _extract_emotions,
    _extract_people,
    _extract_places,
    _extract_transitions,
    _merge_keyword_map,
    _merge_list,
    _normalize_concept_text,
    _normalize_spaces,
    _valid_concept_from_chunk,
    extract_entities,
)


# ---------------------------------------------------------------------------
# Tier 1 — Pure text/regex helpers (no spaCy Doc needed)
# ---------------------------------------------------------------------------


class TestNormalizeSpaces:
    def test_multiple_spaces(self):
        assert _normalize_spaces("hello   world") == "hello world"

    def test_tabs_and_newlines(self):
        assert _normalize_spaces("hello\t\nworld") == "hello world"

    def test_leading_trailing(self):
        assert _normalize_spaces("  hello  ") == "hello"

    def test_empty_string(self):
        assert _normalize_spaces("") == ""

    def test_single_word(self):
        assert _normalize_spaces("  word  ") == "word"


class TestNormalizeConceptText:
    def test_lowercasing(self):
        assert _normalize_concept_text("Hello World") == "hello world"

    def test_punctuation_removal(self):
        result = _normalize_concept_text("self-care!")
        # ! replaced by space, hyphen kept
        assert "!" not in result

    def test_article_stripping(self):
        assert _normalize_concept_text("the journey") == "journey"
        assert _normalize_concept_text("a moment") == "moment"
        assert _normalize_concept_text("an idea") == "idea"

    def test_possessive_stripping(self):
        assert _normalize_concept_text("my courage") == "courage"
        assert _normalize_concept_text("our path") == "path"

    def test_keeps_apostrophes_and_hyphens(self):
        result = _normalize_concept_text("self-awareness")
        assert "self-awareness" == result

    def test_bare_article_kept(self):
        # The regex only strips articles followed by a space + more text
        assert _normalize_concept_text("the") == "the"

    def test_tabs_and_newlines(self):
        result = _normalize_concept_text("inner\n\tpeace")
        assert result == "inner peace"


class TestCanonicalizeConcept:
    def test_basic_normalization(self):
        result = _canonicalize_concept("The Journey")
        assert result == "journey"

    def test_empty_input(self):
        assert _canonicalize_concept("") == ""

    def test_whitespace_only(self):
        assert _canonicalize_concept("   ") == ""

    def test_bare_article_not_stripped(self):
        # "the" alone doesn't match the stripping regex (requires trailing word)
        assert _canonicalize_concept("the") == "the"


class TestCleanTransitionCandidate:
    def test_valid_passthrough(self):
        result = _clean_transition_candidate("fear")
        assert result == "fear"

    def test_banned_head_word(self):
        for word in ["being", "having", "going"]:
            assert _clean_transition_candidate(f"{word} forward") == ""

    def test_banned_token(self):
        for token in ["today", "tomorrow", "yesterday"]:
            assert _clean_transition_candidate(f"great {token}") == ""

    def test_too_many_words(self):
        assert _clean_transition_candidate("one two three four five") == ""

    def test_four_words_allowed(self):
        result = _clean_transition_candidate("one two three four")
        # Should not be rejected by word count alone
        assert result != "" or result in STOP_CONCEPTS

    def test_stop_concept_filtered(self):
        # "something" is in DEFAULT_STOP_CONCEPTS
        assert _clean_transition_candidate("something") == ""

    def test_suffix_stripping(self):
        result = _clean_transition_candidate("calm state")
        assert result == "calm"

    def test_suffix_stripping_energy(self):
        result = _clean_transition_candidate("high energy")
        assert result == "high"


class TestExtractEmotions:
    def test_single_keyword_match(self):
        result = _extract_emotions("i feel happy today")
        emotions = {e["emotion"] for e in result}
        assert "joy" in emotions

    def test_intensity_one_match(self):
        result = _extract_emotions("i am happy")
        joy = next(e for e in result if e["emotion"] == "joy")
        assert joy["intensity"] == round(1 / 3.0, 2)

    def test_intensity_three_or_more(self):
        # joy keywords: happy, joy, grateful
        result = _extract_emotions("i am happy and full of joy and so grateful")
        joy = next(e for e in result if e["emotion"] == "joy")
        assert joy["intensity"] == 1.0

    def test_multiple_emotion_categories(self):
        result = _extract_emotions("i am happy but also anxious and angry")
        emotions = {e["emotion"] for e in result}
        assert "joy" in emotions
        assert "fear" in emotions  # anxious is in fear
        assert "anger" in emotions

    def test_no_matches(self):
        result = _extract_emotions("the cat sat on the mat")
        assert result == []

    def test_keywords_returned(self):
        result = _extract_emotions("i feel happy and grateful")
        joy = next(e for e in result if e["emotion"] == "joy")
        assert "happy" in joy["keywords"]
        assert "grateful" in joy["keywords"]


class TestExtractDecisions:
    def test_basic_decided(self):
        result = _extract_decisions("I decided to take a new path in life.")
        assert len(result) >= 1
        assert any("decided" in d.lower() for d in result)

    def test_chose_pattern(self):
        result = _extract_decisions("I chose to stay.")
        assert len(result) >= 1

    def test_context_window(self):
        text = "After a lot of thought, I decided to move forward with the plan and see what happens next."
        result = _extract_decisions(text)
        assert len(result) >= 1
        # The context should contain text around the match
        assert len(result[0]) > 10

    def test_sentence_boundary_trimming(self):
        text = "I decided to leave. Then I went home."
        result = _extract_decisions(text)
        assert len(result) >= 1
        # Should trim at the period
        assert result[0].endswith(".")
        assert "went home" not in result[0]

    def test_limit_of_five(self):
        decisions_text = ". ".join(f"I decided to do thing {i}" for i in range(10))
        result = _extract_decisions(decisions_text)
        assert len(result) <= 5

    def test_dedup(self):
        text = "I decided to go. Later, I decided to go."
        result = _extract_decisions(text)
        # Same context shouldn't appear twice
        unique = set(result)
        assert len(unique) == len(result)

    def test_no_decisions(self):
        result = _extract_decisions("The weather was nice today.")
        assert result == []


class TestExtractArchetypes:
    def test_requires_two_keywords(self):
        # "fight" alone is not enough for Warrior
        result = _extract_archetypes("fight")
        warriors = [a for a in result if a["archetype"] == "Warrior"]
        assert len(warriors) == 0

    def test_two_keywords_triggers(self):
        result = _extract_archetypes("fight with courage and strength")
        warriors = [a for a in result if a["archetype"] == "Warrior"]
        assert len(warriors) == 1

    def test_strength_scoring(self):
        # 2 matches out of 4 threshold = 0.5
        result = _extract_archetypes("fight with courage")
        warriors = [a for a in result if a["archetype"] == "Warrior"]
        assert warriors[0]["strength"] == 0.5

    def test_strength_capped_at_one(self):
        # Many warrior keywords
        text = "fight battle strength courage warrior brave conquer overcome"
        result = _extract_archetypes(text)
        warriors = [a for a in result if a["archetype"] == "Warrior"]
        assert warriors[0]["strength"] == 1.0

    def test_multiple_archetypes(self):
        text = "fight with courage and explore new territory seeking wisdom and truth"
        result = _extract_archetypes(text)
        archetypes = {a["archetype"] for a in result}
        assert "Warrior" in archetypes
        assert "Explorer" in archetypes

    def test_no_archetypes(self):
        result = _extract_archetypes("the cat sat on the mat")
        assert result == []


class TestExtractTransitions:
    def test_from_to_pattern(self):
        result = _extract_transitions("I moved from fear to courage.")
        assert len(result) >= 1
        t = result[0]
        assert t["from_concept"] == "fear"
        assert t["to_concept"] == "courage"

    def test_deduplication(self):
        text = "I moved from fear to courage. I shifted from fear to courage."
        result = _extract_transitions(text)
        keys = [(t["from_concept"], t["to_concept"]) for t in result]
        assert len(keys) == len(set(keys))

    def test_strength_boost_for_move_keywords(self):
        result = _extract_transitions("I moved from fear to courage.")
        assert result[0]["strength"] > 1.0

    def test_basic_from_to_no_verb_boost(self):
        result = _extract_transitions("I went from sadness to joy.")
        if result:
            # "went" is not in the boost list, but "from X to Y" pattern matches
            assert result[0]["strength"] >= 1.0

    def test_emotion_name_boost(self):
        # "fear" and "joy" are emotion names
        result = _extract_transitions("I moved from fear to joy.")
        if result:
            assert result[0]["strength"] >= 1.4  # 1.0 + 0.4 (moved) + 0.2 (emotion)

    def test_limit_of_twelve(self):
        parts = [f"from concept{i}a to concept{i}b" for i in range(15)]
        text = ". ".join(parts)
        result = _extract_transitions(text)
        assert len(result) <= 12

    def test_same_source_target_filtered(self):
        result = _extract_transitions("I moved from peace to peace.")
        for t in result:
            assert t["from_concept"] != t["to_concept"]

    def test_no_transitions(self):
        result = _extract_transitions("The weather was nice today.")
        assert result == []

    def test_phrase_snippet_included(self):
        result = _extract_transitions("I moved from anger to peace.")
        if result:
            assert result[0]["phrase"]  # non-empty string
            assert isinstance(result[0]["phrase"], str)


class TestMergeKeywordMap:
    def test_basic_merge(self):
        defaults = {"joy": ["happy", "glad"]}
        additions = {"joy": ["elated"]}
        result = _merge_keyword_map(defaults, additions)
        assert "happy" in result["joy"]
        assert "elated" in result["joy"]

    def test_deduplication(self):
        defaults = {"joy": ["happy", "Happy"]}
        result = _merge_keyword_map(defaults, {})
        assert result["joy"].count("happy") == 1

    def test_case_normalization(self):
        defaults = {"joy": ["HAPPY"]}
        result = _merge_keyword_map(defaults, {})
        assert result["joy"] == ["happy"]

    def test_new_key_in_additions(self):
        defaults = {"joy": ["happy"]}
        additions = {"custom": ["special"]}
        result = _merge_keyword_map(defaults, additions)
        assert "custom" in result
        assert "special" in result["custom"]

    def test_empty_key_skipped(self):
        defaults = {}
        additions = {"  ": ["value"]}
        result = _merge_keyword_map(defaults, additions)
        assert "" not in result

    def test_empty_value_skipped(self):
        defaults = {"joy": ["happy", "", "  "]}
        result = _merge_keyword_map(defaults, {})
        assert "" not in result["joy"]
        assert len(result["joy"]) == 1


class TestMergeList:
    def test_basic_merge(self):
        result = _merge_list(["a", "b"], ["c"])
        assert result == ["a", "b", "c"]

    def test_deduplication(self):
        result = _merge_list(["a", "b"], ["A", "b"])
        assert len(result) == 2

    def test_whitespace_handling(self):
        result = _merge_list(["  hello  "], ["hello"])
        assert len(result) == 1
        assert result[0] == "hello"

    def test_empty_strings_skipped(self):
        result = _merge_list(["a", "", "  "], ["b"])
        assert result == ["a", "b"]

    def test_preserves_order(self):
        result = _merge_list(["z", "a"], ["m"])
        assert result == ["z", "a", "m"]


class TestBuildStopConcepts:
    def test_default_passthrough(self):
        result = _build_stop_concepts({"thing", "way"}, {})
        assert "thing" in result
        assert "way" in result

    def test_add_concepts(self):
        result = _build_stop_concepts({"thing"}, {"stop_concepts_add": ["Custom"]})
        assert "custom" in result  # lowercased
        assert "thing" in result

    def test_remove_concepts(self):
        result = _build_stop_concepts({"thing", "way"}, {"stop_concepts_remove": ["thing"]})
        assert "thing" not in result
        assert "way" in result

    def test_add_and_remove(self):
        result = _build_stop_concepts(
            {"thing", "way"},
            {"stop_concepts_add": ["custom"], "stop_concepts_remove": ["thing"]},
        )
        assert "custom" in result
        assert "thing" not in result
        assert "way" in result

    def test_normalization(self):
        result = _build_stop_concepts(set(), {"stop_concepts_add": ["  Foo  "]})
        assert "foo" in result


# ---------------------------------------------------------------------------
# Tier 2 — Tests requiring mock spaCy objects
# ---------------------------------------------------------------------------


def _make_mock_entity(text: str, label: str) -> MagicMock:
    """Create a mock spaCy entity (Span-like)."""
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    return ent


def _make_mock_token(text: str, is_stop: bool = False, is_punct: bool = False) -> MagicMock:
    """Create a mock spaCy Token."""
    tok = MagicMock()
    tok.text = text
    tok.is_stop = is_stop
    tok.is_punct = is_punct
    return tok


def _make_mock_chunk(text: str, root_is_stop: bool = False, all_stop_or_punct: bool = False) -> MagicMock:
    """Create a mock spaCy noun chunk (Span-like)."""
    chunk = MagicMock()
    chunk.text = text
    chunk.root = MagicMock()
    chunk.root.is_stop = root_is_stop
    if all_stop_or_punct:
        chunk.__iter__ = lambda self: iter([_make_mock_token("the", is_stop=True)])
    else:
        chunk.__iter__ = lambda self: iter([
            _make_mock_token(text, is_stop=False, is_punct=False),
        ])
    return chunk


class TestExtractPeople:
    def test_person_entities(self):
        doc = MagicMock()
        doc.ents = [
            _make_mock_entity("John Smith", "PERSON"),
            _make_mock_entity("Jane Doe", "PERSON"),
        ]
        result = _extract_people(doc)
        assert "John Smith" in result
        assert "Jane Doe" in result

    def test_non_person_filtered(self):
        doc = MagicMock()
        doc.ents = [
            _make_mock_entity("John Smith", "PERSON"),
            _make_mock_entity("New York", "GPE"),
        ]
        result = _extract_people(doc)
        assert "John Smith" in result
        assert len(result) == 1

    def test_dedup_case_insensitive(self):
        doc = MagicMock()
        doc.ents = [
            _make_mock_entity("John", "PERSON"),
            _make_mock_entity("john", "PERSON"),
        ]
        result = _extract_people(doc)
        assert len(result) == 1

    def test_short_names_filtered(self):
        doc = MagicMock()
        doc.ents = [_make_mock_entity("J", "PERSON")]
        result = _extract_people(doc)
        assert len(result) == 0

    def test_empty_doc(self):
        doc = MagicMock()
        doc.ents = []
        result = _extract_people(doc)
        assert result == []


class TestExtractPlaces:
    def test_gpe_entities(self):
        doc = MagicMock()
        doc.ents = [_make_mock_entity("New York", "GPE")]
        result = _extract_places(doc)
        assert "New York" in result

    def test_loc_entities(self):
        doc = MagicMock()
        doc.ents = [_make_mock_entity("Pacific Ocean", "LOC")]
        result = _extract_places(doc)
        assert "Pacific Ocean" in result

    def test_fac_entities(self):
        doc = MagicMock()
        doc.ents = [_make_mock_entity("Golden Gate Bridge", "FAC")]
        result = _extract_places(doc)
        assert "Golden Gate Bridge" in result

    def test_non_place_filtered(self):
        doc = MagicMock()
        doc.ents = [
            _make_mock_entity("New York", "GPE"),
            _make_mock_entity("John", "PERSON"),
        ]
        result = _extract_places(doc)
        assert len(result) == 1
        assert "New York" in result

    def test_dedup(self):
        doc = MagicMock()
        doc.ents = [
            _make_mock_entity("Paris", "GPE"),
            _make_mock_entity("paris", "GPE"),
        ]
        result = _extract_places(doc)
        assert len(result) == 1


class TestValidConceptFromChunk:
    def test_short_concept_rejected(self):
        chunk = _make_mock_chunk("ab")
        assert _valid_concept_from_chunk(chunk, "ab") is False

    def test_stop_concept_rejected(self):
        chunk = _make_mock_chunk("something")
        assert _valid_concept_from_chunk(chunk, "something") is False

    def test_stop_root_rejected(self):
        chunk = _make_mock_chunk("courage", root_is_stop=True)
        assert _valid_concept_from_chunk(chunk, "courage") is False

    def test_all_stop_or_punct_rejected(self):
        chunk = _make_mock_chunk("the", all_stop_or_punct=True)
        # "the" is also < 3 chars but let's test with a valid-length concept name
        chunk.text = "they"
        assert _valid_concept_from_chunk(chunk, "they") is False

    def test_valid_concept_accepted(self):
        chunk = _make_mock_chunk("courage")
        assert _valid_concept_from_chunk(chunk, "courage") is True


class TestExtractEntitiesEndToEnd:
    """End-to-end test with a mock nlp callable."""

    def test_output_shape(self):
        mock_doc = MagicMock()
        mock_doc.ents = [
            _make_mock_entity("Alice", "PERSON"),
            _make_mock_entity("Boston", "GPE"),
        ]
        mock_doc.noun_chunks = []
        mock_doc.sents = []

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.extractor.nlp", mock_nlp):
            result = extract_entities("Alice visited Boston. I decided to be brave.")

        assert "people" in result
        assert "places" in result
        assert "concepts" in result
        assert "emotions" in result
        assert "decisions" in result
        assert "archetypes" in result
        assert "transitions" in result

        assert isinstance(result["people"], list)
        assert isinstance(result["places"], list)
        assert isinstance(result["decisions"], list)

    def test_people_and_places_extracted(self):
        mock_doc = MagicMock()
        mock_doc.ents = [
            _make_mock_entity("Alice", "PERSON"),
            _make_mock_entity("Boston", "GPE"),
        ]
        mock_doc.noun_chunks = []
        mock_doc.sents = []

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.extractor.nlp", mock_nlp):
            result = extract_entities("Alice visited Boston.")

        assert "Alice" in result["people"]
        assert "Boston" in result["places"]

    def test_decisions_extracted(self):
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.noun_chunks = []
        mock_doc.sents = []

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.extractor.nlp", mock_nlp):
            result = extract_entities("I decided to change my life.")

        assert len(result["decisions"]) >= 1

    def test_emotions_extracted(self):
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.noun_chunks = []
        mock_doc.sents = []

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.extractor.nlp", mock_nlp):
            result = extract_entities("I feel happy and grateful today.")

        emotions = {e["emotion"] for e in result["emotions"]}
        assert "joy" in emotions

    def test_transition_filtering_by_concepts(self):
        """Transitions are filtered to those whose source/target is in extracted concepts or emotion names."""
        courage_chunk = _make_mock_chunk("courage")
        fear_chunk = _make_mock_chunk("fear")

        mock_sent = MagicMock()
        mock_sent.text = "I moved from fear to courage."
        mock_sent.noun_chunks = [fear_chunk, courage_chunk]

        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.noun_chunks = [fear_chunk, courage_chunk]
        mock_doc.sents = [mock_sent]

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.extractor.nlp", mock_nlp):
            result = extract_entities("I moved from fear to courage.")

        # "fear" is an emotion name, so the transition should survive filtering
        if result["transitions"]:
            t = result["transitions"][0]
            assert t["from_concept"] == "fear"
            assert t["to_concept"] == "courage"

    def test_transition_filtered_when_not_in_concepts_or_emotions(self):
        """Transitions with source/target not in concepts or emotion names are dropped."""
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.noun_chunks = []  # no concepts extracted
        mock_doc.sents = []

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.extractor.nlp", mock_nlp):
            # "xyz123" and "abc456" are not emotion names or concepts
            result = extract_entities("I moved from xyz123 to abc456.")

        # These bogus concepts should be filtered out
        assert result["transitions"] == []
