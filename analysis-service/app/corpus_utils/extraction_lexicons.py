"""Shared extraction lexicons for entity, emotion, decision, and archetype detection.

These dictionaries are the single source of truth used by both the analysis-service
(local_extractor.py) and the graph-service (extractor.py).  Keep them in sync by
copying this file to both services' corpus_utils/ directories.
"""

from __future__ import annotations

VALID_ENTITY_TYPES = {"person", "place", "organization", "concept", "spiritual"}

EMOTION_LEXICON: dict[str, list[str]] = {
    "joy": ["happy", "joy", "grateful", "excited", "delighted", "thrilled", "elated", "blessed", "thankful", "cheerful", "wonderful", "alive"],
    "sadness": ["sad", "grief", "loss", "mourning", "heartbroken", "depressed", "sorrowful", "melancholy", "lonely", "disappointed", "heavy"],
    "anger": ["angry", "furious", "frustrated", "irritated", "resentful", "bitter", "enraged", "annoyed", "hostile"],
    "fear": ["afraid", "anxious", "worried", "scared", "terrified", "nervous", "dread", "panic", "uneasy", "overwhelmed"],
    "love": ["love", "compassion", "affection", "tenderness", "warmth", "caring", "devotion", "adoration", "intimacy"],
    "peace": ["peace", "calm", "serene", "tranquil", "still", "centered", "grounded", "quiet", "settled", "stillness"],
    "confusion": ["confused", "uncertain", "lost", "bewildered", "torn", "conflicted", "ambivalent", "unclear"],
    "hope": ["hope", "hopeful", "optimistic", "encouraged", "inspired", "motivated", "determined", "possibility"],
    "shame": ["shame", "guilt", "embarrassed", "regret", "remorse", "humiliated", "ashamed"],
    "pride": ["proud", "accomplished", "confident", "strong", "capable", "worthy", "empowered"],
}

DECISION_PATTERNS: list[str] = [
    r"I decided",
    r"I've decided",
    r"I choose",
    r"I chose",
    r"I committed",
    r"I commit to",
    r"I'm going to",
    r"I need to",
    r"I realize I must",
    r"the decision is",
    r"my decision",
    r"I'm choosing",
    r"I made the choice",
]

ARCHETYPE_PATTERNS: dict[str, list[str]] = {
    "Warrior": ["fight", "battle", "strength", "courage", "warrior", "brave", "conquer", "overcome", "persist", "endure", "discipline", "resilience"],
    "Sage": ["wisdom", "understand", "insight", "clarity", "truth", "knowledge", "learn", "discern", "contemplate", "reflect", "awareness"],
    "Creator": ["create", "build", "make", "express", "art", "craft", "design", "imagine", "invent", "compose", "write", "creative"],
    "Healer": ["heal", "restore", "recover", "mend", "nurture", "care", "soothe", "comfort", "therapy", "wholeness", "integration"],
    "Explorer": ["explore", "discover", "adventure", "journey", "wander", "seek", "search", "venture", "curious", "new territory"],
    "Lover": ["love", "connect", "intimate", "passion", "heart", "soul", "beloved", "embrace", "together", "union", "vulnerability"],
    "Ruler": ["lead", "order", "control", "structure", "responsibility", "authority", "power", "manage", "organize", "sovereignty"],
    "Magician": ["transform", "change", "shift", "evolve", "transmute", "alchemy", "metamorphosis", "breakthrough", "transcend"],
}

STOP_CONCEPTS: set[str] = {
    "thing", "things", "way", "ways", "time", "times", "day", "days", "lot",
    "lots", "bit", "part", "kind", "something", "everything", "nothing",
    "anyone", "someone", "everybody", "today", "yesterday", "tomorrow",
    "morning", "evening", "night", "week", "month", "year", "moment",
    "sense", "stuff", "place", "point", "idea", "fact", "question",
    "answer", "problem", "i", "me", "my", "mine", "myself",
}

TRANSITION_PATTERNS: list[str] = [
    r"\b(?:move|moved|moving|shift|shifted|shifting|transition|transitioned|transitioning|turn|turned|turning)\s+from\s+(?P<source>[a-z][a-z\s'/-]{1,40}?)\s+(?:to|into|toward|towards)\s+(?P<target>[a-z][a-z\s'/-]{1,40}?)(?=[,.;!?\n]|$)",
    r"\bfrom\s+(?P<source>[a-z][a-z\s'/-]{1,40}?)\s+to\s+(?P<target>[a-z][a-z\s'/-]{1,40}?)(?=[,.;!?\n]|$)",
]
