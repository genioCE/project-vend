"""
Query decomposition via Claude API.

Takes a natural language query and returns typed fragments with primary mass
identification. Claude performs the decomposition; embeddings are computed
separately.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv

from fragments import DecompositionResult, Fragment, FragmentType

# Load .env from project root (override=True in case shell has empty var)
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

DECOMPOSITION_PROMPT = """\
You are a query decomposition engine for a personal writing corpus analysis system.

Given a natural language query, decompose it into typed semantic fragments. Each fragment is a meaningful unit that will be used to activate analysis tools via semantic similarity.

## Fragment Types

- **concept**: Abstract ideas, themes, philosophical constructs (e.g., silence, sovereignty, shame, trust, discipline, recovery)
- **entity**: Named people, places, practices, organizations (e.g., Kyle, Blocworks, climbing, StarSpace46, Mom)
- **temporal**: Time dimensions, change markers, period references (e.g., change over time, since January, last 3 months, recently, since I started climbing)
- **emotional**: Feelings, states, psychological dimensions (e.g., self-trust, integration, agency, valence, stuck, empowered, fragmented)
- **relational**: Connection structure, influence, tension (e.g., relationship with, tension between, influence of, how X connects to Y)
- **archetypal**: Patterns, roles, mythic structures (e.g., Creator, Healer, Warrior, Sovereign, Integrator)

## Rules

1. Extract ALL meaningful fragments from the query. A single query typically has 2-6 fragments.
2. Each fragment should be a short phrase (1-5 words), not a full sentence.
3. A word/phrase can only appear in ONE fragment — no duplicates.
4. If a fragment could fit multiple types, choose the most specific type.
5. Identify the PRIMARY MASS — the fragment the query is fundamentally about. This is the subject being investigated, not the lens through which it's being viewed.
6. Empty categories are signal — don't force fragments into categories where they don't belong.
7. Infer implicit fragments only when strongly implied (e.g., "how have I changed" implies temporal: "change over time").

## Output Format

Return ONLY valid JSON with this structure:
{
  "fragments": [
    {"type": "<fragment_type>", "text": "<fragment_text>"},
    ...
  ],
  "primary_mass_index": <index of primary mass fragment in the array>,
  "reasoning": "<1-2 sentences explaining why you chose this primary mass>"
}"""


def decompose(query: str, model: str = "claude-haiku-4-5-20251001") -> DecompositionResult:
    """
    Decompose a query into typed fragments using Claude.

    Returns a DecompositionResult with fragments and primary mass identification.
    Embeddings are NOT computed here — call embed_decomposition() separately.
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=DECOMPOSITION_PROMPT,
        messages=[{"role": "user", "content": query}],
    )

    text = response.content[0].text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3].strip()

    data = json.loads(text)

    fragments = []
    for f in data["fragments"]:
        ftype = FragmentType(f["type"])
        fragments.append(Fragment(type=ftype, text=f["text"]))

    return DecompositionResult(
        fragments=fragments,
        primary_mass_index=data["primary_mass_index"],
        claude_reasoning=data.get("reasoning", ""),
    )


def embed_decomposition(
    result: DecompositionResult,
    query: str,
    model=None,
) -> DecompositionResult:
    """
    Embed all fragments and the full query using sentence-transformers.
    Mutates and returns the DecompositionResult.
    """
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")

    texts = [f.text for f in result.fragments] + [query]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    for i, fragment in enumerate(result.fragments):
        fragment.embedding = embeddings[i]

    result.query_embedding = embeddings[-1]
    return result
