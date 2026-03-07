"""
Core gravity field mathematics.

Computes the gravitational pull between fragment vectors and tool identity
vectors, applies L2 norm composite activation, and determines which tools fire.

The gravity calculation includes the full query vector as an additional pull
source alongside individual fragments. This ensures that tools semantically
aligned with the overall query intent activate even when individual fragment
texts are narrow (e.g., proper nouns).

Activation uses adaptive gap detection: tools are sorted by composite score,
and the largest relative drop in the top half identifies the natural cut point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fragments import DecompositionResult
from tool_identities import ALWAYS_ACTIVE_TOOLS, META_TOOLS, TOOL_NAMES


@dataclass
class ActivatedTool:
    name: str
    composite_score: float
    primary_pull: float
    query_pull: float  # pull from full query vector
    pulls: list[float]  # per-fragment pulls
    is_meta: bool


@dataclass
class GravityField:
    """Full gravity field computation result."""
    # Raw pull matrix: shape (n_tools, n_fragments + 1) — last column is full query
    pull_matrix: np.ndarray
    # Composite activation per tool: shape (n_tools,)
    composite_scores: np.ndarray
    # Primary mass identification
    claude_primary_index: int
    centroid_primary_index: int
    primary_agreement: bool
    centroid_similarities: np.ndarray  # cos(Q, F_i) for each fragment
    # Activated tools after threshold
    activated: list[ActivatedTool] = field(default_factory=list)
    # All tools with scores (for display)
    all_tools: list[ActivatedTool] = field(default_factory=list)
    # Adaptive threshold info
    adaptive_cutoff: float = 0.0
    gap_position: int = 0


def compute_pull_matrix(
    identity_vectors: np.ndarray,
    all_vectors: np.ndarray,
) -> np.ndarray:
    """
    Compute the full pull matrix: (n_tools, n_vectors).
    All vectors assumed L2-normalized; dot product = cosine similarity.
    """
    return np.asarray(identity_vectors, dtype=np.float64) @ np.asarray(all_vectors, dtype=np.float64).T


def compute_composite_activation(pulls: np.ndarray) -> np.ndarray:
    """
    L2 norm composite activation per tool.

    Input: pull_matrix of shape (n_tools, n_vectors)
    Output: composite scores of shape (n_tools,)

    L2 norm: sqrt(sum(pull_i^2)) — multiple moderate pulls compound
    meaningfully, but with diminishing returns.
    """
    return np.sqrt(np.sum(pulls ** 2, axis=1))


def identify_primary_mass(
    query_vector: np.ndarray,
    fragment_vectors: np.ndarray,
) -> tuple[int, np.ndarray]:
    """
    Centroid distance primary mass identification.
    Returns (index of closest fragment, similarity scores for all fragments).
    """
    q = np.asarray(query_vector, dtype=np.float64)
    f = np.asarray(fragment_vectors, dtype=np.float64)
    similarities = f @ q
    return int(np.argmax(similarities)), similarities.astype(np.float32)


def find_activation_cutoff(
    scores: list[float],
    min_tools: int = 3,
    max_tools: int = 10,
) -> tuple[float, int]:
    """
    Find the natural activation cutoff using gap detection.

    Sorts scores descending, then looks for the largest relative drop
    between consecutive tools in the range [min_tools, max_tools].
    Returns (cutoff_score, gap_position).
    """
    sorted_scores = sorted(scores, reverse=True)

    if len(sorted_scores) <= min_tools:
        return 0.0, len(sorted_scores)

    # Look for largest relative gap in the eligible range
    best_gap = 0.0
    best_pos = min_tools  # default: activate min_tools

    for i in range(min_tools - 1, min(max_tools, len(sorted_scores) - 1)):
        current = sorted_scores[i]
        next_val = sorted_scores[i + 1]

        if current == 0:
            continue

        # Relative gap: how much does the score drop as a fraction of the current?
        relative_gap = (current - next_val) / current

        if relative_gap > best_gap:
            best_gap = relative_gap
            best_pos = i + 1  # activate everything up to and including position i

    # The cutoff is the midpoint between the last activated and first excluded
    cutoff = (sorted_scores[best_pos - 1] + sorted_scores[min(best_pos, len(sorted_scores) - 1)]) / 2

    return cutoff, best_pos


def compute_gravity_field(
    identity_vectors: np.ndarray,
    decomposition: DecompositionResult,
    min_tools: int = 3,
    max_tools: int = 10,
) -> GravityField:
    """
    Full gravity field computation with adaptive activation.

    1. Build vector set: all fragments + full query vector
    2. Compute pull matrix (22 tools x (N fragments + 1 query))
    3. Compute composite activation (L2 norm over ALL pulls including query)
    4. Identify primary mass (centroid distance, fragments only)
    5. Apply adaptive gap-based activation threshold

    The full query vector participates in the gravity calculation because
    individual fragments (especially proper nouns) may embed far from tool
    descriptions, while the full query carries richer semantic context.
    """
    fragment_vectors = np.array(
        [f.embedding for f in decomposition.fragments],
        dtype=np.float32,
    )
    query_vector = decomposition.query_embedding

    # Build combined vector set: fragments + full query
    all_vectors = np.vstack([fragment_vectors, query_vector.reshape(1, -1)])

    # Step 1: Pull matrix (22 tools x (N+1) vectors)
    pull_matrix = compute_pull_matrix(identity_vectors, all_vectors)

    # Step 2: Composite activation over all pull sources
    composite_scores = compute_composite_activation(pull_matrix)

    # Step 3: Primary mass (centroid distance, fragments only)
    centroid_idx, centroid_sims = identify_primary_mass(query_vector, fragment_vectors)

    # Step 4: Build tool list
    primary_idx = decomposition.primary_mass_index
    n_fragments = len(decomposition.fragments)
    all_tools_list = []

    for i, name in enumerate(TOOL_NAMES):
        fragment_pulls = pull_matrix[i, :n_fragments].tolist()
        query_pull = float(pull_matrix[i, n_fragments])  # full query pull
        primary_pull = fragment_pulls[primary_idx]
        composite = float(composite_scores[i])
        is_meta = name in META_TOOLS

        tool = ActivatedTool(
            name=name,
            composite_score=composite,
            primary_pull=primary_pull,
            query_pull=query_pull,
            pulls=fragment_pulls,
            is_meta=is_meta,
        )
        all_tools_list.append(tool)

    # Step 5: Adaptive gap detection for gravity-gated tools
    # Exclude meta-tools and always-active tools from the gap calculation
    gated_tools = [t for t in all_tools_list if not t.is_meta and t.name not in ALWAYS_ACTIVE_TOOLS]
    gated_scores = [t.composite_score for t in gated_tools]
    cutoff, gap_pos = find_activation_cutoff(gated_scores, min_tools, max_tools)

    activated = []

    # Always-active tools fire unconditionally (textual grounding)
    for tool in all_tools_list:
        if tool.name in ALWAYS_ACTIVE_TOOLS:
            activated.append(tool)

    # Gravity-gated tools fire above the adaptive cutoff
    for tool in gated_tools:
        if tool.composite_score >= cutoff:
            activated.append(tool)

    # Meta-tools: only activate if they score in the top 3 of ALL tools
    # (including non-meta), indicating explicit meta-intent
    top_3_threshold = sorted([t.composite_score for t in all_tools_list], reverse=True)[2]
    for tool in all_tools_list:
        if tool.is_meta and tool.composite_score >= top_3_threshold:
            activated.append(tool)

    # Sort by composite score (descending)
    activated.sort(key=lambda t: t.composite_score, reverse=True)
    all_tools_list.sort(key=lambda t: t.composite_score, reverse=True)

    return GravityField(
        pull_matrix=pull_matrix,
        composite_scores=composite_scores,
        claude_primary_index=primary_idx,
        centroid_primary_index=centroid_idx,
        primary_agreement=(primary_idx == centroid_idx),
        centroid_similarities=centroid_sims,
        activated=activated,
        all_tools=all_tools_list,
        adaptive_cutoff=cutoff,
        gap_position=gap_pos,
    )
