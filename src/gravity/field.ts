/**
 * Core gravity field mathematics — Project Vend.
 *
 * Adapted from corpus-intelligence. The math is identical:
 * cosine similarity pull matrix → L2 norm composite → adaptive gap detection.
 *
 * Naming changed from "tools" to "agents" throughout.
 */

import type {
  ActivatedAgent,
  DecompositionResult,
  Fragment,
  GravityField,
  AgentReliability,
} from "./types.js";
import { AGENT_NAMES, ALWAYS_ACTIVE_AGENTS, META_AGENTS } from "./identities.js";

const MIN_AGENTS = 1;
const MAX_AGENTS = 4; // We only have 3-4 agents (for now)

/** Dot product (= cosine similarity when L2-normalized). */
export function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Pull matrix: (n_agents × n_vectors).
 * Each entry is cosine similarity between an agent identity vector
 * and a fragment/event vector.
 */
export function computePullMatrix(
  identityVectors: number[][],
  allVectors: number[][]
): number[][] {
  const nAgents = identityVectors.length;
  const nVectors = allVectors.length;
  const matrix: number[][] = new Array(nAgents);

  for (let i = 0; i < nAgents; i++) {
    matrix[i] = new Array(nVectors);
    for (let j = 0; j < nVectors; j++) {
      matrix[i][j] = dotProduct(identityVectors[i], allVectors[j]);
    }
  }

  return matrix;
}

/**
 * L2 norm composite activation per agent.
 * sqrt(sum(pull_i^2)) — multiple moderate pulls compound meaningfully.
 */
export function computeCompositeActivation(pullMatrix: number[][]): number[] {
  return pullMatrix.map((row) => {
    let sumSq = 0;
    for (const val of row) {
      sumSq += val * val;
    }
    return Math.sqrt(sumSq);
  });
}

/**
 * Centroid distance primary mass identification.
 * Returns the fragment index closest to the full event vector.
 */
export function identifyPrimaryMass(
  queryVector: number[],
  fragmentVectors: number[][]
): number {
  let bestIdx = 0;
  let bestSim = -Infinity;

  for (let i = 0; i < fragmentVectors.length; i++) {
    const sim = dotProduct(queryVector, fragmentVectors[i]);
    if (sim > bestSim) {
      bestSim = sim;
      bestIdx = i;
    }
  }

  return bestIdx;
}

/**
 * Adaptive gap detection for activation cutoff.
 */
export function findActivationCutoff(
  scores: number[],
  minAgents: number = MIN_AGENTS,
  maxAgents: number = MAX_AGENTS
): { cutoff: number; position: number } {
  const sorted = [...scores].sort((a, b) => b - a);

  if (sorted.length <= minAgents) {
    return { cutoff: 0, position: sorted.length };
  }

  let bestGap = 0;
  let bestPos = minAgents;

  for (
    let i = minAgents - 1;
    i < Math.min(maxAgents, sorted.length - 1);
    i++
  ) {
    const current = sorted[i];
    const next = sorted[i + 1];

    if (current === 0) continue;

    const relativeGap = (current - next) / current;

    if (relativeGap > bestGap) {
      bestGap = relativeGap;
      bestPos = i + 1;
    }
  }

  const cutoff =
    (sorted[bestPos - 1] +
      sorted[Math.min(bestPos, sorted.length - 1)]) /
    2;

  return { cutoff, position: bestPos };
}

/**
 * Apply learned reliability bias to composite scores.
 */
export function applyReliabilityBias(
  compositeScores: number[],
  agentNames: string[],
  reliabilityMap: Map<string, AgentReliability>,
  fragments: Fragment[],
  options?: { minActivations?: number; floor?: number }
): number[] {
  const minActivations = options?.minActivations ?? 10;
  const floor = options?.floor ?? 0.65;

  const fragmentTypes = new Set(fragments.map((f) => f.type));

  return compositeScores.map((score, i) => {
    const agentName = agentNames[i];
    const reliability = reliabilityMap.get(agentName);

    if (!reliability || reliability.total_activations < minActivations) {
      return score;
    }

    let bias: number;
    let fragmentWeightSum = 0;
    let fragmentUsefulSum = 0;

    for (const fragType of fragmentTypes) {
      const fragStats = reliability.by_fragment_type[fragType];
      if (fragStats && fragStats.activations >= 3) {
        fragmentWeightSum += fragStats.activations;
        fragmentUsefulSum += fragStats.useful_rate * fragStats.activations;
      }
    }

    if (fragmentWeightSum > 0) {
      bias = fragmentUsefulSum / fragmentWeightSum;
    } else {
      bias = reliability.reliability_score;
    }

    bias = Math.max(floor, bias);
    return score * bias;
  });
}

/**
 * Full gravity field computation.
 */
export function computeGravityField(
  identityVectors: number[][],
  decomposition: DecompositionResult,
  reliabilityMap: Map<string, AgentReliability> = new Map()
): GravityField {
  const fragmentVectors = decomposition.fragments
    .map((f) => f.embedding)
    .filter((e): e is number[] => e != null);
  const queryVector = decomposition.query_embedding;

  if (!queryVector || fragmentVectors.length === 0) {
    throw new Error("Decomposition must be embedded before computing gravity field");
  }

  const allVectors = [...fragmentVectors, queryVector];

  // Pull matrix
  const pullMatrix = computePullMatrix(identityVectors, allVectors);

  // Composite activation (raw)
  const compositeScores = computeCompositeActivation(pullMatrix);

  // Apply reliability bias
  const biasedCompositeScores = applyReliabilityBias(
    compositeScores,
    AGENT_NAMES,
    reliabilityMap,
    decomposition.fragments
  );

  // Primary mass
  const centroidPrimaryIndex = identifyPrimaryMass(queryVector, fragmentVectors);
  const primaryIdx = decomposition.primary_mass_index;
  const nFragments = decomposition.fragments.length;

  // Build agent list
  const allAgents: ActivatedAgent[] = AGENT_NAMES.map((name, i) => {
    const fragmentPulls = pullMatrix[i].slice(0, nFragments);
    const queryPull = pullMatrix[i][nFragments];
    const primaryPull = fragmentPulls[primaryIdx] ?? 0;

    return {
      name,
      composite_score: biasedCompositeScores[i],
      primary_pull: primaryPull,
      query_pull: queryPull,
      is_meta: META_AGENTS.has(name),
      is_always_active: ALWAYS_ACTIVE_AGENTS.has(name),
    };
  });

  // Adaptive gap detection
  const gatedAgents = allAgents.filter(
    (a) => !a.is_meta && !ALWAYS_ACTIVE_AGENTS.has(a.name)
  );
  const gatedScores = gatedAgents.map((a) => a.composite_score);
  const { cutoff } = findActivationCutoff(gatedScores);

  const activated: ActivatedAgent[] = [];

  // Always-active agents
  for (const agent of allAgents) {
    if (ALWAYS_ACTIVE_AGENTS.has(agent.name)) {
      activated.push(agent);
    }
  }

  // Gravity-gated agents
  for (const agent of gatedAgents) {
    if (agent.composite_score >= cutoff) {
      activated.push(agent);
    }
  }

  // Meta agents: only if in top 2
  const sortedScores = allAgents
    .map((a) => a.composite_score)
    .sort((a, b) => b - a);
  const top2Threshold = sortedScores[1] ?? 0;
  for (const agent of allAgents) {
    if (agent.is_meta && agent.composite_score >= top2Threshold) {
      if (!activated.find((a) => a.name === agent.name)) {
        activated.push(agent);
      }
    }
  }

  // Sort by composite score descending
  activated.sort((a, b) => b.composite_score - a.composite_score);

  return {
    pull_matrix: pullMatrix,
    composite_scores: compositeScores,
    biased_composite_scores: biasedCompositeScores,
    primary_mass_index: primaryIdx,
    centroid_primary_index: centroidPrimaryIndex,
    activated,
    adaptive_cutoff: cutoff,
  };
}
