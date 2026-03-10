/**
 * Core gravity field mathematics.
 *
 * Port of gravity/gravity_field.py to TypeScript.
 * Computes gravitational pull between fragment vectors and tool identity
 * vectors, applies L2 norm composite activation, and determines which tools fire.
 */

import type {
  ActivatedTool,
  DecompositionResult,
  Fragment,
  GravityField,
  ToolReliability,
} from "./types.js";
import { TOOL_NAMES, ALWAYS_ACTIVE_TOOLS, META_TOOLS } from "./identities.js";
import { getReliabilityMap, getOutcomeCount } from "./ledger.js";

const MIN_TOOLS = 3;
const MAX_TOOLS = 10;

/**
 * Compute adaptive reliability floor based on ledger size.
 *
 * With sparse data, we use a high floor (0.65) to avoid over-penalizing
 * tools we don't have enough signal on. As data accumulates, we lower
 * the floor to let the reliability signal dominate.
 */
function getAdaptiveFloor(): number {
  const outcomeCount = getOutcomeCount();
  if (outcomeCount < 50) return 0.65;
  if (outcomeCount < 100) return 0.55;
  return 0.50;
}

/** Dot product of two vectors (cosine similarity when L2-normalized). */
export function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Compute pull matrix: (n_tools × n_vectors).
 * Each entry is cosine similarity between a tool identity vector and a
 * fragment/query vector (both L2-normalized, so dot product = cosine sim).
 */
export function computePullMatrix(
  identityVectors: number[][],
  allVectors: number[][]
): number[][] {
  const nTools = identityVectors.length;
  const nVectors = allVectors.length;
  const matrix: number[][] = new Array(nTools);

  for (let i = 0; i < nTools; i++) {
    matrix[i] = new Array(nVectors);
    for (let j = 0; j < nVectors; j++) {
      matrix[i][j] = dotProduct(identityVectors[i], allVectors[j]);
    }
  }

  return matrix;
}

/**
 * L2 norm composite activation per tool.
 * sqrt(sum(pull_i^2)) — multiple moderate pulls compound meaningfully,
 * but with diminishing returns.
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
 * Returns the index of the fragment closest to the full query vector.
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
 * Find the natural activation cutoff using gap detection.
 *
 * Sorts scores descending, then looks for the largest relative drop
 * between consecutive tools in the range [minTools, maxTools].
 */
export function findActivationCutoff(
  scores: number[],
  minTools: number = MIN_TOOLS,
  maxTools: number = MAX_TOOLS
): { cutoff: number; position: number } {
  const sorted = [...scores].sort((a, b) => b - a);

  if (sorted.length <= minTools) {
    return { cutoff: 0, position: sorted.length };
  }

  let bestGap = 0;
  let bestPos = minTools;

  for (
    let i = minTools - 1;
    i < Math.min(maxTools, sorted.length - 1);
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
 *
 * Tools with historical track records of useful results get a boost;
 * tools that frequently error or return empty results are dampened.
 *
 * Uses fragment-conditioned bias when available: for each tool,
 * compute a weighted average of useful_rate for the fragment types
 * present in the current query.
 */
export function applyReliabilityBias(
  compositeScores: number[],
  toolNames: string[],
  reliabilityMap: Map<string, ToolReliability>,
  fragments: Fragment[],
  options?: { minActivations?: number; floor?: number }
): number[] {
  const minActivations = options?.minActivations ?? 10;
  const floor = options?.floor ?? getAdaptiveFloor();

  // Extract fragment types present in this query
  const fragmentTypes = new Set(fragments.map((f) => f.type));

  return compositeScores.map((score, i) => {
    const toolName = toolNames[i];
    const reliability = reliabilityMap.get(toolName);

    // Not enough data — return raw score
    if (!reliability || reliability.total_activations < minActivations) {
      return score;
    }

    // Fragment-conditioned bias: weighted average of useful_rate for query's fragment types
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
      // Use fragment-conditioned useful_rate
      bias = fragmentUsefulSum / fragmentWeightSum;
    } else {
      // Fall back to overall reliability_score
      bias = reliability.reliability_score;
    }

    // Apply floor so no tool is fully suppressed
    bias = Math.max(floor, bias);

    return score * bias;
  });
}

/**
 * Full gravity field computation with adaptive activation.
 *
 * 1. Build vector set: all fragments + full query vector
 * 2. Compute pull matrix (22 tools × (N fragments + 1 query))
 * 3. Compute composite activation (L2 norm over ALL pulls including query)
 * 4. Identify primary mass (centroid distance, fragments only)
 * 5. Apply adaptive gap-based activation threshold
 */
export function computeGravityField(
  identityVectors: number[][],
  decomposition: DecompositionResult
): GravityField {
  const fragmentVectors = decomposition.fragments
    .map((f) => f.embedding)
    .filter((e): e is number[] => e != null);
  const queryVector = decomposition.query_embedding;

  if (!queryVector || fragmentVectors.length === 0) {
    throw new Error("Decomposition must be embedded before computing gravity field");
  }

  // Build combined vector set: fragments + full query
  const allVectors = [...fragmentVectors, queryVector];

  // Step 1: Pull matrix
  const pullMatrix = computePullMatrix(identityVectors, allVectors);

  // Step 2: Composite activation (raw scores)
  const compositeScores = computeCompositeActivation(pullMatrix);

  // Step 2b: Apply learned reliability bias
  const reliabilityMap = getReliabilityMap();
  const biasedCompositeScores = applyReliabilityBias(
    compositeScores,
    TOOL_NAMES,
    reliabilityMap,
    decomposition.fragments
  );

  // Step 3: Primary mass (centroid distance, fragments only)
  const centroidPrimaryIndex = identifyPrimaryMass(queryVector, fragmentVectors);
  const primaryIdx = decomposition.primary_mass_index;
  const nFragments = decomposition.fragments.length;

  // Step 4: Build tool list (using biased scores for activation decisions)
  const allTools: ActivatedTool[] = TOOL_NAMES.map((name, i) => {
    const fragmentPulls = pullMatrix[i].slice(0, nFragments);
    const queryPull = pullMatrix[i][nFragments];
    const primaryPull = fragmentPulls[primaryIdx] ?? 0;

    return {
      name,
      composite_score: biasedCompositeScores[i], // Use biased scores
      primary_pull: primaryPull,
      query_pull: queryPull,
      is_meta: META_TOOLS.has(name),
      is_always_active: ALWAYS_ACTIVE_TOOLS.has(name),
    };
  });

  // Step 5: Adaptive gap detection for gravity-gated tools (using biased scores)
  const gatedTools = allTools.filter(
    (t) => !t.is_meta && !ALWAYS_ACTIVE_TOOLS.has(t.name)
  );
  const gatedScores = gatedTools.map((t) => t.composite_score);
  const { cutoff, position } = findActivationCutoff(gatedScores);

  const activated: ActivatedTool[] = [];

  // Always-active tools fire unconditionally
  for (const tool of allTools) {
    if (ALWAYS_ACTIVE_TOOLS.has(tool.name)) {
      activated.push(tool);
    }
  }

  // Gravity-gated tools fire above adaptive cutoff
  for (const tool of gatedTools) {
    if (tool.composite_score >= cutoff) {
      activated.push(tool);
    }
  }

  // Meta-tools: activate if in top 5 of ALL tool scores
  const sortedAllScores = allTools
    .map((t) => t.composite_score)
    .sort((a, b) => b - a);
  const top5Threshold = sortedAllScores[4] ?? 0;
  for (const tool of allTools) {
    if (tool.is_meta && tool.composite_score >= top5Threshold) {
      activated.push(tool);
    }
  }

  // Track activated names for deduplication
  const activatedNames = new Set(activated.map((t) => t.name));

  // Meta-tool keyword bypass: explicit meta-queries should always fire the right tool
  const queryLower = decomposition.extracted.search_query.toLowerCase();
  const metaKeywordMap: Record<string, string[]> = {
    get_writing_stats: ["how much have i written", "how many entries", "how many words", "writing stats", "total words", "word count total"],
    list_available_metrics: ["what metrics", "which metrics", "available metrics", "what can i track", "what can i measure"],
  };
  for (const [toolName, keywords] of Object.entries(metaKeywordMap)) {
    if (keywords.some((kw) => queryLower.includes(kw))) {
      if (!activatedNames.has(toolName)) {
        const tool = allTools.find((t) => t.name === toolName);
        if (tool) {
          activated.push(tool);
          activatedNames.add(toolName);
        }
      }
    }
  }

  // Semantic boost: when extracted params signal specific needs, ensure
  // the most relevant tools are included even if below adaptive cutoff
  const metricsExtracted = decomposition.extracted.metrics.length > 0;

  if (metricsExtracted) {
    // Metric queries should always include timeseries tools
    for (const name of ["query_time_series", "get_metric_summary"]) {
      if (!activatedNames.has(name)) {
        const tool = allTools.find((t) => t.name === name);
        if (tool) {
          activated.push(tool);
          activatedNames.add(name);
        }
      }
    }
  }

  // Sort activated by composite score descending
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
