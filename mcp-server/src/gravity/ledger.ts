/**
 * Gravity Ledger — Learning loop for tool reliability.
 *
 * Records outcomes from every orchestrated query and aggregates them into
 * per-tool reliability profiles. These profiles are used to bias composite
 * scores so that historically reliable tools get a boost and flaky tools
 * get dampened.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type {
  FragmentType,
  GravityOutcome,
  ToolOutcome,
  ToolReliability,
  ToolResult,
  OrchestratedResult,
  FragmentTypeStats,
} from "./types.js";

// ─── Configuration ─────────────────────────────────────────────────────

const LEDGER_PATH = path.resolve(
  process.cwd(),
  "data",
  "gravity-ledger.jsonl"
);

// Exponential decay half-life: 50 queries → weight = 0.5 at queriesAgo=50
// decay constant = ln(2) / half_life ≈ 0.693 / 50 ≈ 0.0139
const DECAY_CONSTANT = 0.0139;

// Recompute reliability after this many new queries
const RECOMPUTE_INTERVAL = 10;

// ─── Module State ──────────────────────────────────────────────────────

let _outcomes: GravityOutcome[] = [];
let _reliabilityMap: Map<string, ToolReliability> = new Map();
let _queriesSinceRecompute = 0;
let _initialized = false;

// ─── isEmpty Detection ─────────────────────────────────────────────────

/**
 * Determine if a tool result is "empty" — meaning it provides no useful data.
 *
 * Empty conditions:
 * - Explicit error object {"error": ...}
 * - Empty array []
 * - Object where all array values are empty (only if there ARE array values)
 * - Result string < 50 bytes after removing whitespace (only if not parseable JSON with content)
 */
export function isEmpty(resultStr: string): boolean {
  const trimmed = resultStr.replace(/\s/g, "");

  // Very short results that can't be valid JSON with content
  if (trimmed.length < 3) {
    return true;
  }

  try {
    const parsed = JSON.parse(resultStr);

    // Explicit error
    if (parsed && typeof parsed === "object" && "error" in parsed) {
      return true;
    }

    // Empty array
    if (Array.isArray(parsed) && parsed.length === 0) {
      return true;
    }

    // Non-empty array is definitely useful
    if (Array.isArray(parsed) && parsed.length > 0) {
      return false;
    }

    // Object analysis
    if (parsed && typeof parsed === "object") {
      const values = Object.values(parsed);
      const arrayValues = values.filter(Array.isArray) as unknown[][];

      // If there are array values, check if any has content
      if (arrayValues.length > 0) {
        const hasNonEmptyArray = arrayValues.some((arr) => arr.length > 0);
        if (hasNonEmptyArray) {
          return false; // Has content
        }
        // All arrays are empty — check if there are any non-array values
        const nonArrayValues = values.filter((v) => !Array.isArray(v));
        if (nonArrayValues.length === 0) {
          return true; // Only empty arrays
        }
      }

      // For objects without arrays, fall through to size check
    }
  } catch {
    // Not valid JSON — fall through to size check
  }

  // Fallback: if trimmed size is very small, consider empty
  if (trimmed.length < 50) {
    return true;
  }

  return false;
}

// ─── Recording ─────────────────────────────────────────────────────────

/**
 * Build ToolOutcome from a ToolResult.
 */
function buildToolOutcome(toolResult: ToolResult): ToolOutcome {
  const errored = !!toolResult.error;
  const empty = errored ? true : isEmpty(toolResult.result);
  const resultSize = Buffer.byteLength(toolResult.result, "utf8");

  return {
    tool: toolResult.tool,
    composite_score: toolResult.composite_score,
    duration_ms: toolResult.duration_ms,
    errored,
    empty,
    result_size: resultSize,
  };
}

/**
 * Record an orchestration outcome to the ledger.
 *
 * Appends to JSONL file and updates in-memory state.
 * Triggers reliability recomputation every RECOMPUTE_INTERVAL queries.
 */
export async function recordOutcome(
  result: OrchestratedResult,
  toolResults: ToolResult[]
): Promise<void> {
  const outcome: GravityOutcome = {
    query: result.query,
    fragments: result.fragments.map((f) => ({ type: f.type, text: f.text })),
    activated_tools: result.activated_tools,
    tool_outcomes: toolResults.map(buildToolOutcome),
    timestamp: new Date().toISOString(),
  };

  // Append to in-memory store
  _outcomes.push(outcome);
  _queriesSinceRecompute++;

  // Append to JSONL file (fire-and-forget from caller's perspective)
  const line = JSON.stringify(outcome) + "\n";

  // Ensure data directory exists
  const dir = path.dirname(LEDGER_PATH);
  await fs.promises.mkdir(dir, { recursive: true });

  // Append to file
  await fs.promises.appendFile(LEDGER_PATH, line, "utf8");

  // Recompute reliability periodically
  if (_queriesSinceRecompute >= RECOMPUTE_INTERVAL) {
    computeReliability();
    _queriesSinceRecompute = 0;
  }
}

// ─── Loading ───────────────────────────────────────────────────────────

/**
 * Load existing ledger entries from disk on startup.
 */
export function loadLedger(): void {
  if (_initialized) return;

  try {
    if (!fs.existsSync(LEDGER_PATH)) {
      console.error("[gravity] ledger: no existing ledger file, starting fresh");
      _initialized = true;
      return;
    }

    const content = fs.readFileSync(LEDGER_PATH, "utf8");
    const lines = content.split("\n").filter((line) => line.trim());

    for (const line of lines) {
      try {
        const outcome = JSON.parse(line) as GravityOutcome;
        _outcomes.push(outcome);
      } catch {
        // Skip malformed lines
        console.error("[gravity] ledger: skipping malformed line");
      }
    }

    console.error(
      `[gravity] ledger: loaded ${_outcomes.length} historical outcomes`
    );

    // Initial reliability computation
    computeReliability();
    _initialized = true;
  } catch (err) {
    console.error(`[gravity] ledger: failed to load ledger: ${err}`);
    _initialized = true;
  }
}

// ─── Aggregation ───────────────────────────────────────────────────────

/**
 * Compute exponential decay weight for an outcome.
 * queriesAgo = 0 means most recent, weight = 1.0
 */
export function decayWeight(queriesAgo: number): number {
  return Math.exp(-DECAY_CONSTANT * queriesAgo);
}

interface WeightedToolStats {
  totalWeight: number;
  errorWeight: number;
  emptyWeight: number;
  usefulWeight: number;
  sizeWeightedSum: number;
  byFragmentType: Map<FragmentType, { totalWeight: number; usefulWeight: number }>;
}

/**
 * Compute reliability profiles for all tools based on historical outcomes.
 */
export function computeReliability(): void {
  const totalOutcomes = _outcomes.length;
  if (totalOutcomes === 0) {
    _reliabilityMap.clear();
    return;
  }

  const toolStats = new Map<string, WeightedToolStats>();

  // Process outcomes in reverse chronological order (newest first)
  for (let i = totalOutcomes - 1; i >= 0; i--) {
    const outcome = _outcomes[i];
    const queriesAgo = totalOutcomes - 1 - i;
    const weight = decayWeight(queriesAgo);

    // Extract fragment types for this query
    const fragmentTypes = new Set(outcome.fragments.map((f) => f.type));

    for (const toolOutcome of outcome.tool_outcomes) {
      let stats = toolStats.get(toolOutcome.tool);
      if (!stats) {
        stats = {
          totalWeight: 0,
          errorWeight: 0,
          emptyWeight: 0,
          usefulWeight: 0,
          sizeWeightedSum: 0,
          byFragmentType: new Map(),
        };
        toolStats.set(toolOutcome.tool, stats);
      }

      stats.totalWeight += weight;

      if (toolOutcome.errored) {
        stats.errorWeight += weight;
      } else if (toolOutcome.empty) {
        stats.emptyWeight += weight;
      } else {
        stats.usefulWeight += weight;
        stats.sizeWeightedSum += toolOutcome.result_size * weight;
      }

      // Track per-fragment-type stats
      for (const fragType of fragmentTypes) {
        let fragStats = stats.byFragmentType.get(fragType);
        if (!fragStats) {
          fragStats = { totalWeight: 0, usefulWeight: 0 };
          stats.byFragmentType.set(fragType, fragStats);
        }
        fragStats.totalWeight += weight;
        if (!toolOutcome.errored && !toolOutcome.empty) {
          fragStats.usefulWeight += weight;
        }
      }
    }
  }

  // Build reliability map
  const newMap = new Map<string, ToolReliability>();

  for (const [tool, stats] of toolStats) {
    const errorRate = stats.totalWeight > 0 ? stats.errorWeight / stats.totalWeight : 0;
    const emptyRate = stats.totalWeight > 0 ? stats.emptyWeight / stats.totalWeight : 0;
    const usefulRate = stats.totalWeight > 0 ? stats.usefulWeight / stats.totalWeight : 0;
    const avgResultSize = stats.usefulWeight > 0 ? stats.sizeWeightedSum / stats.usefulWeight : 0;

    const byFragmentType: Partial<Record<FragmentType, FragmentTypeStats>> = {};
    for (const [fragType, fragStats] of stats.byFragmentType) {
      byFragmentType[fragType] = {
        activations: Math.round(fragStats.totalWeight), // Approximate
        useful_rate: fragStats.totalWeight > 0 ? fragStats.usefulWeight / fragStats.totalWeight : 0,
      };
    }

    newMap.set(tool, {
      tool,
      total_activations: Math.round(stats.totalWeight), // Approximate due to decay
      error_rate: errorRate,
      empty_rate: emptyRate,
      useful_rate: usefulRate,
      avg_result_size: avgResultSize,
      reliability_score: usefulRate,
      by_fragment_type: byFragmentType,
    });
  }

  _reliabilityMap = newMap;
  console.error(
    `[gravity] ledger: recomputed reliability for ${newMap.size} tools from ${totalOutcomes} outcomes`
  );
}

// ─── Accessors ─────────────────────────────────────────────────────────

/**
 * Get the full reliability map.
 * Loads ledger on first access if not already loaded.
 */
export function getReliabilityMap(): Map<string, ToolReliability> {
  if (!_initialized) {
    loadLedger();
  }
  return _reliabilityMap;
}

/**
 * Get reliability for a specific tool.
 */
export function getReliabilityForTool(name: string): ToolReliability | undefined {
  if (!_initialized) {
    loadLedger();
  }
  return _reliabilityMap.get(name);
}

/**
 * Get the raw outcome count (for diagnostics).
 */
export function getOutcomeCount(): number {
  if (!_initialized) {
    loadLedger();
  }
  return _outcomes.length;
}
