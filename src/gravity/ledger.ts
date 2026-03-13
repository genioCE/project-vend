/**
 * Gravity Ledger — Learning loop for agent reliability.
 *
 * Records outcomes from every orchestrated event and aggregates them into
 * per-agent reliability profiles. Profiles bias composite scores so that
 * historically reliable agents get boosted and flaky agents get dampened.
 *
 * Adapted from corpus-intelligence: tool → agent naming throughout.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type {
  FragmentType,
  GravityOutcome,
  AgentOutcome,
  AgentReliability,
  AgentResult,
  OrchestratedResult,
  FragmentTypeStats,
} from "./types.js";

// ─── Configuration ─────────────────────────────────────────────────────

const LEDGER_PATH = path.resolve(
  process.cwd(),
  "data",
  "gravity-ledger.jsonl"
);

const DECAY_CONSTANT = 0.0139; // half-life: 50 events
const RECOMPUTE_INTERVAL = 10;

// ─── Module State ──────────────────────────────────────────────────────

let _outcomes: GravityOutcome[] = [];
let _reliabilityMap: Map<string, AgentReliability> = new Map();
let _eventsSinceRecompute = 0;
let _initialized = false;

// ─── isEmpty Detection ─────────────────────────────────────────────────

export function isEmpty(resultStr: string): boolean {
  const trimmed = resultStr.replace(/\s/g, "");

  if (trimmed.length < 3) return true;

  try {
    const parsed = JSON.parse(resultStr);

    if (parsed && typeof parsed === "object" && "error" in parsed) return true;
    if (Array.isArray(parsed) && parsed.length === 0) return true;
    if (Array.isArray(parsed) && parsed.length > 0) return false;

    if (parsed && typeof parsed === "object") {
      const values = Object.values(parsed);
      const arrayValues = values.filter(Array.isArray) as unknown[][];
      if (arrayValues.length > 0) {
        const hasNonEmptyArray = arrayValues.some((arr) => arr.length > 0);
        if (hasNonEmptyArray) return false;
        const nonArrayValues = values.filter((v) => !Array.isArray(v));
        if (nonArrayValues.length === 0) return true;
      }
    }
  } catch {
    // Not valid JSON
  }

  if (trimmed.length < 50) return true;
  return false;
}

// ─── Recording ─────────────────────────────────────────────────────────

function buildAgentOutcome(agentResult: AgentResult): AgentOutcome {
  const errored = !!agentResult.error;
  const empty = errored ? true : isEmpty(agentResult.result);
  const resultSize = Buffer.byteLength(agentResult.result, "utf8");

  return {
    agent: agentResult.agent,
    composite_score: agentResult.composite_score,
    duration_ms: agentResult.duration_ms,
    errored,
    empty,
    result_size: resultSize,
  };
}

export async function recordOutcome(
  result: OrchestratedResult,
  agentResults: AgentResult[]
): Promise<void> {
  const outcome: GravityOutcome = {
    event_description: result.event_description,
    fragments: result.fragments.map((f) => ({ type: f.type, text: f.text })),
    activated_agents: result.activated_agents,
    agent_outcomes: agentResults.map(buildAgentOutcome),
    timestamp: new Date().toISOString(),
  };

  _outcomes.push(outcome);
  _eventsSinceRecompute++;

  const line = JSON.stringify(outcome) + "\n";
  const dir = path.dirname(LEDGER_PATH);
  await fs.promises.mkdir(dir, { recursive: true });
  await fs.promises.appendFile(LEDGER_PATH, line, "utf8");

  if (_eventsSinceRecompute >= RECOMPUTE_INTERVAL) {
    computeReliability();
    _eventsSinceRecompute = 0;
  }
}

// ─── Loading ───────────────────────────────────────────────────────────

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
        console.error("[gravity] ledger: skipping malformed line");
      }
    }

    console.error(
      `[gravity] ledger: loaded ${_outcomes.length} historical outcomes`
    );
    computeReliability();
    _initialized = true;
  } catch (err) {
    console.error(`[gravity] ledger: failed to load: ${err}`);
    _initialized = true;
  }
}

// ─── Aggregation ───────────────────────────────────────────────────────

export function decayWeight(eventsAgo: number): number {
  return Math.exp(-DECAY_CONSTANT * eventsAgo);
}

interface WeightedAgentStats {
  totalWeight: number;
  errorWeight: number;
  emptyWeight: number;
  usefulWeight: number;
  sizeWeightedSum: number;
  byFragmentType: Map<FragmentType, { totalWeight: number; usefulWeight: number }>;
}

export function computeReliability(): void {
  const totalOutcomes = _outcomes.length;
  if (totalOutcomes === 0) {
    _reliabilityMap.clear();
    return;
  }

  const agentStats = new Map<string, WeightedAgentStats>();

  for (let i = totalOutcomes - 1; i >= 0; i--) {
    const outcome = _outcomes[i];
    const eventsAgo = totalOutcomes - 1 - i;
    const weight = decayWeight(eventsAgo);

    const fragmentTypes = new Set(outcome.fragments.map((f) => f.type));

    for (const agentOutcome of outcome.agent_outcomes) {
      let stats = agentStats.get(agentOutcome.agent);
      if (!stats) {
        stats = {
          totalWeight: 0,
          errorWeight: 0,
          emptyWeight: 0,
          usefulWeight: 0,
          sizeWeightedSum: 0,
          byFragmentType: new Map(),
        };
        agentStats.set(agentOutcome.agent, stats);
      }

      stats.totalWeight += weight;

      if (agentOutcome.errored) {
        stats.errorWeight += weight;
      } else if (agentOutcome.empty) {
        stats.emptyWeight += weight;
      } else {
        stats.usefulWeight += weight;
        stats.sizeWeightedSum += agentOutcome.result_size * weight;
      }

      for (const fragType of fragmentTypes) {
        let fragStats = stats.byFragmentType.get(fragType);
        if (!fragStats) {
          fragStats = { totalWeight: 0, usefulWeight: 0 };
          stats.byFragmentType.set(fragType, fragStats);
        }
        fragStats.totalWeight += weight;
        if (!agentOutcome.errored && !agentOutcome.empty) {
          fragStats.usefulWeight += weight;
        }
      }
    }
  }

  const newMap = new Map<string, AgentReliability>();

  for (const [agent, stats] of agentStats) {
    const errorRate = stats.totalWeight > 0 ? stats.errorWeight / stats.totalWeight : 0;
    const emptyRate = stats.totalWeight > 0 ? stats.emptyWeight / stats.totalWeight : 0;
    const usefulRate = stats.totalWeight > 0 ? stats.usefulWeight / stats.totalWeight : 0;
    const avgResultSize = stats.usefulWeight > 0 ? stats.sizeWeightedSum / stats.usefulWeight : 0;

    const byFragmentType: Partial<Record<FragmentType, FragmentTypeStats>> = {};
    for (const [fragType, fragStats] of stats.byFragmentType) {
      byFragmentType[fragType] = {
        activations: Math.round(fragStats.totalWeight),
        useful_rate: fragStats.totalWeight > 0 ? fragStats.usefulWeight / fragStats.totalWeight : 0,
      };
    }

    newMap.set(agent, {
      agent,
      total_activations: Math.round(stats.totalWeight),
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
    `[gravity] ledger: recomputed reliability for ${newMap.size} agents from ${totalOutcomes} outcomes`
  );
}

// ─── Accessors ─────────────────────────────────────────────────────────

export function getReliabilityMap(): Map<string, AgentReliability> {
  if (!_initialized) loadLedger();
  return _reliabilityMap;
}

export function getReliabilityForAgent(name: string): AgentReliability | undefined {
  if (!_initialized) loadLedger();
  return _reliabilityMap.get(name);
}

export function getOutcomeCount(): number {
  if (!_initialized) loadLedger();
  return _outcomes.length;
}
