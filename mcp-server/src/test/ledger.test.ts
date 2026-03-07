import test from "node:test";
import assert from "node:assert/strict";
import {
  isEmpty,
  decayWeight,
} from "../gravity/ledger.js";
import { applyReliabilityBias } from "../gravity/field.js";
import type {
  Fragment,
  FragmentType,
  ToolReliability,
} from "../gravity/types.js";

// ── isEmpty detection ───────────────────────────────────────────────────

test("isEmpty returns true for empty string", () => {
  assert.ok(isEmpty(""));
});

test("isEmpty returns true for empty array", () => {
  assert.ok(isEmpty("[]"));
});

test("isEmpty returns true for empty object", () => {
  assert.ok(isEmpty("{}"));
});

test("isEmpty returns true for error object", () => {
  assert.ok(isEmpty('{"error": "No data found"}'));
});

test("isEmpty returns true for object with all empty arrays", () => {
  assert.ok(isEmpty('{"flows": [], "concepts": [], "name": "test"}'));
});

test("isEmpty returns true for short result (<50 bytes)", () => {
  assert.ok(isEmpty('{"count": 0}'));
});

test("isEmpty returns false for meaningful result", () => {
  const result = JSON.stringify({
    entries: [
      { id: "1", date: "2025-01-01", text: "This is a meaningful entry" },
    ],
    count: 1,
  });
  assert.ok(!isEmpty(result));
});

test("isEmpty returns false for array with items", () => {
  const result = JSON.stringify([
    { id: "1", text: "Entry one" },
    { id: "2", text: "Entry two" },
  ]);
  assert.ok(!isEmpty(result));
});

test("isEmpty returns false for object with non-empty arrays", () => {
  const result = JSON.stringify({
    flows: [{ from: "a", to: "b" }],
    concepts: [],
  });
  assert.ok(!isEmpty(result));
});

// ── Exponential decay weighting ─────────────────────────────────────────

test("decayWeight returns 1.0 for queriesAgo=0", () => {
  const weight = decayWeight(0);
  assert.ok(Math.abs(weight - 1.0) < 0.001);
});

test("decayWeight returns ~0.5 for queriesAgo=50 (half-life)", () => {
  const weight = decayWeight(50);
  // half-life = 50 means weight at 50 should be ~0.5
  assert.ok(weight > 0.45 && weight < 0.55, `Expected ~0.5, got ${weight}`);
});

test("decayWeight returns ~0.25 for queriesAgo=100 (two half-lives)", () => {
  const weight = decayWeight(100);
  assert.ok(weight > 0.2 && weight < 0.3, `Expected ~0.25, got ${weight}`);
});

test("decayWeight decreases monotonically", () => {
  const w0 = decayWeight(0);
  const w10 = decayWeight(10);
  const w50 = decayWeight(50);
  const w100 = decayWeight(100);
  assert.ok(w0 > w10);
  assert.ok(w10 > w50);
  assert.ok(w50 > w100);
});

// ── Reliability score computation (unit test with mock data) ───────────

function makeReliability(
  tool: string,
  activations: number,
  usefulRate: number,
  byFragType?: Partial<Record<FragmentType, { activations: number; useful_rate: number }>>
): ToolReliability {
  return {
    tool,
    total_activations: activations,
    error_rate: 0.1,
    empty_rate: 1 - usefulRate - 0.1,
    useful_rate: usefulRate,
    avg_result_size: 5000,
    reliability_score: usefulRate,
    by_fragment_type: byFragType ?? {},
  };
}

function makeFragment(type: FragmentType, text: string): Fragment {
  return { type, text };
}

// ── applyReliabilityBias ────────────────────────────────────────────────

test("applyReliabilityBias returns raw scores when no reliability data", () => {
  const scores = [0.8, 0.6, 0.4];
  const toolNames = ["tool_a", "tool_b", "tool_c"];
  const reliabilityMap = new Map<string, ToolReliability>();
  const fragments = [makeFragment("concept", "test")];

  const biased = applyReliabilityBias(scores, toolNames, reliabilityMap, fragments);

  assert.deepEqual(biased, scores);
});

test("applyReliabilityBias returns raw scores when activations < minActivations", () => {
  const scores = [0.8, 0.6, 0.4];
  const toolNames = ["tool_a", "tool_b", "tool_c"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 5, 0.9)], // Only 5 activations
    ["tool_b", makeReliability("tool_b", 3, 0.5)],
    ["tool_c", makeReliability("tool_c", 9, 0.7)],
  ]);
  const fragments = [makeFragment("concept", "test")];

  const biased = applyReliabilityBias(scores, toolNames, reliabilityMap, fragments);

  // All tools have < 10 activations, so raw scores returned
  assert.deepEqual(biased, scores);
});

test("applyReliabilityBias applies reliability multiplier", () => {
  const scores = [0.8, 0.6, 0.4];
  const toolNames = ["tool_a", "tool_b", "tool_c"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 100, 0.9)],
    ["tool_b", makeReliability("tool_b", 100, 0.5)],
    ["tool_c", makeReliability("tool_c", 100, 0.3)],
  ]);
  const fragments = [makeFragment("concept", "test")];

  const biased = applyReliabilityBias(scores, toolNames, reliabilityMap, fragments);

  // tool_a: 0.8 * 0.9 = 0.72
  // tool_b: 0.6 * 0.5 = 0.30
  // tool_c: 0.4 * max(0.3, 0.3) = 0.12
  assert.ok(Math.abs(biased[0] - 0.72) < 0.001, `Expected 0.72, got ${biased[0]}`);
  assert.ok(Math.abs(biased[1] - 0.30) < 0.001, `Expected 0.30, got ${biased[1]}`);
  assert.ok(Math.abs(biased[2] - 0.12) < 0.001, `Expected 0.12, got ${biased[2]}`);
});

test("applyReliabilityBias applies floor to prevent full suppression", () => {
  const scores = [0.8];
  const toolNames = ["tool_a"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 100, 0.1)], // Very low reliability
  ]);
  const fragments = [makeFragment("concept", "test")];

  const biased = applyReliabilityBias(scores, toolNames, reliabilityMap, fragments);

  // Should use floor of 0.3, not 0.1
  // 0.8 * 0.3 = 0.24
  assert.ok(Math.abs(biased[0] - 0.24) < 0.001, `Expected 0.24, got ${biased[0]}`);
});

test("applyReliabilityBias uses custom floor option", () => {
  const scores = [0.8];
  const toolNames = ["tool_a"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 100, 0.1)],
  ]);
  const fragments = [makeFragment("concept", "test")];

  const biased = applyReliabilityBias(
    scores,
    toolNames,
    reliabilityMap,
    fragments,
    { floor: 0.5 }
  );

  // Should use floor of 0.5
  // 0.8 * 0.5 = 0.4
  assert.ok(Math.abs(biased[0] - 0.4) < 0.001, `Expected 0.4, got ${biased[0]}`);
});

test("applyReliabilityBias uses custom minActivations option", () => {
  const scores = [0.8];
  const toolNames = ["tool_a"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 5, 0.9)],
  ]);
  const fragments = [makeFragment("concept", "test")];

  // With minActivations=3, should apply bias
  const biased = applyReliabilityBias(
    scores,
    toolNames,
    reliabilityMap,
    fragments,
    { minActivations: 3 }
  );

  assert.ok(Math.abs(biased[0] - 0.72) < 0.001, `Expected 0.72, got ${biased[0]}`);
});

// ── Fragment-type conditioning ──────────────────────────────────────────

test("applyReliabilityBias uses fragment-conditioned rates when available", () => {
  const scores = [0.8];
  const toolNames = ["tool_a"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 100, 0.5, {
      concept: { activations: 50, useful_rate: 0.9 },
      entity: { activations: 50, useful_rate: 0.1 },
    })],
  ]);
  const fragments = [makeFragment("concept", "test")];

  const biased = applyReliabilityBias(scores, toolNames, reliabilityMap, fragments);

  // Should use fragment-conditioned rate of 0.9 for "concept"
  // 0.8 * 0.9 = 0.72
  assert.ok(Math.abs(biased[0] - 0.72) < 0.001, `Expected 0.72, got ${biased[0]}`);
});

test("applyReliabilityBias averages rates across multiple fragment types", () => {
  const scores = [0.8];
  const toolNames = ["tool_a"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 100, 0.5, {
      concept: { activations: 50, useful_rate: 0.8 },
      entity: { activations: 50, useful_rate: 0.6 },
    })],
  ]);
  const fragments = [
    makeFragment("concept", "idea"),
    makeFragment("entity", "Kyle"),
  ];

  const biased = applyReliabilityBias(scores, toolNames, reliabilityMap, fragments);

  // Weighted average: (50*0.8 + 50*0.6) / (50+50) = 70/100 = 0.7
  // 0.8 * 0.7 = 0.56
  assert.ok(Math.abs(biased[0] - 0.56) < 0.001, `Expected 0.56, got ${biased[0]}`);
});

test("applyReliabilityBias falls back to overall rate when fragment data insufficient", () => {
  const scores = [0.8];
  const toolNames = ["tool_a"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 100, 0.5, {
      concept: { activations: 2, useful_rate: 0.9 }, // Only 2 activations, below threshold of 3
    })],
  ]);
  const fragments = [makeFragment("concept", "test")];

  const biased = applyReliabilityBias(scores, toolNames, reliabilityMap, fragments);

  // Should fall back to overall reliability_score of 0.5
  // 0.8 * 0.5 = 0.4
  assert.ok(Math.abs(biased[0] - 0.4) < 0.001, `Expected 0.4, got ${biased[0]}`);
});

test("applyReliabilityBias handles mixed fragment availability", () => {
  const scores = [0.8];
  const toolNames = ["tool_a"];
  const reliabilityMap = new Map<string, ToolReliability>([
    ["tool_a", makeReliability("tool_a", 100, 0.5, {
      concept: { activations: 50, useful_rate: 0.8 },
      // No data for "entity" type
    })],
  ]);
  const fragments = [
    makeFragment("concept", "idea"),
    makeFragment("entity", "Kyle"), // No fragment-type data for this
  ];

  const biased = applyReliabilityBias(scores, toolNames, reliabilityMap, fragments);

  // Only "concept" has data with activations >= 3
  // Uses only concept rate: 0.8
  // 0.8 * 0.8 = 0.64
  assert.ok(Math.abs(biased[0] - 0.64) < 0.001, `Expected 0.64, got ${biased[0]}`);
});

// ── Metric normalization (regression test) ──────────────────────────────

import { resolveMetric } from "../timeseries/metrics.js";

test("resolveMetric normalizes hyphens to underscores", () => {
  const r = resolveMetric("self-trust");
  assert.ok(r);
  assert.equal(r!.kind, "column");
  assert.equal(r!.column, "self_trust");
});

test("resolveMetric normalizes case and trims whitespace", () => {
  const r = resolveMetric("  SELF_TRUST  ");
  assert.ok(r);
  assert.equal(r!.kind, "column");
  assert.equal(r!.column, "self_trust");
});

test("resolveMetric normalizes mixed case with hyphens", () => {
  const r = resolveMetric("Time-Orientation");
  assert.ok(r);
  assert.equal(r!.kind, "column");
  assert.equal(r!.column, "time_orientation");
});

test("resolveMetric normalizes archetype prefix but preserves filterValue case", () => {
  const r = resolveMetric("ARCHETYPE:Creator");
  assert.ok(r);
  assert.equal(r!.kind, "archetype");
  assert.equal(r!.filterValue, "Creator"); // Case preserved for DB query
});
