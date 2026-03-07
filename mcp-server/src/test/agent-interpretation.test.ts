import test from "node:test";
import assert from "node:assert/strict";
import { postProcessInterpretiveResponse } from "../agent.js";

test("postProcessInterpretiveResponse removes duplicate paragraphs", () => {
  const input = [
    "Kelsey appears as a central stabilizing relationship.",
    "",
    "Kelsey appears as a central stabilizing relationship.",
    "",
    "Amy appears mostly in transition and rupture contexts.",
  ].join("\n");

  const output = postProcessInterpretiveResponse(input, false);
  const occurrences = output.match(
    /Kelsey appears as a central stabilizing relationship\./g
  );
  assert.equal(occurrences?.length ?? 0, 1);
  assert.match(output, /Amy appears mostly in transition and rupture contexts\./);
});

test("postProcessInterpretiveResponse removes duplicate bullets", () => {
  const input = [
    "Evidence:",
    "- [2025-05-01] Calm is often tied to time in nature.",
    "- [2025-05-01] Calm is often tied to time in nature.",
    "- [2025-06-10] Stress clusters around work uncertainty.",
  ].join("\n");

  const output = postProcessInterpretiveResponse(input, false);
  const repeated = output.match(/Calm is often tied to time in nature\./g);
  assert.equal(repeated?.length ?? 0, 1);
  assert.match(output, /Stress clusters around work uncertainty\./);
});

test("postProcessInterpretiveResponse adds temporal note when dated contrast is missing", () => {
  const input = "Core read: The entries suggest meaningful change but lack explicit date anchors.";
  const output = postProcessInterpretiveResponse(input, true);
  assert.match(output, /Shift over time: I need more distinct dated evidence/i);
});

test("postProcessInterpretiveResponse does not force temporal note when years are present", () => {
  const input = "In 2024 this was fear-heavy, while in 2025 it shifted toward agency.";
  const output = postProcessInterpretiveResponse(input, true);
  assert.doesNotMatch(output, /Shift over time: I need more distinct dated evidence/i);
});
