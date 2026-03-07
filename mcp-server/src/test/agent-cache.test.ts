import test from "node:test";
import assert from "node:assert/strict";
import { buildGraphRagQueryKey } from "../agent.js";

test("buildGraphRagQueryKey normalizes whitespace and case", () => {
  const key = buildGraphRagQueryKey("  HeLLo   World \n\n  again ");
  assert.equal(key, "hello world again");
});

test("buildGraphRagQueryKey bounds key length", () => {
  const long = "x".repeat(1000);
  const key = buildGraphRagQueryKey(long);
  assert.equal(key.length, 512);
});
