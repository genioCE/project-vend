/**
 * Main gravity orchestration pipeline — Project Vend.
 *
 * End-to-end: decompose event → embed → gravity field → dispatch agents → assemble.
 */

import type { OrchestratedResult } from "./types.js";
import { decomposeEvent, embedDecomposition } from "./decompose.js";
import { getIdentityVectors } from "./identities.js";
import { computeGravityField } from "./field.js";
import { dispatchAgents } from "./dispatch.js";
import { assembleResults } from "./assemble.js";

/**
 * Orchestrate a vending system event through the gravity pipeline.
 *
 * @param event - Either a structured event object or free-text description
 * @param embedFn - Function to embed text arrays into vectors
 * @param signal - Optional abort signal
 */
export async function orchestrate(
  event: string | Record<string, unknown>,
  embedFn: (texts: string[]) => Promise<number[][]>,
  signal?: AbortSignal
): Promise<OrchestratedResult> {
  const start = performance.now();

  const eventDescription =
    typeof event === "string" ? event : JSON.stringify(event);

  console.error(`[gravity] event: "${eventDescription.slice(0, 120)}"`);

  // 1. Decompose event into fragments
  const decomposition = decomposeEvent(event);
  const frags = decomposition.fragments
    .map((f) => `[${f.type}] ${f.text}`)
    .join(", ");
  console.error(`[gravity] fragments: ${frags}`);

  // 2. Embed fragments + full event
  await embedDecomposition(decomposition, eventDescription, embedFn);

  // 3. Load agent identity vectors
  const identityVectors = await getIdentityVectors(embedFn, signal);

  // 4. Compute gravity field → activated agents
  const field = computeGravityField(identityVectors, decomposition);
  const agentNames = field.activated.map((a) => a.name).join(", ");
  console.error(
    `[gravity] activated ${field.activated.length} agents: ${agentNames}`
  );

  // 5. Dispatch activated agents in parallel
  const agentResults = await dispatchAgents(
    field.activated,
    decomposition.extracted,
    eventDescription,
    signal
  );

  // 6. Assemble results
  const result = assembleResults(
    eventDescription,
    decomposition,
    field,
    agentResults
  );
  result.total_ms = performance.now() - start;

  console.error(
    `[gravity] done in ${result.total_ms.toFixed(0)}ms — ${result.results.length} agent results`
  );

  return result;
}
