/**
 * Main gravity orchestration pipeline.
 *
 * End-to-end: decompose → embed → gravity field → dispatch → assemble.
 */

import type { OrchestratedResult } from "./types.js";
import { decompose, embedDecomposition } from "./decompose.js";
import { getIdentityVectors } from "./identities.js";
import { computeGravityField } from "./field.js";
import { dispatchTools } from "./dispatch.js";
import { assembleResults } from "./assemble.js";
import { recordOutcome } from "./ledger.js";

export async function orchestrate(
  query: string,
  signal?: AbortSignal
): Promise<OrchestratedResult> {
  const start = performance.now();

  console.error(`[gravity] orchestrated_query: "${query}"`);

  // 1. Decompose query into fragments + extract params
  const decomposition = await decompose(query, signal);
  const frags = decomposition.fragments.map((f) => `[${f.type}] ${f.text}`).join(", ");
  const tokenInfo = decomposition.token_usage
    ? ` (${decomposition.token_usage.source}: ${decomposition.token_usage.total_tokens} tokens)`
    : "";
  console.error(`[gravity] fragments: ${frags}${tokenInfo}`);

  // 2. Embed fragments + full query via /embed endpoint
  await embedDecomposition(decomposition, query, signal);

  // 3. Load identity vectors (cached in memory after first call)
  const identityVectors = await getIdentityVectors(signal);

  // 4. Compute gravity field → activated tools
  const field = computeGravityField(identityVectors, decomposition);
  const toolNames = field.activated.map((t) => t.name).join(", ");
  console.error(`[gravity] activated ${field.activated.length} tools: ${toolNames}`);

  // 5. Dispatch activated tools in parallel
  const toolResults = await dispatchTools(
    field.activated,
    decomposition.extracted,
    query,
    signal
  );

  // 6. Assemble results
  const result = assembleResults(query, decomposition, field, toolResults);
  result.total_ms = performance.now() - start;

  // 7. Record outcome to ledger (fire-and-forget)
  recordOutcome(result, toolResults).catch((err) =>
    console.error(`[gravity] ledger: failed to record outcome: ${err}`)
  );

  const errored = result.results.filter((r) => r.error);
  const errorSummary = errored.length
    ? `, ${errored.length} errors: ${errored.map((r) => `${r.tool}(${r.error!.slice(0, 60)})`).join(", ")}`
    : "";
  console.error(
    `[gravity] done in ${result.total_ms.toFixed(0)}ms — ${result.results.length} results${errorSummary}`
  );

  return result;
}
