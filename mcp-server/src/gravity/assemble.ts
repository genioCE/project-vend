/**
 * Result assembly.
 *
 * Orders and formats tool results for Claude's synthesis.
 * Results are ordered by composite score (highest gravity = most relevant).
 */

import type {
  DecompositionResult,
  GravityField,
  OrchestratedResult,
  ToolResult,
} from "./types.js";

export function assembleResults(
  query: string,
  decomposition: DecompositionResult,
  field: GravityField,
  toolResults: ToolResult[]
): OrchestratedResult {
  // Sort by composite score descending (most relevant first)
  const sorted = [...toolResults].sort(
    (a, b) => b.composite_score - a.composite_score
  );

  const primaryFragment =
    decomposition.fragments[decomposition.primary_mass_index];

  return {
    query,
    fragments: decomposition.fragments.map((f) => ({
      type: f.type,
      text: f.text,
    })),
    primary_mass: primaryFragment?.text ?? query,
    activated_tools: field.activated.map((t) => t.name),
    results: sorted,
    total_ms: 0, // set by caller
  };
}
