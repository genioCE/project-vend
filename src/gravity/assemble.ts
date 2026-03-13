/**
 * Result assembly — Project Vend.
 *
 * Orders agent results by composite gravity score for synthesis.
 */

import type {
  DecompositionResult,
  GravityField,
  OrchestratedResult,
  AgentResult,
} from "./types.js";

export function assembleResults(
  eventDescription: string,
  decomposition: DecompositionResult,
  field: GravityField,
  agentResults: AgentResult[]
): OrchestratedResult {
  const sorted = [...agentResults].sort(
    (a, b) => b.composite_score - a.composite_score
  );

  const primaryFragment =
    decomposition.fragments[decomposition.primary_mass_index];

  return {
    event_description: eventDescription,
    fragments: decomposition.fragments.map((f) => ({
      type: f.type,
      text: f.text,
    })),
    primary_mass: primaryFragment?.text ?? eventDescription,
    activated_agents: field.activated.map((a) => a.name),
    results: sorted,
    total_ms: 0,
    token_usage: decomposition.token_usage,
  };
}
