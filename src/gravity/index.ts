/**
 * Gravity engine — public API.
 */

export { orchestrate } from "./orchestrate.js";
export { decomposeEvent, embedDecomposition } from "./decompose.js";
export { computeGravityField } from "./field.js";
export { dispatchAgents } from "./dispatch.js";
export { assembleResults } from "./assemble.js";
export { loadLedger, getReliabilityMap, getOutcomeCount } from "./ledger.js";
export { AGENTS, AGENT_NAMES, getIdentityVectors } from "./identities.js";

export type {
  FragmentType,
  Fragment,
  DecompositionResult,
  ExtractedParams,
  ActivatedAgent,
  GravityField,
  AgentResult,
  OrchestratedResult,
  AgentReliability,
  GravityOutcome,
} from "./types.js";
