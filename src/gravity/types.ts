/**
 * Gravity engine types — adapted for Project Vend.
 *
 * Fragment taxonomy: Product, Signal, Temporal, Context, Customer, Machine
 * Activation targets: Vending system agents (not corpus tools)
 */

export type FragmentType =
  | "product"
  | "signal"
  | "temporal"
  | "context"
  | "customer"
  | "machine";

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  source: "llm" | "rule-based" | "fallback";
}

export interface Fragment {
  type: FragmentType;
  text: string;
  embedding?: number[];
}

export interface ExtractedParams {
  products: string[];
  signals: string[];
  machine_ids: string[];
  date_ranges: Array<{ start?: string; end?: string }>;
  context_tags: string[];
  search_query: string;
}

export interface DecompositionResult {
  fragments: Fragment[];
  primary_mass_index: number;
  reasoning: string;
  extracted: ExtractedParams;
  query_embedding?: number[];
  token_usage?: TokenUsage;
}

export interface ActivatedAgent {
  name: string;
  composite_score: number;
  primary_pull: number;
  query_pull: number;
  is_meta: boolean;
  is_always_active: boolean;
}

export interface GravityField {
  pull_matrix: number[][];
  composite_scores: number[];
  biased_composite_scores: number[];
  primary_mass_index: number;
  centroid_primary_index: number;
  activated: ActivatedAgent[];
  adaptive_cutoff: number;
}

export interface AgentResult {
  agent: string;
  result: string;
  composite_score: number;
  duration_ms: number;
  error?: string;
}

export interface OrchestratedResult {
  event_description: string;
  fragments: Fragment[];
  primary_mass: string;
  activated_agents: string[];
  results: AgentResult[];
  total_ms: number;
  token_usage?: TokenUsage;
}

// ─── Gravity Ledger Types ────────────────────────────────────────────

export interface AgentOutcome {
  agent: string;
  composite_score: number;
  duration_ms: number;
  errored: boolean;
  empty: boolean;
  result_size: number;
}

export interface GravityOutcome {
  event_description: string;
  fragments: { type: FragmentType; text: string }[];
  activated_agents: string[];
  agent_outcomes: AgentOutcome[];
  timestamp: string;
}

export interface FragmentTypeStats {
  activations: number;
  useful_rate: number;
}

export interface AgentReliability {
  agent: string;
  total_activations: number;
  error_rate: number;
  empty_rate: number;
  useful_rate: number;
  avg_result_size: number;
  reliability_score: number;
  by_fragment_type: Partial<Record<FragmentType, FragmentTypeStats>>;
}
