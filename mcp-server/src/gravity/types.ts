export type FragmentType =
  | "concept"
  | "entity"
  | "temporal"
  | "emotional"
  | "relational"
  | "archetypal";

export interface Fragment {
  type: FragmentType;
  text: string;
  embedding?: number[];
}

export interface ExtractedParams {
  entities: string[];
  concepts: string[];
  date_ranges: Array<{ start?: string; end?: string }>;
  metrics: string[];
  search_query: string;
}

export interface DecompositionResult {
  fragments: Fragment[];
  primary_mass_index: number;
  reasoning: string;
  extracted: ExtractedParams;
  query_embedding?: number[];
}

export interface ActivatedTool {
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
  activated: ActivatedTool[];
  adaptive_cutoff: number;
}

export interface ToolResult {
  tool: string;
  result: string;
  composite_score: number;
  duration_ms: number;
  error?: string;
}

export interface OrchestratedResult {
  query: string;
  fragments: Fragment[];
  primary_mass: string;
  activated_tools: string[];
  results: ToolResult[];
  total_ms: number;
}

// ─── Gravity Ledger Types ────────────────────────────────────────────

export interface ToolOutcome {
  tool: string;
  composite_score: number;
  duration_ms: number;
  errored: boolean;
  empty: boolean;
  result_size: number;
}

export interface GravityOutcome {
  query: string;
  fragments: { type: FragmentType; text: string }[];
  activated_tools: string[];
  tool_outcomes: ToolOutcome[];
  timestamp: string;
}

export interface FragmentTypeStats {
  activations: number;
  useful_rate: number;
}

export interface ToolReliability {
  tool: string;
  total_activations: number;
  error_rate: number;
  empty_rate: number;
  useful_rate: number;
  avg_result_size: number;
  reliability_score: number;
  by_fragment_type: Partial<Record<FragmentType, FragmentTypeStats>>;
}
