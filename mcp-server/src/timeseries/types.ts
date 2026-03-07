export const DIMENSIONS = [
  "valence",
  "activation",
  "agency",
  "certainty",
  "relational_openness",
  "self_trust",
  "time_orientation",
  "integration",
] as const;

export type Dimension = (typeof DIMENSIONS)[number];

export type DataQuality = "clean" | "suspect_zero" | "mock";

export interface DataPoint {
  date: string;
  value: number;
}

export interface EntryDataPoint extends DataPoint {
  entry_id: string;
}

export interface AnomalyPoint {
  date: string;
  entry_id?: string;
  value: number;
  baseline_mean: number;
  baseline_std: number;
  z_score: number;
}

export interface CorrelationResult {
  pearson_r: number;
  p_value: number;
  n: number;
  data_points: Array<{ date: string; value_a: number; value_b: number }>;
  interpretation: string;
}

export interface MetricSummary {
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  current_trend: "rising" | "falling" | "stable";
  entries_count: number;
}

export interface MetricInfo {
  metric_name: string;
  metric_type: "dimension" | "entry_stat" | "archetype" | "concept";
  description: string;
}

export interface FilterCondition {
  metric: string;
  operator: ">" | ">=" | "<" | "<=" | "=" | "!=";
  value: number;
}
