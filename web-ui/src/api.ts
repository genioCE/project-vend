export interface SSEEvent {
  type: "text_delta" | "tool_call" | "tool_result" | "done" | "error";
  request_id?: string;
  content?: string;
  tool?: string;
  input?: Record<string, unknown>;
  preview?: string;
  center?: string;
  message?: string;
}

export interface GraphFeedbackPayload {
  signal: "up" | "down";
  query?: string;
  note?: string;
  concepts?: string[];
  people?: string[];
  places?: string[];
  sources?: string[];
}

export interface FeedbackPromptItem {
  id: string;
  tag: string;
  title: string;
  count: number;
  latest_at: string;
  sample_notes: string[];
  prompt: string;
}

export interface FeedbackReviewData {
  summary: {
    total_events: number;
    helpful_count: number;
    off_target_count: number;
    off_target_with_notes: number;
  };
  prompt_backlog: FeedbackPromptItem[];
  recent_off_target: Array<{
    id: number;
    timestamp: string;
    query: string;
    note: string;
    concepts: string[];
    people: string[];
    places: string[];
    sources: string[];
  }>;
}

export interface AnalysisBundleRequest {
  entry_id: string;
  text: string;
  entry_date?: string;
  source_file?: string;
  query?: string;
  chunk_ids?: string[];
  max_summary_sentences?: number;
}

export interface AnalysisBundleResponse {
  entry_id: string;
  summary: {
    analysis_id: string;
    summary: string;
    highlights: string[];
    key_terms: string[];
    coverage_ratio: number;
    version: {
      schema_version: string;
      prompt_version: string;
      model_version: string;
      mock: boolean;
    };
  };
  state: {
    analysis_id: string;
    state_profile: {
      score_range: {
        min: number;
        max: number;
      };
      dimensions: Array<{
        dimension: string;
        score: number;
        low_anchor: string;
        high_anchor: string;
        label: string;
      }>;
    };
    observed_text_signals: Array<{
      signal_id: string;
      signal: string;
      category: string;
      direction: "low" | "high" | "neutral";
      dimensions: string[];
      weight: number;
    }>;
    inferred_state_labels: Array<{
      dimension: string;
      label: string;
      score: number;
      rationale: string;
      supporting_signal_ids: string[];
      confidence: number;
    }>;
    confidence: {
      overall: number;
      by_dimension: Array<{
        dimension: string;
        value: number;
      }>;
    };
    version: {
      schema_version: string;
      prompt_version: string;
      model_version: string;
      mock: boolean;
    };
  };
  context: {
    packet_id: string;
    temporal_focus: string;
    context_brief: string;
    retrieval_context: Array<{
      chunk_id: string;
      source_file?: string;
      relevance_score: number;
      rationale: string;
    }>;
    graph_context: Array<{
      subject: string;
      relation: string;
      object: string;
      weight: number;
    }>;
    version: {
      schema_version: string;
      prompt_version: string;
      model_version: string;
      mock: boolean;
    };
  };
}

export interface FeedbackHealthData {
  total_events: number;
  active_events: number;
  archived_events: number;
  up_count: number;
  down_count: number;
  concepts_affected: number;
  last_tuning: string | null;
  last_aggregation: string | null;
  velocity_7d: number;
}

export interface TuningPreviewData {
  concept_boosts: Record<string, number>;
  graduated_count: number;
  skipped: string[];
}

export interface TuningApplyResult {
  ok: boolean;
  graduated_count: number;
  concept_boosts: Record<string, number>;
  tuning_path: string;
}

export interface AggregateResult {
  archived: number;
  remaining: number;
}

export type ChatMode = "classic" | "converse";

export interface ModelInfo {
  id: string;
  label: string;
  size_gb: number | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
  default: string;
}

export async function getModels(): Promise<ModelsResponse> {
  const response = await fetch("/api/models");
  if (!response.ok) {
    throw new Error(`Models request failed: ${response.status}`);
  }
  return response.json();
}

export async function sendMessage(
  conversationId: string,
  message: string,
  onEvent: (event: SSEEvent) => void,
  graphrag: boolean = false,
  mode: ChatMode = "classic",
  signal?: AbortSignal,
  replaceLastUser: boolean = false,
  requestId?: string,
  model?: string
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
      graphrag,
      mode,
      replace_last_user: replaceLastUser,
      request_id: requestId,
      model,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop()!;

    for (const part of parts) {
      const lines = part.split("\n");
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6)) as SSEEvent;
            if (requestId && data.request_id && data.request_id !== requestId) {
              continue;
            }
            onEvent(data);
          } catch {
            // skip malformed lines
          }
        }
      }
    }
  }

  // Process any remaining buffer
  if (buffer.trim()) {
    const lines = buffer.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6)) as SSEEvent;
          if (requestId && data.request_id && data.request_id !== requestId) {
            continue;
          }
          onEvent(data);
        } catch {
          // skip
        }
      }
    }
  }
}

export async function checkHealth(): Promise<{
  status: string;
  ollama: string;
  graph: string;
  analysis?: string;
}> {
  const response = await fetch("/api/health");
  return response.json();
}

export async function analyzeEntry(
  payload: AnalysisBundleRequest
): Promise<AnalysisBundleResponse> {
  const response = await fetch("/api/analysis/entry", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Analysis request failed: ${response.status}`);
  }
  return response.json();
}

export async function submitGraphFeedback(
  payload: GraphFeedbackPayload
): Promise<Record<string, unknown>> {
  const response = await fetch("/api/graph/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Feedback request failed: ${response.status}`);
  }

  return response.json();
}

export async function getGraphFeedbackReview(
  topN: number = 15,
  recentLimit: number = 40
): Promise<FeedbackReviewData> {
  const response = await fetch(
    `/api/graph/feedback_review?top_n=${topN}&recent_limit=${recentLimit}`
  );
  if (!response.ok) {
    throw new Error(`Feedback review request failed: ${response.status}`);
  }
  return response.json();
}

export async function getGraphFeedbackHealth(): Promise<FeedbackHealthData> {
  const response = await fetch("/api/graph/feedback/health");
  if (!response.ok) {
    throw new Error(`Feedback health request failed: ${response.status}`);
  }
  return response.json();
}

export async function getGraphFeedbackTuningPreview(): Promise<TuningPreviewData> {
  const response = await fetch("/api/graph/feedback/tuning_preview");
  if (!response.ok) {
    throw new Error(`Tuning preview request failed: ${response.status}`);
  }
  return response.json();
}

export async function applyGraphFeedbackTuning(): Promise<TuningApplyResult> {
  const response = await fetch("/api/graph/feedback/apply_tuning", {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Apply tuning request failed: ${response.status}`);
  }
  return response.json();
}

export async function aggregateGraphFeedback(): Promise<AggregateResult> {
  const response = await fetch("/api/graph/feedback/aggregate", {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Aggregate request failed: ${response.status}`);
  }
  return response.json();
}
