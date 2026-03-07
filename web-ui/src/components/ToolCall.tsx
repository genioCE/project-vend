import type { ToolCallInfo } from "../App";

const TOOL_LABELS: Record<string, string> = {
  search_writings: "Searching writings",
  get_entries_by_date: "Fetching entries by date",
  find_recurring_themes: "Finding recurring themes",
  get_writing_stats: "Getting writing stats",
  get_recent_entries: "Getting recent entries",
  search_by_keyword: "Searching by keyword",
  graph_search: "GraphRAG search",
  find_connected_concepts: "Finding connected concepts",
  trace_concept_evolution: "Tracing concept evolution",
  get_concept_flows: "Tracing concept flows",
  find_entity_relationships: "Finding entity relationships",
  compare_periods: "Comparing periods",
  get_decision_context: "Finding decisions",
  get_archetype_patterns: "Finding archetype patterns",
  summarize_entry: "Summarizing entry",
  label_internal_state: "Labeling state",
  build_context_packet: "Building context packet",
};

function formatInput(input: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined && value !== null) {
      parts.push(`${key}: "${value}"`);
    }
  }
  return parts.join(", ");
}

interface ToolCallProps {
  toolCall: ToolCallInfo;
}

export default function ToolCall({ toolCall }: ToolCallProps) {
  const label = TOOL_LABELS[toolCall.tool] || toolCall.tool;
  const inputStr = formatInput(toolCall.input);
  const isCalling = toolCall.status === "calling";
  const isGraphTool =
    toolCall.tool.startsWith("graph") ||
    toolCall.tool.startsWith("find_connected") ||
    toolCall.tool.startsWith("trace_") ||
    toolCall.tool.startsWith("get_concept_flows") ||
    toolCall.tool.startsWith("find_entity") ||
    toolCall.tool.startsWith("compare_") ||
    toolCall.tool.startsWith("get_decision") ||
    toolCall.tool.startsWith("get_archetype");

  return (
    <div className={`tool-call${isGraphTool ? " tool-call-graph" : ""}`}>
      {isCalling ? (
        <span className="tool-call-spinner">{"\u21BB"}</span>
      ) : (
        <span className="tool-call-done">{"\u2713"}</span>
      )}
      <span className="tool-call-name">{label}</span>
      {inputStr && <span className="tool-call-input">({inputStr})</span>}
    </div>
  );
}
