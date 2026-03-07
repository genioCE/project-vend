import { useMemo } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import ToolCall from "./ToolCall";
import type { DisplayMessage } from "../App";

interface MessageProps {
  message: DisplayMessage;
  isStreaming?: boolean;
  onShowGraph?: (center: string) => void;
  onGraphFeedback?: (signal: "up" | "down", center: string) => void;
  graphFeedbackSignal?: "up" | "down";
  canEditUser?: boolean;
  isEditingUser?: boolean;
  userEditDraft?: string;
  onStartUserEdit?: () => void;
  onChangeUserEditDraft?: (value: string) => void;
  onCancelUserEdit?: () => void;
  onSubmitUserEdit?: () => void;
  userEditSubmitting?: boolean;
}

marked.setOptions({
  breaks: true,
  gfm: true,
});

const GRAPH_TOOLS = new Set([
  "graph_search",
  "find_connected_concepts",
  "trace_concept_evolution",
  "get_concept_flows",
  "find_entity_relationships",
  "compare_periods",
  "get_decision_context",
  "get_archetype_patterns",
]);

export default function Message({
  message,
  isStreaming,
  onShowGraph,
  onGraphFeedback,
  graphFeedbackSignal,
  canEditUser,
  isEditingUser,
  userEditDraft,
  onStartUserEdit,
  onChangeUserEditDraft,
  onCancelUserEdit,
  onSubmitUserEdit,
  userEditSubmitting,
}: MessageProps) {
  const html = useMemo(() => {
    if (!message.content) return "";
    const rendered = marked.parse(message.content, { async: false }) as string;
    return DOMPurify.sanitize(rendered);
  }, [message.content]);

  // Extract a concept name from graph tool calls for "Show Graph" button
  const graphConcept = useMemo(() => {
    if (!message.toolCalls) return null;
    for (const tc of message.toolCalls) {
      if (GRAPH_TOOLS.has(tc.tool) && tc.input) {
        const candidates = [tc.center, tc.input.name, tc.input.keyword];
        for (const candidate of candidates) {
          if (typeof candidate === "string") {
            const trimmed = candidate.trim();
            if (trimmed && trimmed.length <= 120) {
              return trimmed;
            }
          }
        }
      }
    }
    return null;
  }, [message.toolCalls]);

  if (message.role === "user") {
    if (isEditingUser) {
      return (
        <div className="message message-user message-user-editing">
          <textarea
            className="message-user-editor"
            value={userEditDraft || ""}
            onChange={(e) => onChangeUserEditDraft?.(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                onSubmitUserEdit?.();
              } else if (e.key === "Escape") {
                e.preventDefault();
                onCancelUserEdit?.();
              }
            }}
            rows={3}
            autoFocus
          />
          <div className="message-user-edit-actions">
            <button
              className="message-user-edit-btn"
              onClick={onCancelUserEdit}
              disabled={userEditSubmitting}
            >
              Cancel
            </button>
            <button
              className="message-user-edit-btn primary"
              onClick={onSubmitUserEdit}
              disabled={userEditSubmitting || !(userEditDraft || "").trim()}
            >
              {userEditSubmitting ? "Updating..." : "Update & Resubmit"}
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className="message message-user">
        <div className="message-user-content">{message.content}</div>
        {canEditUser && (
          <div className="message-user-actions">
            <button className="message-user-edit-link" onClick={onStartUserEdit}>
              Edit
            </button>
          </div>
        )}
      </div>
    );
  }

  const hasToolCalls = message.toolCalls && message.toolCalls.length > 0;
  const hasContent = message.content.length > 0;
  const isThinking =
    isStreaming && !hasContent && (!hasToolCalls || message.toolCalls!.every((tc) => tc.status === "done"));

  return (
    <div className="message message-assistant">
      {hasToolCalls && (
        <div className="tool-calls">
          {message.toolCalls!.map((tc, i) => (
            <ToolCall key={`${tc.tool}-${i}`} toolCall={tc} />
          ))}
        </div>
      )}

      {isThinking && (
        <div className="thinking">
          <div className="thinking-dots">
            <span />
            <span />
            <span />
          </div>
          thinking...
        </div>
      )}

      {hasContent && (
        <div
          className="message-assistant-content"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )}

      {graphConcept && !isStreaming && (
        <div className="graph-actions">
          {onShowGraph && (
            <button
              className="show-graph-btn"
              onClick={() => onShowGraph(graphConcept)}
            >
              Show Graph: {graphConcept}
            </button>
          )}
          {onGraphFeedback && (
            <div className="graph-feedback">
              <button
                className={`graph-feedback-btn ${graphFeedbackSignal === "up" ? "active" : ""}`}
                onClick={() => onGraphFeedback("up", graphConcept)}
              >
                Helpful
              </button>
              <button
                className={`graph-feedback-btn ${graphFeedbackSignal === "down" ? "active" : ""}`}
                onClick={() => onGraphFeedback("down", graphConcept)}
              >
                Off-target
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
