import { useState, useRef, useEffect } from "react";
import {
  analyzeEntry,
  sendMessage,
  submitGraphFeedback,
  type AnalysisBundleResponse,
} from "../api";
import Message from "./Message";
import type { ChatMode, DisplayMessage, ToolCallInfo } from "../App";

interface ChatProps {
  conversationId: string;
  messages: DisplayMessage[];
  onMessagesChange: (messages: DisplayMessage[]) => void;
  onUpdateTitle: (title: string) => void;
  graphRagMode: boolean;
  chatMode: ChatMode;
  onShowGraph: (center: string) => void;
  selectedModel?: string;
}

interface PendingOffTargetFeedback {
  messageId: string;
  center: string;
  query?: string;
}

interface AnalysisModalState {
  request?: AnalysisBundleResponse;
  error?: string;
  entryPreview: string;
}

interface StartStreamOptions {
  replaceLastUser?: boolean;
  reuseUserMessageId?: string;
}

function normalizeQuery(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function findLastUserIndex(msgs: DisplayMessage[]): number {
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === "user") return i;
  }
  return -1;
}

function getLatestUserMessageId(msgs: DisplayMessage[]): string | undefined {
  const idx = findLastUserIndex(msgs);
  if (idx < 0) return undefined;
  return msgs[idx].id;
}

function getLatestUserMessage(msgs: DisplayMessage[]): DisplayMessage | undefined {
  const idx = findLastUserIndex(msgs);
  return idx >= 0 ? msgs[idx] : undefined;
}

function toPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export default function Chat({
  conversationId,
  messages,
  onMessagesChange,
  onUpdateTitle,
  graphRagMode,
  chatMode,
  onShowGraph,
  selectedModel,
}: ChatProps) {
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isEditQueued, setIsEditQueued] = useState(false);
  const [editingUserMessageId, setEditingUserMessageId] = useState<string | null>(
    null
  );
  const [editingUserDraft, setEditingUserDraft] = useState("");
  const [streamingMessage, setStreamingMessage] = useState<DisplayMessage | null>(null);
  const [feedbackByMessage, setFeedbackByMessage] = useState<Record<string, "up" | "down">>({});
  const [pendingOffTarget, setPendingOffTarget] = useState<PendingOffTargetFeedback | null>(null);
  const [offTargetNote, setOffTargetNote] = useState("");
  const [offTargetSaving, setOffTargetSaving] = useState(false);
  const [offTargetError, setOffTargetError] = useState<string | null>(null);
  const [analysisModal, setAnalysisModal] = useState<AnalysisModalState | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesRef = useRef(messages);
  const inFlightPromptRef = useRef<string | null>(null);
  const inFlightUserMessageIdRef = useRef<string | null>(null);
  const pendingResubmitRef = useRef<string | null>(null);
  const awaitingEditAbortRef = useRef(false);
  const suppressAbortCommitRef = useRef(false);
  const activeRequestIdRef = useRef<string | null>(null);

  function commitMessages(next: DisplayMessage[]): void {
    messagesRef.current = next;
    onMessagesChange(next);
  }

  function appendAssistantMessage(message: DisplayMessage): void {
    commitMessages([...messagesRef.current, message]);
  }

  function updateInFlightUserMessageContent(nextContent: string): void {
    const activeId = inFlightUserMessageIdRef.current;
    if (!activeId) return;

    const idx = messagesRef.current.findIndex(
      (msg) => msg.id === activeId && msg.role === "user"
    );
    if (idx < 0) return;
    if (messagesRef.current[idx].content === nextContent) return;

    const next = [...messagesRef.current];
    next[idx] = { ...next[idx], content: nextContent };
    commitMessages(next);
  }

  function queueEditResubmit(nextText: string): void {
    const activePrompt = inFlightPromptRef.current;
    if (!activePrompt) return;

    const normalizedNext = normalizeQuery(nextText);
    const normalizedActive = normalizeQuery(activePrompt);
    if (!normalizedNext || normalizedNext === normalizedActive) {
      pendingResubmitRef.current = null;
      setIsEditQueued(false);
      return;
    }

    pendingResubmitRef.current = nextText.trim();
    setIsEditQueued(true);
    if (!awaitingEditAbortRef.current) {
      awaitingEditAbortRef.current = true;
      abortControllerRef.current?.abort();
    }
  }

  function getPendingResubmitText(): string {
    const pending = pendingResubmitRef.current;
    return typeof pending === "string" ? pending.trim() : "";
  }

  function isActiveRequest(requestId: string): boolean {
    return activeRequestIdRef.current === requestId;
  }

  async function startStream(
    rawText: string,
    options: StartStreamOptions = {}
  ): Promise<void> {
    const text = rawText.trim();
    if (!text) return;
    const requestId = crypto.randomUUID();
    activeRequestIdRef.current = requestId;

    if (!options.replaceLastUser && messagesRef.current.length === 0) {
      onUpdateTitle(text.substring(0, 50));
    }

    let nextMessages = messagesRef.current;
    let userMessageId = options.reuseUserMessageId;

    if (options.replaceLastUser) {
      let userIndex = -1;
      if (options.reuseUserMessageId) {
        userIndex = nextMessages.findIndex(
          (msg) => msg.id === options.reuseUserMessageId && msg.role === "user"
        );
      }
      if (userIndex < 0) {
        userIndex = findLastUserIndex(nextMessages);
      }

      if (userIndex >= 0) {
        const existing = nextMessages[userIndex];
        userMessageId = existing.id;
        if (existing.content !== text) {
          const updated = [...nextMessages];
          updated[userIndex] = { ...existing, content: text };
          nextMessages = updated;
        }
      } else {
        const userMsg: DisplayMessage = {
          id: crypto.randomUUID(),
          role: "user",
          content: text,
        };
        userMessageId = userMsg.id;
        nextMessages = [...nextMessages, userMsg];
      }
    } else {
      const userMsg: DisplayMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
      };
      userMessageId = userMsg.id;
      nextMessages = [...nextMessages, userMsg];
    }

    commitMessages(nextMessages);

    setIsStreaming(true);
    setIsEditQueued(false);
    inFlightPromptRef.current = text;
    inFlightUserMessageIdRef.current = userMessageId || null;
    pendingResubmitRef.current = null;
    awaitingEditAbortRef.current = false;
    suppressAbortCommitRef.current = false;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const assistantMsg: DisplayMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      toolCalls: [],
    };
    setStreamingMessage(assistantMsg);

    let currentContent = "";
    let currentToolCalls: ToolCallInfo[] = [];
    let committed = false;

    try {
      await sendMessage(
        conversationId,
        text,
        (event) => {
          if (!isActiveRequest(requestId)) return;
          switch (event.type) {
            case "tool_call":
              currentToolCalls = [
                ...currentToolCalls,
                {
                  tool: event.tool!,
                  input: event.input || {},
                  status: "calling",
                },
              ];
              setStreamingMessage((prev) =>
                prev ? { ...prev, toolCalls: [...currentToolCalls] } : null
              );
              break;

            case "tool_result":
              currentToolCalls = currentToolCalls.map((tc) =>
                tc.tool === event.tool && tc.status === "calling"
                  ? {
                      ...tc,
                      status: "done" as const,
                      preview: event.preview,
                      center: event.center || tc.center,
                    }
                  : tc
              );
              setStreamingMessage((prev) =>
                prev ? { ...prev, toolCalls: [...currentToolCalls] } : null
              );
              break;

            case "text_delta":
              currentContent += event.content || "";
              setStreamingMessage((prev) =>
                prev ? { ...prev, content: currentContent } : null
              );
              break;

            case "error":
              currentContent += `\n\nError: ${event.message}`;
              setStreamingMessage((prev) =>
                prev ? { ...prev, content: currentContent } : null
              );
              break;

            case "done":
              break;
          }
        },
        graphRagMode,
        chatMode,
        controller.signal,
        options.replaceLastUser === true,
        requestId,
        selectedModel
      );

      if (!isActiveRequest(requestId)) return;

      const finalMsg: DisplayMessage = {
        ...assistantMsg,
        content: currentContent,
        toolCalls: currentToolCalls,
      };
      appendAssistantMessage(finalMsg);
      committed = true;
    } catch (err) {
      if (!isActiveRequest(requestId)) return;
      const isAbort =
        (err instanceof DOMException && err.name === "AbortError") ||
        (err instanceof Error && /abort/i.test(err.message));

      const queued = getPendingResubmitText();
      const shouldAutoResubmit =
        queued.length > 0 && normalizeQuery(queued) !== normalizeQuery(text);
      const suppressAbortCommit = suppressAbortCommitRef.current;

      if (isAbort) {
        if (shouldAutoResubmit || suppressAbortCommit) {
          committed = true;
        } else {
          const partial = currentContent.trim()
            ? currentContent
            : "[Generation stopped]";
          const stoppedMsg: DisplayMessage = {
            ...assistantMsg,
            content: partial,
            toolCalls: currentToolCalls,
          };
          appendAssistantMessage(stoppedMsg);
          committed = true;
        }
      } else {
        const errorMsg: DisplayMessage = {
          ...assistantMsg,
          content: `Connection error: ${String(err)}`,
        };
        appendAssistantMessage(errorMsg);
        committed = true;
      }
    } finally {
      if (!isActiveRequest(requestId)) return;
      const queued = getPendingResubmitText();
      const shouldAutoResubmit =
        queued.length > 0 && normalizeQuery(queued) !== normalizeQuery(text);
      const queuedText = shouldAutoResubmit ? queued : "";
      const replaceUserMessageId = inFlightUserMessageIdRef.current || undefined;

      if (!committed && currentContent.trim()) {
        appendAssistantMessage({
          ...assistantMsg,
          content: currentContent,
          toolCalls: currentToolCalls,
        });
      }

      abortControllerRef.current = null;
      setStreamingMessage(null);
      setIsStreaming(false);
      awaitingEditAbortRef.current = false;
      inFlightPromptRef.current = null;
      suppressAbortCommitRef.current = false;

      if (shouldAutoResubmit) {
        pendingResubmitRef.current = null;
        void startStream(queuedText, {
          replaceLastUser: true,
          reuseUserMessageId: replaceUserMessageId,
        });
        return;
      }

      inFlightUserMessageIdRef.current = null;
      pendingResubmitRef.current = null;
      activeRequestIdRef.current = null;
      setIsEditQueued(false);
    }
  }

  function handleInputChange(nextValue: string): void {
    setInput(nextValue);
  }

  function handleSend(): void {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    void startStream(text);
  }

  async function handleAnalyzeLastEntry(): Promise<void> {
    if (isStreaming || isAnalyzing) return;
    const latestUserMessage = getLatestUserMessage(messagesRef.current);
    if (!latestUserMessage || latestUserMessage.role !== "user") return;

    setIsAnalyzing(true);
    try {
      const chunkId = `ui-${latestUserMessage.id}-chunk-000`;
      const payload = {
        entry_id: latestUserMessage.id,
        text: latestUserMessage.content,
        source_file: `ui://${conversationId}`,
        chunk_ids: [chunkId],
        query: latestUserMessage.content.slice(0, 400),
      };
      const result = await analyzeEntry(payload);
      setAnalysisModal({
        request: result,
        entryPreview: latestUserMessage.content,
      });
    } catch (err) {
      setAnalysisModal({
        error: `Analysis error: ${String(err)}`,
        entryPreview: latestUserMessage.content,
      });
    } finally {
      setIsAnalyzing(false);
    }
  }

  function handleStartInlineEdit(messageId: string, content: string): void {
    const latestUserId = getLatestUserMessageId(messagesRef.current);
    if (!latestUserId || latestUserId !== messageId) return;

    if (isStreaming) {
      pendingResubmitRef.current = null;
      awaitingEditAbortRef.current = false;
      setIsEditQueued(false);
      suppressAbortCommitRef.current = true;
      abortControllerRef.current?.abort();
    }

    setEditingUserMessageId(messageId);
    setEditingUserDraft(content);
  }

  function handleCancelInlineEdit(): void {
    setEditingUserMessageId(null);
    setEditingUserDraft("");
  }

  function handleSubmitInlineEdit(): void {
    const editId = editingUserMessageId;
    const nextText = editingUserDraft.trim();
    if (!editId || !nextText) return;

    const currentMessages = messagesRef.current;
    const userIndex = currentMessages.findIndex(
      (msg) => msg.id === editId && msg.role === "user"
    );
    if (userIndex < 0) return;

    const latestUserIndex = findLastUserIndex(currentMessages);
    if (latestUserIndex !== userIndex) return;

    const currentUserMessage = currentMessages[userIndex];
    if (normalizeQuery(currentUserMessage.content) === normalizeQuery(nextText)) {
      setEditingUserMessageId(null);
      setEditingUserDraft("");
      return;
    }

    const updatedUser: DisplayMessage = {
      ...currentUserMessage,
      content: nextText,
    };

    const truncated = [...currentMessages.slice(0, userIndex), updatedUser];
    commitMessages(truncated);
    setEditingUserMessageId(null);
    setEditingUserDraft("");

    if (isStreaming) {
      inFlightUserMessageIdRef.current = updatedUser.id;
      updateInFlightUserMessageContent(nextText);
      queueEditResubmit(nextText);
      return;
    }

    void startStream(nextText, {
      replaceLastUser: true,
      reuseUserMessageId: updatedUser.id,
    });
  }

  function handleStop() {
    setEditingUserMessageId(null);
    setEditingUserDraft("");
    pendingResubmitRef.current = null;
    awaitingEditAbortRef.current = false;
    suppressAbortCommitRef.current = false;
    setIsEditQueued(false);
    abortControllerRef.current?.abort();
  }

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if (!editingUserMessageId) return;
    const exists = messages.some(
      (msg) => msg.id === editingUserMessageId && msg.role === "user"
    );
    const latestUserId = getLatestUserMessageId(messages);
    if (!exists || latestUserId !== editingUserMessageId) {
      setEditingUserMessageId(null);
      setEditingUserDraft("");
    }
  }, [messages, editingUserMessageId]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingMessage]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 200) + "px";
    }
  }, [input]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      activeRequestIdRef.current = null;
    };
  }, []);

  useEffect(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsStreaming(false);
    setIsEditQueued(false);
    setEditingUserMessageId(null);
    setEditingUserDraft("");
    setStreamingMessage(null);
    setInput("");
    setIsAnalyzing(false);
    inFlightPromptRef.current = null;
    inFlightUserMessageIdRef.current = null;
    pendingResubmitRef.current = null;
    awaitingEditAbortRef.current = false;
    suppressAbortCommitRef.current = false;
    activeRequestIdRef.current = null;

    setFeedbackByMessage({});
    setPendingOffTarget(null);
    setOffTargetNote("");
    setOffTargetError(null);
    setAnalysisModal(null);
  }, [conversationId]);

  async function submitOffTargetFeedback(withNote: boolean) {
    if (!pendingOffTarget) return;
    setOffTargetSaving(true);
    setOffTargetError(null);
    try {
      await submitGraphFeedback({
        signal: "down",
        query: pendingOffTarget.query,
        concepts: [pendingOffTarget.center],
        note: withNote ? offTargetNote.trim() || undefined : undefined,
      });
      setFeedbackByMessage((prev) => ({
        ...prev,
        [pendingOffTarget.messageId]: "down",
      }));
      setPendingOffTarget(null);
      setOffTargetNote("");
    } catch (err) {
      setOffTargetError(String(err));
    } finally {
      setOffTargetSaving(false);
    }
  }

  async function handleGraphFeedback(
    messageId: string,
    signal: "up" | "down",
    center: string,
    query?: string
  ) {
    if (signal === "down") {
      setPendingOffTarget({ messageId, center, query });
      setOffTargetNote("");
      setOffTargetError(null);
      return;
    }

    setFeedbackByMessage((prev) => ({ ...prev, [messageId]: "up" }));
    try {
      await submitGraphFeedback({
        signal,
        query,
        concepts: [center],
      });
    } catch {
      setFeedbackByMessage((prev) => {
        const next = { ...prev };
        delete next[messageId];
        return next;
      });
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSend();
    }
  }

  const allMessages = streamingMessage
    ? [...messages, streamingMessage]
    : messages;
  const latestUserMessageId = getLatestUserMessageId(messages);

  return (
    <>
      <div className="messages">
        {allMessages.length === 0 ? (
          <div className="messages-empty">
            {chatMode === "converse"
              ? graphRagMode
                ? "Converse with your thoughts — grounded by your corpus and graph context."
                : "Converse with your thoughts."
              : graphRagMode
                ? "GraphRAG mode — ask about concepts, people, patterns, or relationships..."
                : "Ask about your writing..."}
          </div>
        ) : (
          allMessages.map((msg, i) => (
            (() => {
              let priorUserMessage: string | undefined;
              for (let j = i - 1; j >= 0; j--) {
                if (allMessages[j].role === "user") {
                  priorUserMessage = allMessages[j].content;
                  break;
                }
              }
              return (
                <Message
                  key={msg.id}
                  message={msg}
                  isStreaming={
                    streamingMessage !== null && i === allMessages.length - 1
                  }
                  canEditUser={msg.role === "user" && msg.id === latestUserMessageId}
                  isEditingUser={msg.role === "user" && msg.id === editingUserMessageId}
                  userEditDraft={msg.id === editingUserMessageId ? editingUserDraft : undefined}
                  onStartUserEdit={() =>
                    msg.role === "user"
                      ? handleStartInlineEdit(msg.id, msg.content)
                      : undefined
                  }
                  onChangeUserEditDraft={setEditingUserDraft}
                  onCancelUserEdit={handleCancelInlineEdit}
                  onSubmitUserEdit={handleSubmitInlineEdit}
                  userEditSubmitting={isStreaming && msg.id === editingUserMessageId}
                  onShowGraph={onShowGraph}
                  onGraphFeedback={(signal, center) =>
                    handleGraphFeedback(msg.id, signal, center, priorUserMessage)
                  }
                  graphFeedbackSignal={feedbackByMessage[msg.id]}
                />
              );
            })()
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <div className="input-row">
          <textarea
            ref={textareaRef}
            className="input-textarea"
            value={input}
            onChange={(e) => handleInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              chatMode === "converse"
                ? graphRagMode
                  ? "Converse with your thoughts (graph-grounded when helpful)..."
                  : "Converse with your thoughts..."
                : graphRagMode
                  ? "Ask about relationships, patterns, archetypes..."
                  : "Ask about your writing..."
            }
            rows={1}
            disabled={isStreaming}
          />
          <button
            className="send-btn"
            onClick={isStreaming ? handleStop : handleSend}
            disabled={!isStreaming && !input.trim()}
          >
            {isStreaming ? "Stop" : "Send"}
          </button>
          <button
            className="analysis-btn"
            onClick={() => void handleAnalyzeLastEntry()}
            disabled={isStreaming || isAnalyzing || !getLatestUserMessage(messages)}
            title="Run mock analysis on the latest user message"
          >
            {isAnalyzing ? "Analyzing..." : "Analyze Last Entry"}
          </button>
        </div>
        <div className="input-hint">
          {"\u2318"}+Enter to send
          {isStreaming && (
            <span className="graphrag-badge">
              {isEditQueued
                ? "Updating..."
                : "Edit latest user message to resubmit"}
            </span>
          )}
          {graphRagMode && <span className="graphrag-badge">GraphRAG</span>}
          <span className="graphrag-badge">
            {chatMode === "converse" ? "Converse" : "Classic"}
          </span>
        </div>
      </div>

      {pendingOffTarget && (
        <div className="offtarget-modal-overlay" onClick={() => setPendingOffTarget(null)}>
          <div className="offtarget-modal" onClick={(e) => e.stopPropagation()}>
            <div className="offtarget-modal-title">How can we improve this answer?</div>
            <div className="offtarget-modal-subtitle">
              Your note will be saved and used to generate improvement prompts for system-language tuning.
            </div>
            <textarea
              className="offtarget-textarea"
              value={offTargetNote}
              onChange={(e) => setOffTargetNote(e.target.value)}
              placeholder="Example: Too repetitive, and I want more dated evidence tied to specific entries."
              rows={5}
              maxLength={2000}
            />
            {offTargetError && <div className="offtarget-error">{offTargetError}</div>}
            <div className="offtarget-actions">
              <button
                className="offtarget-btn"
                onClick={() => setPendingOffTarget(null)}
                disabled={offTargetSaving}
              >
                Cancel
              </button>
              <button
                className="offtarget-btn"
                onClick={() => void submitOffTargetFeedback(false)}
                disabled={offTargetSaving}
              >
                Submit Without Note
              </button>
              <button
                className="offtarget-btn primary"
                onClick={() => void submitOffTargetFeedback(true)}
                disabled={offTargetSaving}
              >
                {offTargetSaving ? "Saving..." : "Submit Feedback"}
              </button>
            </div>
          </div>
        </div>
      )}

      {analysisModal && (
        <div
          className="analysis-modal-overlay"
          onClick={() => setAnalysisModal(null)}
        >
          <div className="analysis-modal" onClick={(e) => e.stopPropagation()}>
            <div className="analysis-modal-header">
              <div>
                <div className="analysis-modal-title">Entry Analysis</div>
                <div className="analysis-modal-subtitle">
                  {analysisModal.entryPreview.slice(0, 120)}
                  {analysisModal.entryPreview.length > 120 ? "..." : ""}
                </div>
              </div>
              <button
                className="graph-panel-close"
                onClick={() => setAnalysisModal(null)}
              >
                {"\u2715"}
              </button>
            </div>

            <div className="analysis-modal-body">
              {analysisModal.error && (
                <div className="analysis-modal-error">{analysisModal.error}</div>
              )}

              {!analysisModal.error && analysisModal.request && (
                <>
                  <div className="analysis-section">
                    <div className="analysis-section-title">Entry Summary</div>
                    <p className="analysis-summary-text">
                      {analysisModal.request.summary.summary}
                    </p>
                    <div className="analysis-meta-line">
                      Coverage:{" "}
                      {toPercent(analysisModal.request.summary.coverage_ratio)}
                    </div>
                    <div className="analysis-meta-line">
                      Key terms:{" "}
                      {analysisModal.request.summary.key_terms.slice(0, 6).join(", ") || "n/a"}
                    </div>
                  </div>

                  <div className="analysis-section">
                    <div className="analysis-section-title">State Labels</div>
                    <div className="analysis-meta-line">
                      Overall confidence:{" "}
                      {toPercent(analysisModal.request.state.confidence.overall)}
                    </div>
                    <ul className="analysis-list">
                      {analysisModal.request.state.inferred_state_labels.map((item) => (
                        <li key={`${item.dimension}-${item.label}`}>
                          <strong>{item.dimension}</strong>: {item.label} (score{" "}
                          {item.score.toFixed(2)}, confidence {toPercent(item.confidence)})
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="analysis-section">
                    <div className="analysis-section-title">Context Packet</div>
                    <div className="analysis-meta-line">
                      Packet: {analysisModal.request.context.packet_id}
                    </div>
                    <div className="analysis-meta-line">
                      Temporal focus: {analysisModal.request.context.temporal_focus}
                    </div>
                    <div className="analysis-subtitle">Retrieval anchors</div>
                    {analysisModal.request.context.retrieval_context.length === 0 ? (
                      <div className="analysis-meta-line">None</div>
                    ) : (
                      <ul className="analysis-list">
                        {analysisModal.request.context.retrieval_context
                          .slice(0, 5)
                          .map((item) => (
                            <li key={item.chunk_id}>
                              <code>{item.chunk_id}</code> ({toPercent(item.relevance_score)})
                            </li>
                          ))}
                      </ul>
                    )}
                  </div>

                  <div className="analysis-section">
                    <div className="analysis-section-title">Version</div>
                    <div className="analysis-meta-line">
                      schema: {analysisModal.request.summary.version.schema_version}
                    </div>
                    <div className="analysis-meta-line">
                      prompt: {analysisModal.request.summary.version.prompt_version}
                    </div>
                    <div className="analysis-meta-line">
                      model: {analysisModal.request.summary.version.model_version}
                    </div>
                    <div className="analysis-meta-line">
                      mock: {String(analysisModal.request.summary.version.mock)}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
