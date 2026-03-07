import { useState, useEffect, useCallback } from "react";
import Chat from "./components/Chat";
import GraphPanel from "./components/GraphPanel";
import FeedbackReviewPanel from "./components/FeedbackReviewPanel";
import { getModels, type ModelInfo } from "./api";
import {
  initStorage,
  getConversations,
  saveConversations,
  getMessages,
  saveMessages,
} from "./storage";
import type { Conversation, DisplayMessage } from "./storage";

export type ChatMode = "classic" | "converse";

export default function App() {
  const [loading, setLoading] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [graphRagMode, setGraphRagMode] = useState(false);
  const [chatMode, setChatMode] = useState<ChatMode>("classic");
  const [graphPanel, setGraphPanel] = useState<{ visible: boolean; center?: string }>({
    visible: false,
  });
  const [feedbackReviewVisible, setFeedbackReviewVisible] = useState(false);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);

  // Initialize IndexedDB and load conversations
  useEffect(() => {
    let cancelled = false;
    initStorage()
      .then(() => getConversations())
      .then((convs) => {
        if (cancelled) return;
        setConversations(convs);
        setCurrentId(convs[0]?.id ?? null);
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  // Fetch available models
  useEffect(() => {
    let cancelled = false;
    getModels()
      .then((data) => {
        if (cancelled) return;
        setAvailableModels(data.models);
        setSelectedModel(data.default);
      })
      .catch(() => {
        // Models endpoint unavailable — no dropdown shown
      });
    return () => { cancelled = true; };
  }, []);

  // Load messages when switching conversations
  useEffect(() => {
    if (currentId) {
      getMessages(currentId).then(setMessages);
    } else {
      setMessages([]);
    }
  }, [currentId]);

  // Persist messages when they change
  const persistMessages = useCallback(
    (msgs: DisplayMessage[]) => {
      if (currentId) {
        saveMessages(currentId, msgs);
      }
    },
    [currentId]
  );

  function handleNewChat() {
    const id = crypto.randomUUID();
    const conv: Conversation = {
      id,
      title: "New conversation",
      createdAt: Date.now(),
    };
    const updated = [conv, ...conversations];
    setConversations(updated);
    saveConversations(updated);
    setCurrentId(id);
    setMessages([]);
  }

  function handleSelectConversation(id: string) {
    setCurrentId(id);
  }

  function handleUpdateTitle(id: string, title: string) {
    setConversations((prev) => {
      const updated = prev.map((c) =>
        c.id === id ? { ...c, title: title.substring(0, 50) } : c
      );
      saveConversations(updated);
      return updated;
    });
  }

  function handleMessagesChange(msgs: DisplayMessage[]) {
    setMessages(msgs);
    persistMessages(msgs);
  }

  function handleShowGraph(center: string) {
    setGraphPanel({ visible: true, center });
  }

  if (loading) {
    return null;
  }

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={handleNewChat}>
            + New Chat
          </button>
        </div>
        <div className="sidebar-list">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`sidebar-item ${conv.id === currentId ? "active" : ""}`}
              onClick={() => handleSelectConversation(conv.id)}
            >
              <div>{conv.title}</div>
              <div className="sidebar-item-date">
                {new Date(conv.createdAt).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          Model: {selectedModel || "default"} &middot; local
        </div>
      </div>

      <div className="main">
        <div className="main-header">
          <span>Corpus Intelligence</span>
          <div className="main-header-actions">
            {availableModels.length > 0 && (
              <select
                className="model-chooser"
                value={selectedModel || ""}
                onChange={(e) => setSelectedModel(e.target.value || undefined)}
              >
                {availableModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}{m.size_gb ? ` (${m.size_gb}GB)` : ""}
                  </option>
                ))}
              </select>
            )}
            <button
              className="feedback-review-btn"
              onClick={() => setFeedbackReviewVisible(true)}
            >
              Review Feedback
            </button>
            <div className="chat-mode-toggle">
              <button
                className={`chat-mode-btn ${chatMode === "classic" ? "active" : ""}`}
                onClick={() => setChatMode("classic")}
              >
                Classic
              </button>
              <button
                className={`chat-mode-btn ${chatMode === "converse" ? "active" : ""}`}
                onClick={() => setChatMode("converse")}
              >
                Converse
              </button>
            </div>
            <label className="graphrag-toggle">
              <input
                type="checkbox"
                checked={graphRagMode}
                onChange={(e) => setGraphRagMode(e.target.checked)}
              />
              <span className="graphrag-toggle-slider" />
              <span className="graphrag-toggle-label">GraphRAG</span>
            </label>
          </div>
        </div>
        {currentId ? (
          <Chat
            conversationId={currentId}
            messages={messages}
            onMessagesChange={handleMessagesChange}
            onUpdateTitle={(title) => handleUpdateTitle(currentId, title)}
            graphRagMode={graphRagMode}
            chatMode={chatMode}
            onShowGraph={handleShowGraph}
            selectedModel={selectedModel}
          />
        ) : (
          <div className="messages">
            <div className="messages-empty">
              Start a new conversation to explore your writing corpus.
            </div>
          </div>
        )}
      </div>

      <GraphPanel
        visible={graphPanel.visible}
        center={graphPanel.center}
        onClose={() => setGraphPanel({ visible: false })}
      />
      <FeedbackReviewPanel
        visible={feedbackReviewVisible}
        onClose={() => setFeedbackReviewVisible(false)}
      />
    </div>
  );
}

export type { Conversation, DisplayMessage, ToolCallInfo } from "./storage";
