import { useEffect, useState } from "react";
import {
  getGraphFeedbackReview,
  getGraphFeedbackHealth,
  getGraphFeedbackTuningPreview,
  applyGraphFeedbackTuning,
  type FeedbackReviewData,
  type FeedbackPromptItem,
  type FeedbackHealthData,
  type TuningPreviewData,
} from "../api";

interface FeedbackReviewPanelProps {
  visible: boolean;
  onClose: () => void;
}

function formatTimestamp(value: string): string {
  if (!value) return "";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;
  return dt.toLocaleString();
}

function toClipboard(text: string): void {
  void navigator.clipboard?.writeText(text);
}

function PromptCard({ item }: { item: FeedbackPromptItem }) {
  return (
    <div className="feedback-card" key={item.id}>
      <div className="feedback-card-top">
        <div>
          <div className="feedback-card-title">{item.title}</div>
          <div className="feedback-card-meta">
            {item.count} reports {item.latest_at ? `· latest ${formatTimestamp(item.latest_at)}` : ""}
          </div>
        </div>
        <button className="feedback-copy-btn" onClick={() => toClipboard(item.prompt)}>
          Copy Prompt
        </button>
      </div>
      {item.sample_notes.length > 0 && (
        <div className="feedback-notes">
          {item.sample_notes.slice(0, 2).map((note, idx) => (
            <div key={`${item.id}-note-${idx}`}>"{note}"</div>
          ))}
        </div>
      )}
      <pre className="feedback-prompt">{item.prompt}</pre>
    </div>
  );
}

function HealthBar({ health }: { health: FeedbackHealthData }) {
  return (
    <div className="feedback-health-bar">
      <div className="feedback-health-item">
        <span className="feedback-health-value">{health.velocity_7d}</span>
        <span className="feedback-health-label">events/day</span>
      </div>
      <div className="feedback-health-item">
        <span className="feedback-health-value">{health.active_events}</span>
        <span className="feedback-health-label">active</span>
      </div>
      <div className="feedback-health-item">
        <span className="feedback-health-value">{health.archived_events}</span>
        <span className="feedback-health-label">archived</span>
      </div>
      <div className="feedback-health-item">
        <span className="feedback-health-value">{health.concepts_affected}</span>
        <span className="feedback-health-label">concepts</span>
      </div>
      <div className="feedback-health-item">
        <span className="feedback-health-value">
          {health.last_tuning ? formatTimestamp(health.last_tuning) : "never"}
        </span>
        <span className="feedback-health-label">last tuning</span>
      </div>
    </div>
  );
}

function TuningSection() {
  const [preview, setPreview] = useState<TuningPreviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function loadPreview() {
    setLoading(true);
    setMessage(null);
    try {
      const data = await getGraphFeedbackTuningPreview();
      setPreview(data);
    } catch (err) {
      setMessage(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleApply() {
    setApplying(true);
    setMessage(null);
    try {
      const result = await applyGraphFeedbackTuning();
      setMessage(`Applied ${result.graduated_count} concept boost(s).`);
      setPreview(null);
    } catch (err) {
      setMessage(String(err));
    } finally {
      setApplying(false);
    }
  }

  const boostEntries = preview
    ? Object.entries(preview.concept_boosts).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];

  return (
    <div className="feedback-tuning-section">
      <div className="feedback-section-title">Auto-Tuning</div>
      <div className="feedback-tuning-actions">
        <button
          className="feedback-refresh-btn"
          onClick={() => void loadPreview()}
          disabled={loading}
        >
          {loading ? "Loading..." : "Preview Tuning"}
        </button>
        {preview && boostEntries.length > 0 && (
          <button
            className="feedback-copy-btn"
            onClick={() => void handleApply()}
            disabled={applying}
          >
            {applying ? "Applying..." : "Apply Tuning"}
          </button>
        )}
      </div>
      {message && <div className="feedback-tuning-message">{message}</div>}
      {preview && boostEntries.length === 0 && (
        <div className="feedback-empty">No concepts ready for graduation yet.</div>
      )}
      {preview && boostEntries.length > 0 && (
        <table className="feedback-tuning-table">
          <thead>
            <tr>
              <th>Concept</th>
              <th>Feedback Weight</th>
              <th>Proposed Boost</th>
            </tr>
          </thead>
          <tbody>
            {boostEntries.map(([term, boost]) => (
              <tr key={term}>
                <td>{term}</td>
                <td>{(boost / 0.25).toFixed(2)}</td>
                <td>{boost.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function FeedbackReviewPanel({
  visible,
  onClose,
}: FeedbackReviewPanelProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<FeedbackReviewData | null>(null);
  const [health, setHealth] = useState<FeedbackHealthData | null>(null);

  async function loadReview() {
    setLoading(true);
    setError(null);
    try {
      const [review, healthData] = await Promise.all([
        getGraphFeedbackReview(20, 60),
        getGraphFeedbackHealth(),
      ]);
      setData(review);
      setHealth(healthData);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (visible) {
      void loadReview();
    }
  }, [visible]);

  if (!visible) return null;

  return (
    <div className="feedback-panel-overlay" onClick={onClose}>
      <div className="feedback-panel" onClick={(e) => e.stopPropagation()}>
        <div className="feedback-panel-header">
          <div>
            <div className="feedback-panel-title">Feedback Review</div>
            <div className="feedback-panel-subtitle">
              Review off-target notes and generated prompts for Codex/Claude updates.
            </div>
          </div>
          <div className="feedback-panel-header-actions">
            <button className="feedback-refresh-btn" onClick={() => void loadReview()}>
              Refresh
            </button>
            <button className="graph-panel-close" onClick={onClose}>
              {"\u2715"}
            </button>
          </div>
        </div>

        <div className="feedback-panel-body">
          {loading && <div className="graph-panel-loading">Loading feedback review...</div>}
          {error && <div className="feedback-error">{error}</div>}

          {!loading && !error && data && (
            <>
              {health && <HealthBar health={health} />}

              <div className="feedback-summary-grid">
                <div className="feedback-summary-item">
                  <div className="feedback-summary-value">{data.summary.total_events}</div>
                  <div className="feedback-summary-label">Total feedback events</div>
                </div>
                <div className="feedback-summary-item">
                  <div className="feedback-summary-value">{data.summary.helpful_count}</div>
                  <div className="feedback-summary-label">Helpful clicks</div>
                </div>
                <div className="feedback-summary-item">
                  <div className="feedback-summary-value">{data.summary.off_target_count}</div>
                  <div className="feedback-summary-label">Off-target clicks</div>
                </div>
                <div className="feedback-summary-item">
                  <div className="feedback-summary-value">{data.summary.off_target_with_notes}</div>
                  <div className="feedback-summary-label">Off-target notes</div>
                </div>
              </div>

              <TuningSection />

              <div className="feedback-section-title">Prompt Backlog</div>
              {data.prompt_backlog.length === 0 ? (
                <div className="feedback-empty">No prompt backlog yet. Add off-target notes first.</div>
              ) : (
                <div className="feedback-cards">
                  {data.prompt_backlog.map((item) => (
                    <PromptCard item={item} key={item.id} />
                  ))}
                </div>
              )}

              <div className="feedback-section-title">Recent Off-target Notes</div>
              {data.recent_off_target.length === 0 ? (
                <div className="feedback-empty">No off-target notes submitted yet.</div>
              ) : (
                <div className="feedback-events">
                  {data.recent_off_target.slice(0, 20).map((event) => (
                    <div className="feedback-event" key={`event-${event.id}-${event.timestamp}`}>
                      <div className="feedback-event-meta">
                        {formatTimestamp(event.timestamp)}
                      </div>
                      {event.query && <div className="feedback-event-query">Q: {event.query}</div>}
                      <div className="feedback-event-note">{event.note}</div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
