from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["mock_mode"] is True


def test_summarize_entry_is_deterministic():
    request = {
        "entry_id": "entry-001",
        "text": "I felt calm this morning. I planned my week and wrote down next actions. "
        "By lunch I noticed more focus and less noise.",
        "entry_date": "2026-02-20",
        "source_file": "2026-02-20.md",
        "chunk_ids": ["chunk-1", "chunk-2"],
        "max_summary_sentences": 2,
    }

    first = client.post("/summarize/entry", json=request)
    second = client.post("/summarize/entry", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()

    payload = first.json()
    assert payload["entry_id"] == "entry-001"
    assert payload["granularity"] == "entry"
    assert len(payload["key_terms"]) > 0
    assert payload["provenance"]["chunk_ids"] == ["chunk-1", "chunk-2"]


def test_state_label_response_shape():
    request = {
        "entry_id": "entry-002",
        "text": (
            "I feel heavy and stuck, but I can choose next steps. "
            "I am still conflicted, yet more open and moving forward with a plan."
        ),
        "chunk_ids": ["chunk-a"],
    }

    first = client.post("/state/label", json=request)
    second = client.post("/state/label", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()

    payload = first.json()
    assert payload["entry_id"] == "entry-002"
    assert len(payload["state_profile"]["dimensions"]) == 8
    assert payload["state_profile"]["score_range"] == {"min": -1.0, "max": 1.0}
    assert len(payload["observed_text_signals"]) > 0
    assert len(payload["inferred_state_labels"]) == 8
    assert len(payload["confidence"]["by_dimension"]) == 8
    assert 0.0 <= payload["confidence"]["overall"] <= 1.0
    assert payload["version"]["schema_version"] == "state-profile-v1"


def test_context_packet_uses_provided_inputs():
    request = {
        "query": "How does planning relate to calm focus?",
        "date_start": "2026-01-01",
        "date_end": "2026-02-20",
        "retrieval_hits": [
            {
                "chunk_id": "chunk-z",
                "source_file": "2026-02-18.md",
                "excerpt": "I planned two deep work blocks and felt calm.",
                "relevance_score": 0.91,
            }
        ],
        "graph_signals": [
            {
                "subject": "planning",
                "relation": "ASSOCIATED_WITH",
                "object": "focus",
                "weight": 0.88,
            }
        ],
    }

    response = client.post("/context/packet", json=request)
    assert response.status_code == 200

    payload = response.json()
    assert payload["query"] == request["query"]
    assert len(payload["retrieval_context"]) == 1
    assert payload["retrieval_context"][0]["chunk_id"] == "chunk-z"
    assert len(payload["graph_context"]) == 1
    assert payload["provenance"]["chunk_ids"] == ["chunk-z"]
