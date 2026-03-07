from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


client = TestClient(app)


def _payload(entry_id: str, text_a: str, text_b: str):
    return {
        "entry_id": entry_id,
        "source_file": f"{entry_id}.md",
        "chunks": [
            {
                "chunk_id": f"{entry_id}::chunk-001",
                "text": text_a,
                "source_file": f"{entry_id}.md",
            },
            {
                "chunk_id": f"{entry_id}::chunk-002",
                "text": text_b,
                "source_file": f"{entry_id}.md",
            },
        ],
    }


def test_generate_state_label_and_fetch():
    entry_id = "state-label-001"
    payload = _payload(
        entry_id,
        "I feel stuck but I am choosing to move forward.",
        "By evening I felt more open and planned tomorrow's next step.",
    )
    payload["force_regenerate"] = True

    generate = client.post("/state-label/generate", json=payload)
    assert generate.status_code == 200
    generated = generate.json()

    assert generated["entry_id"] == entry_id
    assert len(generated["state_profile"]["dimensions"]) == 8
    assert generated["confidence"]["overall"] >= 0.0
    assert generated["confidence"]["overall"] <= 1.0
    assert len(generated["confidence"]["by_dimension"]) == 8
    assert generated["processing"]["schema_version"] == "state-label-v1"
    assert generated["provenance"]["chunk_ids"] == [
        f"{entry_id}::chunk-001",
        f"{entry_id}::chunk-002",
    ]

    fetched = client.get(f"/state-label/{entry_id}")
    assert fetched.status_code == 200
    assert fetched.json() == generated


def test_cached_without_force_regenerate():
    entry_id = "state-label-002"
    first_payload = _payload(
        entry_id,
        "I am stuck and uncertain, replaying what happened yesterday.",
        "I could maybe choose one next step tomorrow.",
    )
    first_payload["force_regenerate"] = True

    first = client.post("/state-label/generate", json=first_payload)
    assert first.status_code == 200
    first_json = first.json()

    second_payload = _payload(
        entry_id,
        "Completely different text should not replace existing state label.",
        "No force flag means the stored record is returned.",
    )
    second_payload["force_regenerate"] = False

    second = client.post("/state-label/generate", json=second_payload)
    assert second.status_code == 200

    assert second.json() == first_json


def test_force_regenerate_replaces():
    entry_id = "state-label-003"
    first_payload = _payload(
        entry_id,
        "I felt fragmented and confused.",
        "I stayed guarded and replayed old regrets.",
    )
    first_payload["force_regenerate"] = True

    first = client.post("/state-label/generate", json=first_payload)
    assert first.status_code == 200
    first_json = first.json()

    second_payload = _payload(
        entry_id,
        "I feel coherent and grounded now.",
        "I decided to build tomorrow with trusted support.",
    )
    second_payload["force_regenerate"] = True

    second = client.post("/state-label/generate", json=second_payload)
    assert second.status_code == 200
    second_json = second.json()

    assert second_json["processing"]["created_at"] != first_json["processing"]["created_at"]


def test_legacy_state_label_endpoint_unchanged():
    response = client.post(
        "/state/label",
        json={
            "entry_id": "legacy-test-001",
            "text": "I feel stuck but I am choosing to move forward.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["entry_id"] == "legacy-test-001"
    assert len(data["state_profile"]["dimensions"]) == 8
    assert "version" in data
