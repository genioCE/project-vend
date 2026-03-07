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


def test_generate_entry_summary_and_fetch_persisted_record():
    entry_id = "entry-persist-001"
    payload = _payload(
        entry_id,
        "I felt heavy in the morning but chose one clear action.",
        "By evening I felt more open and planned tomorrow's next step.",
    )
    payload["force_regenerate"] = True

    generate = client.post("/entry-summary/generate", json=payload)
    assert generate.status_code == 200
    generated = generate.json()

    assert generated["entry_id"] == entry_id
    assert generated["short_summary"]
    assert generated["detailed_summary"]
    assert isinstance(generated["themes"], list)
    assert isinstance(generated["entities"], list)
    assert isinstance(generated["decisions_actions"], list)
    assert generated["processing"]["schema_version"] == "entry-summary-v1"
    assert generated["provenance"]["chunk_ids"] == [
        f"{entry_id}::chunk-001",
        f"{entry_id}::chunk-002",
    ]

    fetched = client.get(f"/entry-summary/{entry_id}")
    assert fetched.status_code == 200
    assert fetched.json() == generated


def test_generate_without_force_returns_existing_record():
    entry_id = "entry-persist-002"
    first_payload = _payload(
        entry_id,
        "I am stuck and uncertain, replaying what happened yesterday.",
        "I could maybe choose one next step tomorrow.",
    )
    first_payload["force_regenerate"] = True

    first = client.post("/entry-summary/generate", json=first_payload)
    assert first.status_code == 200
    first_json = first.json()

    second_payload = _payload(
        entry_id,
        "This completely different text should not replace existing summary.",
        "No force flag means the stored record is returned.",
    )
    second_payload["force_regenerate"] = False

    second = client.post("/entry-summary/generate", json=second_payload)
    assert second.status_code == 200

    assert second.json() == first_json


def test_force_regenerate_replaces_record_content():
    entry_id = "entry-persist-003"
    first_payload = _payload(
        entry_id,
        "I felt fragmented and confused.",
        "I stayed guarded and replayed old regrets.",
    )
    first_payload["force_regenerate"] = True

    first = client.post("/entry-summary/generate", json=first_payload)
    assert first.status_code == 200
    first_json = first.json()

    second_payload = _payload(
        entry_id,
        "I feel coherent and grounded now.",
        "I decided to build tomorrow with trusted support.",
    )
    second_payload["force_regenerate"] = True

    second = client.post("/entry-summary/generate", json=second_payload)
    assert second.status_code == 200
    second_json = second.json()

    assert second_json["short_summary"] != first_json["short_summary"]
    assert second_json["detailed_summary"] != first_json["detailed_summary"]
    assert second_json["processing"]["created_at"] != first_json["processing"]["created_at"]
