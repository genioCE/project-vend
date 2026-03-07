#!/usr/bin/env python3
"""Migrate analysis data from v5 to v6 in-place.

Transforms:
  1. Entities: flat strings -> typed objects {"name": str, "type": str}
  2. Themes: tighten to 2-4 words, strip "through X" pattern
  3. Entity normalization: case-fold, merge variants, split compounds

Usage:
  python migrate_v5_to_v6.py /path/to/analysis.sqlite --dry-run
  python migrate_v5_to_v6.py /path/to/analysis.sqlite --with-llm
  python migrate_v5_to_v6.py /path/to/analysis.sqlite --heuristic-only
  python migrate_v5_to_v6.py /path/to/analysis.sqlite --verify-only

Requires: sqlite3 (stdlib). Optional: httpx (pip install httpx) for --with-llm.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_PROMPT_VERSION = "entry-summary-prompt-ollama-v6"

VALID_ENTITY_TYPES = {"person", "place", "organization", "concept", "spiritual"}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://10.0.10.232:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# ---------------------------------------------------------------------------
# Spiritual entities — small closed set
# ---------------------------------------------------------------------------

SPIRITUAL_ENTITIES: set[str] = {
    "god", "higher power", "higher-power", "hp", "the universe",
    "divine", "spirit", "holy spirit", "jesus", "christ",
}

# ---------------------------------------------------------------------------
# Known entities — hand-curated for accuracy
# ---------------------------------------------------------------------------

KNOWN_PERSONS: set[str] = {
    "kelsey", "brian", "kyle", "ruth", "matt", "morgan", "jerry",
    "justice", "tony", "donny", "antonio", "arica", "luke", "amy",
    "adina", "lane", "steven", "gene", "mom", "dad", "dylan",
    "harris", "ana", "johna", "anj", "tiffany", "lindsey", "cam",
    "milan", "butch", "johnny", "vanielle", "van", "danny", "candice",
}

KNOWN_PLACES: set[str] = {
    "hix", "star space", "starspace", "larkspur", "asian district",
    "plaza district", "oklahoma city", "okc", "norman", "tulsa",
    "gym", "home", "church", "park",
}

KNOWN_ORGS: set[str] = {
    "interworks", "aa", "na", "genio", "blocworks", "lyt group",
    "okie doke productions", "anmf",
}

KNOWN_CONCEPTS: set[str] = {
    "writing", "addiction", "recovery", "sobriety", "mindfulness",
    "vulnerability", "creativity", "entrepreneurship", "discipline",
    "loneliness", "anxiety", "depression", "healing", "self-awareness",
    "self-care", "presence", "acceptance", "fear", "shame", "grief",
    "hope", "love", "trust", "faith", "meditation", "exercise",
    "therapy", "journaling", "gratitude", "growth", "rosin",
    "gpt", "chatgpt", "claude", "llm", "ai", "agi",
    "docker", "fastapi", "neo4j", "chromadb", "python", "typescript",
    "semantic search", "vector database", "data engineering",
    "social media", "technology",
}

# ---------------------------------------------------------------------------
# Archetype normalization
# ---------------------------------------------------------------------------

# Archetype Jer variants — keep distinct, classify as person
ARCHETYPE_JER_PATTERN = re.compile(
    r"^(mythic|sovereign|anointed|abundant|universal|genius|billionaire|"
    r"legendary|integrated|mythical|king|warrior|sage|creator)\s+jer(?:ry)?$",
    re.IGNORECASE,
)

# Jerry self-references — normalize to "Jerry"
JERRY_SELF_PATTERN = re.compile(
    r"^jerr?y?\s*\((?:author|the\s+(?:author|writer)|self|writer)\)$",
    re.IGNORECASE,
)
JEREMIAH_PATTERN = re.compile(
    r"^jeremiah\s*\(.*\)$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Sentence-entity splitting
# ---------------------------------------------------------------------------

PAREN_LIST_PATTERN = re.compile(
    r"^(.+?)\s*\(([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*(?:,?\s*(?:and\s+)?[A-Z][a-z]+)?)\)$"
)

# ---------------------------------------------------------------------------
# Theme transformation patterns
# ---------------------------------------------------------------------------

THEME_THROUGH_PATTERN = re.compile(r"^(.+?)\s+through\s+.+$", re.IGNORECASE)
THEME_VERSUS_PATTERN = re.compile(r"\bversus\b", re.IGNORECASE)
THEME_AS_A_PATTERN = re.compile(r"\bas\s+a\b", re.IGNORECASE)
# Trailing prepositional phrases after 4+ words
THEME_TRAILING_PREP = re.compile(
    r"^(.{10,})\s+(?:in|for|of|with|from|after|during|about|between)\s+.+$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Entity type classification
# ---------------------------------------------------------------------------


def classify_entity_heuristic(name: str) -> str | None:
    """Classify an entity by heuristic rules. Returns type or None if ambiguous."""
    lower = name.lower().strip()

    # Spiritual — closed set
    if lower in SPIRITUAL_ENTITIES:
        return "spiritual"

    # Known lookups
    if lower in KNOWN_PERSONS:
        return "person"
    if lower in KNOWN_PLACES:
        return "place"
    if lower in KNOWN_ORGS:
        return "organization"
    if lower in KNOWN_CONCEPTS:
        return "concept"

    # Archetype Jer variants
    if ARCHETYPE_JER_PATTERN.match(name.strip()):
        return "person"

    # Jerry self-references
    if JERRY_SELF_PATTERN.match(name.strip()) or JEREMIAH_PATTERN.match(name.strip()):
        return "person"

    # All lowercase single word → likely concept
    words = name.split()
    if len(words) == 1 and name == name.lower():
        return "concept"

    # Title-case single word (not a common English word) → likely person
    if len(words) == 1 and name[0].isupper() and name[1:].islower():
        # Exclude common non-person title-case words
        non_person = {
            "Recovery", "Writing", "Addiction", "Love", "God", "Entrepreneurship",
            "Gratitude", "Grief", "Mindfulness", "Rosin", "Technology", "Gym",
            "Plaza", "Universal",
        }
        if name in non_person:
            return None  # Let other rules or LLM handle
        return "person"

    # 2-word title case → likely person name
    if len(words) == 2 and all(w[0].isupper() for w in words if w):
        # Check if it's a known place pattern
        place_indicators = {"district", "city", "street", "park", "church", "house", "space"}
        if words[-1].lower() in place_indicators:
            return "place"
        return "person"

    # All caps short string → likely org or acronym concept
    if name == name.upper() and len(name) <= 6:
        return "concept"

    # Contains place indicators
    place_words = {"church", "park", "gym", "house", "room", "school", "district",
                   "street", "avenue", "city", "space", "building", "center", "library"}
    if any(w.lower() in place_words for w in words):
        return "place"

    return None  # Ambiguous — needs LLM


def classify_entities_llm(
    entities: list[str],
    batch_size: int = 50,
) -> dict[str, str]:
    """Classify ambiguous entities using Ollama in batches."""
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx required for --with-llm. Install: pip install httpx")
        sys.exit(1)

    classifications: dict[str, str] = {}
    batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]

    for batch_idx, batch in enumerate(batches):
        numbered = "\n".join(f"{i+1}. \"{e}\"" for i, e in enumerate(batch))
        prompt = (
            "Classify each entity from a personal journal into exactly one type:\n"
            "- person: a human being (by name, role, or relationship)\n"
            "- place: a physical location, building, area, or venue\n"
            "- organization: a company, group, institution, or community\n"
            "- concept: an idea, tool, project, technology, or abstract noun\n"
            "- spiritual: God, Higher Power, divine entities\n\n"
            "Return strict JSON: {\"classifications\": [{\"name\": \"...\", \"type\": \"...\"}]}\n\n"
            f"Entities:\n{numbered}"
        )

        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": "You classify entities. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1},
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
                parsed = json.loads(content)
                items = parsed.get("classifications", [])
                for item in items:
                    if isinstance(item, dict) and "name" in item and "type" in item:
                        etype = item["type"].lower().strip()
                        if etype in VALID_ENTITY_TYPES:
                            classifications[item["name"]] = etype
                        else:
                            classifications[item["name"]] = "concept"
            print(f"  LLM batch {batch_idx + 1}/{len(batches)}: classified {len(items)} entities")
        except Exception as e:
            print(f"  LLM batch {batch_idx + 1}/{len(batches)} FAILED: {e}")
            # Fallback: classify as concept
            for ent in batch:
                if ent not in classifications:
                    classifications[ent] = "concept"

    return classifications


# ---------------------------------------------------------------------------
# Entity normalization
# ---------------------------------------------------------------------------


def build_canonical_map(
    entity_counts: Counter[str],
    type_map: dict[str, str],
) -> dict[str, str]:
    """Build normalized_lower -> canonical_name mapping."""
    # Group entities by lowercase form
    groups: dict[str, list[tuple[str, int]]] = {}
    for entity, count in entity_counts.items():
        key = entity.lower().strip()
        if key not in groups:
            groups[key] = []
        groups[key].append((entity, count))

    canonical: dict[str, str] = {}
    for lower_key, variants in groups.items():
        etype = type_map.get(variants[0][0], "concept")

        if etype == "spiritual":
            # Canonical form for spiritual entities
            if "god" in lower_key:
                canonical[lower_key] = "God"
            elif "higher power" in lower_key:
                canonical[lower_key] = "Higher Power"
            else:
                # Pick highest frequency
                best = max(variants, key=lambda x: x[1])
                canonical[lower_key] = best[0]
        elif etype == "person":
            # Title case for persons
            best = max(variants, key=lambda x: x[1])
            canonical[lower_key] = best[0]  # Keep the most common form
        elif etype == "place":
            best = max(variants, key=lambda x: x[1])
            canonical[lower_key] = best[0]
        elif etype == "organization":
            best = max(variants, key=lambda x: x[1])
            canonical[lower_key] = best[0]
        else:
            # Concepts: prefer lowercase
            if len(variants) == 1:
                canonical[lower_key] = variants[0][0]
            else:
                # If there's a lowercase variant, prefer it
                lower_variants = [v for v in variants if v[0] == v[0].lower()]
                if lower_variants:
                    canonical[lower_key] = max(lower_variants, key=lambda x: x[1])[0]
                else:
                    canonical[lower_key] = max(variants, key=lambda x: x[1])[0]

    return canonical


def normalize_entity(
    entity: str,
    type_map: dict[str, str],
    canonical_map: dict[str, str],
) -> list[dict[str, str]]:
    """Normalize a single entity. May return multiple entities (splitting compounds)."""
    name = entity.strip()
    if not name:
        return []

    # Jerry self-references → "Jerry" (person)
    if JERRY_SELF_PATTERN.match(name) or JEREMIAH_PATTERN.match(name):
        return [{"name": "Jerry", "type": "person"}]

    # Sentence-length entities with parenthetical lists → split
    m = PAREN_LIST_PATTERN.match(name)
    if m and len(name.split()) > 5:
        group_name = m.group(1).strip()
        members = [n.strip() for n in re.split(r",\s*(?:and\s+)?", m.group(2))]
        result = [{"name": group_name, "type": type_map.get(group_name, "concept")}]
        for member in members:
            if member:
                mtype = type_map.get(member, classify_entity_heuristic(member) or "person")
                canon = canonical_map.get(member.lower().strip(), member)
                result.append({"name": canon, "type": mtype})
        return result

    # If still >5 words and no pattern match, truncate to meaningful part
    if len(name.split()) > 5:
        # Take first 4 words as the entity name
        truncated = " ".join(name.split()[:4])
        etype = type_map.get(name, type_map.get(truncated, "concept"))
        return [{"name": truncated, "type": etype}]

    # Standard normalization
    lower = name.lower().strip()
    canon = canonical_map.get(lower, name)
    etype = type_map.get(name, type_map.get(canon, "concept"))

    return [{"name": canon, "type": etype}]


# ---------------------------------------------------------------------------
# Theme transformation
# ---------------------------------------------------------------------------


def tighten_theme(theme: str) -> str:
    """Apply heuristic rules to tighten a theme to 2-4 words."""
    t = theme.strip()
    if not t:
        return t

    # Strip "through X" suffix
    m = THEME_THROUGH_PATTERN.match(t)
    if m:
        t = m.group(1).strip()

    # Compress "versus" → "vs"
    t = THEME_VERSUS_PATTERN.sub("vs", t)

    # Compress "as a" → "as" when it makes the theme long
    words = t.split()
    if len(words) > 4:
        t = THEME_AS_A_PATTERN.sub("as", t)
        words = t.split()

    # Strip trailing prepositional phrases if still >4 words
    if len(words) > 4:
        m = THEME_TRAILING_PREP.match(t)
        if m:
            candidate = m.group(1).strip()
            if len(candidate.split()) >= 2:
                t = candidate
                words = t.split()

    # Hard truncate to 4 words
    if len(words) > 4:
        t = " ".join(words[:4])

    return t


def tighten_themes_llm(themes: list[str], batch_size: int = 80) -> dict[str, str]:
    """Use Ollama to rewrite themes that resist heuristic tightening."""
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx required for --with-llm. Install: pip install httpx")
        sys.exit(1)

    rewrites: dict[str, str] = {}
    batches = [themes[i:i + batch_size] for i in range(0, len(themes), batch_size)]

    for batch_idx, batch in enumerate(batches):
        numbered = "\n".join(f"{i+1}. \"{t}\"" for i, t in enumerate(batch))
        prompt = (
            "Rewrite each journal theme to be 2-4 words. Remove filler words. "
            "Keep the psychological core meaning. Do NOT use 'through' as a connector.\n\n"
            "Return strict JSON: {\"rewrites\": [{\"original\": \"...\", \"rewritten\": \"...\"}]}\n\n"
            f"Themes:\n{numbered}"
        )

        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": "You rewrite themes concisely. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1},
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
                parsed = json.loads(content)
                items = parsed.get("rewrites", [])
                for item in items:
                    if isinstance(item, dict) and "original" in item and "rewritten" in item:
                        rewritten = item["rewritten"].strip()
                        if rewritten and len(rewritten.split()) <= 4:
                            rewrites[item["original"]] = rewritten
            print(f"  LLM theme batch {batch_idx + 1}/{len(batches)}: rewrote {len(items)} themes")
        except Exception as e:
            print(f"  LLM theme batch {batch_idx + 1}/{len(batches)} FAILED: {e}")

    return rewrites


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------


def load_entries(db_path: str) -> list[tuple[str, dict[str, Any], str]]:
    """Load all entries from SQLite. Returns (entry_id, payload_dict, prompt_version)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT entry_id, payload_json, prompt_version FROM entry_summaries")
    rows = cur.fetchall()
    conn.close()

    entries = []
    for row in rows:
        entry_id = row[0]
        payload = json.loads(row[1])
        prompt_version = row[2]
        entries.append((entry_id, payload, prompt_version))
    return entries


def extract_unique_entities(entries: list[tuple[str, dict, str]]) -> Counter[str]:
    """Extract all unique entities with frequency counts."""
    counter: Counter[str] = Counter()
    for _, payload, _ in entries:
        for entity in payload.get("entities", []):
            if isinstance(entity, str) and entity.strip():
                counter[entity.strip()] += 1
    return counter


def extract_all_themes(entries: list[tuple[str, dict, str]]) -> Counter[str]:
    """Extract all unique themes with frequency counts."""
    counter: Counter[str] = Counter()
    for _, payload, _ in entries:
        for theme in payload.get("themes", []):
            if isinstance(theme, str) and theme.strip():
                counter[theme.strip()] += 1
    return counter


def build_type_map(
    entity_counts: Counter[str],
    use_llm: bool = False,
) -> dict[str, str]:
    """Build entity -> type mapping using heuristics + optional LLM."""
    type_map: dict[str, str] = {}
    ambiguous: list[str] = []

    for entity in entity_counts:
        etype = classify_entity_heuristic(entity)
        if etype:
            type_map[entity] = etype
        else:
            ambiguous.append(entity)

    print(f"\n  Heuristic classified: {len(type_map)}/{len(entity_counts)}")
    print(f"  Ambiguous (need LLM): {len(ambiguous)}")

    if ambiguous and use_llm:
        print(f"\n  Classifying {len(ambiguous)} ambiguous entities via Ollama...")
        llm_types = classify_entities_llm(ambiguous)
        for entity, etype in llm_types.items():
            type_map[entity] = etype
        print(f"  LLM classified: {len(llm_types)} entities")

    # Default remaining to concept
    for entity in ambiguous:
        if entity not in type_map:
            type_map[entity] = "concept"

    return type_map


def transform_entry(
    payload: dict[str, Any],
    type_map: dict[str, str],
    canonical_map: dict[str, str],
    theme_rewrites: dict[str, str],
) -> dict[str, Any]:
    """Transform a single entry's payload to v6 format."""
    # --- Transform entities ---
    old_entities = payload.get("entities", [])
    new_entities: list[dict[str, str]] = []
    seen_names: set[str] = set()

    for entity in old_entities:
        if isinstance(entity, str):
            results = normalize_entity(entity, type_map, canonical_map)
            for typed_ent in results:
                name_key = typed_ent["name"].lower().strip()
                if name_key not in seen_names and typed_ent["name"].strip():
                    seen_names.add(name_key)
                    new_entities.append(typed_ent)
        elif isinstance(entity, dict):
            # Already typed (shouldn't happen in v5, but be safe)
            name = entity.get("name", "")
            if name and name.lower().strip() not in seen_names:
                seen_names.add(name.lower().strip())
                new_entities.append(entity)

    payload["entities"] = new_entities

    # --- Transform themes ---
    old_themes = payload.get("themes", [])
    new_themes: list[str] = []
    seen_themes: set[str] = set()

    for theme in old_themes:
        if not isinstance(theme, str) or not theme.strip():
            continue

        # Apply heuristic tightening
        tightened = tighten_theme(theme)

        # If still >4 words, check LLM rewrites
        if len(tightened.split()) > 4 and theme in theme_rewrites:
            tightened = theme_rewrites[theme]

        # Final safety: hard truncate
        words = tightened.split()
        if len(words) > 4:
            tightened = " ".join(words[:4])

        # Dedup
        key = tightened.lower().strip()
        if key and key not in seen_themes:
            seen_themes.add(key)
            new_themes.append(tightened)

    payload["themes"] = new_themes

    # --- Bump processing metadata ---
    if "processing" in payload:
        payload["processing"]["prompt_version"] = TARGET_PROMPT_VERSION

    return payload


def write_entries(
    db_path: str,
    transformed: list[tuple[str, dict[str, Any]]],
) -> None:
    """Write transformed entries back to SQLite in a single transaction."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    for entry_id, payload in transformed:
        payload_json = json.dumps(payload, ensure_ascii=False)
        cur.execute(
            "UPDATE entry_summaries SET payload_json = ?, prompt_version = ?, updated_at = ? "
            "WHERE entry_id = ?",
            (payload_json, TARGET_PROMPT_VERSION, now, entry_id),
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_migration(db_path: str) -> bool:
    """Verify all entries are valid v6 format."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT entry_id, payload_json, prompt_version FROM entry_summaries")
    rows = cur.fetchall()
    conn.close()

    errors = 0
    type_counts: Counter[str] = Counter()
    theme_word_counts: Counter[int] = Counter()
    long_entities = 0
    total_entities = 0
    total_themes = 0

    for entry_id, payload_json, prompt_version in rows:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            print(f"  ERROR: Invalid JSON for {entry_id}")
            errors += 1
            continue

        if prompt_version != TARGET_PROMPT_VERSION:
            print(f"  WARNING: {entry_id} still at {prompt_version}")

        # Check entities are typed objects
        for ent in payload.get("entities", []):
            total_entities += 1
            if not isinstance(ent, dict):
                print(f"  ERROR: Non-dict entity in {entry_id}: {ent}")
                errors += 1
                continue
            if "name" not in ent or "type" not in ent:
                print(f"  ERROR: Missing name/type in {entry_id}: {ent}")
                errors += 1
                continue
            if ent["type"] not in VALID_ENTITY_TYPES:
                print(f"  ERROR: Invalid type '{ent['type']}' in {entry_id}")
                errors += 1
            type_counts[ent["type"]] += 1
            if len(ent["name"].split()) > 5:
                long_entities += 1

        # Check themes are tight
        for theme in payload.get("themes", []):
            total_themes += 1
            wc = len(theme.split())
            theme_word_counts[wc] += 1

    # Report
    print(f"\n{'=' * 60}")
    print("VERIFICATION REPORT")
    print(f"{'=' * 60}")
    print(f"\nTotal entries: {len(rows)}")
    print(f"Errors: {errors}")

    print(f"\nEntity type distribution ({total_entities} total):")
    for etype, count in type_counts.most_common():
        print(f"  {etype}: {count} ({100 * count / max(total_entities, 1):.1f}%)")

    print(f"\nLong entities (>5 words): {long_entities}")

    print(f"\nTheme word count distribution ({total_themes} total):")
    over_4 = 0
    for wc in sorted(theme_word_counts):
        count = theme_word_counts[wc]
        pct = 100 * count / max(total_themes, 1)
        print(f"  {wc} words: {count} ({pct:.1f}%)")
        if wc > 4:
            over_4 += count
    print(f"  Themes >4 words: {over_4} ({100 * over_4 / max(total_themes, 1):.1f}%)")

    return errors == 0


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------


def dry_run_report(
    entries: list[tuple[str, dict, str]],
    type_map: dict[str, str],
    canonical_map: dict[str, str],
    theme_rewrites: dict[str, str],
) -> None:
    """Report what the migration would do without modifying anything."""
    entity_counts = extract_unique_entities(entries)
    theme_counts = extract_all_themes(entries)

    # Entity type distribution
    print(f"\n{'=' * 60}")
    print("DRY RUN — Entity Classification")
    print(f"{'=' * 60}")

    type_dist: Counter[str] = Counter()
    for entity in entity_counts:
        etype = type_map.get(entity, "concept")
        type_dist[etype] += entity_counts[entity]

    print(f"\nTotal unique entities: {len(entity_counts)}")
    print(f"Total entity mentions: {sum(entity_counts.values())}")
    print(f"\nType distribution (by mention count):")
    for etype, count in type_dist.most_common():
        print(f"  {etype}: {count}")

    # Sample classifications
    print(f"\nSample person entities:")
    persons = [(e, c) for e, c in entity_counts.most_common() if type_map.get(e) == "person"]
    for e, c in persons[:15]:
        print(f"  {c:3d}x  {e}")

    print(f"\nSample concept entities:")
    concepts = [(e, c) for e, c in entity_counts.most_common() if type_map.get(e) == "concept"]
    for e, c in concepts[:15]:
        print(f"  {c:3d}x  {e}")

    print(f"\nSample spiritual entities:")
    spiritual = [(e, c) for e, c in entity_counts.most_common() if type_map.get(e) == "spiritual"]
    for e, c in spiritual[:10]:
        print(f"  {c:3d}x  {e}")

    # Case normalization examples
    print(f"\nCanonical name examples:")
    shown = 0
    for lower_key, canon in sorted(canonical_map.items()):
        # Show ones where canonical differs from some variant
        variants = [e for e in entity_counts if e.lower().strip() == lower_key and e != canon]
        if variants:
            print(f"  {variants} → \"{canon}\"")
            shown += 1
            if shown >= 15:
                break

    # Theme transformation
    print(f"\n{'=' * 60}")
    print("DRY RUN — Theme Transformation")
    print(f"{'=' * 60}")

    transformed_count = 0
    still_long = 0
    sample_transforms: list[tuple[str, str]] = []

    for theme in theme_counts:
        tightened = tighten_theme(theme)
        if len(tightened.split()) > 4 and theme in theme_rewrites:
            tightened = theme_rewrites[theme]
        if len(tightened.split()) > 4:
            tightened = " ".join(tightened.split()[:4])
            still_long += 1
        if tightened != theme:
            transformed_count += 1
            if len(sample_transforms) < 20:
                sample_transforms.append((theme, tightened))

    print(f"\nTotal unique themes: {len(theme_counts)}")
    print(f"Would transform: {transformed_count}")
    print(f"Still >4 words after transform: {still_long}")

    print(f"\nSample transformations:")
    for original, tightened in sample_transforms:
        print(f"  \"{original}\"")
        print(f"    → \"{tightened}\"")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Migrate analysis data from v5 to v6")
    parser.add_argument("db_path", help="Path to analysis.sqlite")
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying")
    parser.add_argument("--with-llm", action="store_true", help="Use Ollama for ambiguous entities/themes")
    parser.add_argument("--heuristic-only", action="store_true", help="Skip LLM, use heuristics only")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    args = parser.parse_args()

    db_path = args.db_path
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    # Verify-only mode
    if args.verify_only:
        print(f"Verifying {db_path}...")
        ok = verify_migration(db_path)
        sys.exit(0 if ok else 1)

    # Load entries
    print(f"Loading entries from {db_path}...")
    entries = load_entries(db_path)
    print(f"Loaded {len(entries)} entries")

    # Extract unique entities
    print("\nExtracting unique entities...")
    entity_counts = extract_unique_entities(entries)
    print(f"Found {len(entity_counts)} unique entities ({sum(entity_counts.values())} mentions)")

    # Classify entity types
    print("\nClassifying entity types...")
    use_llm = args.with_llm and not args.heuristic_only
    type_map = build_type_map(entity_counts, use_llm=use_llm)

    # Build canonical name mapping
    print("\nBuilding canonical name mapping...")
    canonical_map = build_canonical_map(entity_counts, type_map)
    collisions = sum(1 for k, v in canonical_map.items() if any(
        e != v for e in entity_counts if e.lower().strip() == k
    ))
    print(f"  {collisions} case collisions resolved")

    # Theme transformation
    print("\nPreparing theme transformations...")
    theme_counts = extract_all_themes(entries)

    # Find themes that resist heuristic tightening
    stubborn_themes: list[str] = []
    for theme in theme_counts:
        tightened = tighten_theme(theme)
        if len(tightened.split()) > 4:
            stubborn_themes.append(theme)

    print(f"  {len(stubborn_themes)} themes need LLM rewrite after heuristics")

    theme_rewrites: dict[str, str] = {}
    if stubborn_themes and use_llm:
        print(f"\n  Rewriting {len(stubborn_themes)} themes via Ollama...")
        theme_rewrites = tighten_themes_llm(stubborn_themes)
        print(f"  LLM rewrote: {len(theme_rewrites)} themes")

    # Dry-run mode
    if args.dry_run:
        dry_run_report(entries, type_map, canonical_map, theme_rewrites)
        print(f"\n[DRY RUN] No changes written.")
        return

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.v5-backup-{timestamp}"
    print(f"\nBacking up to {backup_path}...")
    shutil.copy2(db_path, backup_path)

    # Transform all entries
    print(f"\nTransforming {len(entries)} entries...")
    transformed: list[tuple[str, dict[str, Any]]] = []
    for entry_id, payload, _ in entries:
        new_payload = transform_entry(payload, type_map, canonical_map, theme_rewrites)
        transformed.append((entry_id, new_payload))

    # Write back
    print(f"Writing {len(transformed)} entries to database...")
    write_entries(db_path, transformed)
    print("Write complete.")

    # Verify
    print(f"\nVerifying migration...")
    ok = verify_migration(db_path)
    if ok:
        print("\nMigration successful!")
    else:
        print(f"\nMigration completed with errors. Backup at: {backup_path}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
