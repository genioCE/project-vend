# V6 Prompt: Typed Entities + Tighter Themes

Two changes bundled into prompt version `entry-summary-prompt-ollama-v6`.

## Change 1: Typed Entities

### What
Entities go from flat strings to typed objects:
```json
// Before (v5)
"entities": ["Kelsey", "Hix", "Interworks", "GPT"]

// After (v6)
"entities": [
  {"name": "Kelsey", "type": "person"},
  {"name": "Hix", "type": "place"},
  {"name": "Interworks", "type": "organization"},
  {"name": "GPT", "type": "concept"}
]
```

Four types: `person`, `place`, `organization`, `concept`

### Files to change

**`analysis-service/app/entry_summary_provider.py`**
- Bump `prompt_version` → `"entry-summary-prompt-ollama-v6"`
- Update `_FULL_SYSTEM_PROMPT`: Replace flat entity instruction with typed object instruction + examples
- Update `_CHUNK_EXTRACT_SYSTEM_PROMPT`: Same typed entity instruction
- Update example JSON in prompt to show typed entity format
- Add `_sanitize_typed_entities()` function: parse list of `{"name": ..., "type": ...}` dicts, validate type is one of 4, fallback to "concept" for unknown types, handle LLM returning flat strings (wrap as `{"name": str, "type": "concept"}`)
- Update `_parse_full_response()`: use `_sanitize_typed_entities()` for entities instead of `_sanitize_list()`
- Update `_generate_chunked()`: same change for chunk merge
- Update `EntrySummaryGeneration` dataclass: `entities: list[str]` → `entities: list[dict]`

**`analysis-service/app/models.py`**
- Add `TypedEntity` model: `name: str`, `type: str` (enum: person/place/organization/concept)
- Change `EntrySummaryRecord.entities` from `list[str]` to `list[TypedEntity]`

**`analysis-service/app/batch_analyze.py`**
- Update `_is_entry_done()`: replace dict-format heuristic check with prompt version check (v6). The old `"'" in entity` check no longer makes sense since entities are now dicts by design.

**`mcp-server/src/analysis-client.ts`**
- Add `TypedEntity` interface: `{ name: string; type: string }`
- Change `EntrySummaryRecord.entities` from `string[]` to `TypedEntity[]`

**`mcp-server/src/tools.ts`**
- Update entity display: `- ${e}` → `- ${e.name} (${e.type})`

## Change 2: Tighter Themes

### What
Themes go from 3-6 words to 2-4 words. Kill the "through X" suffix pattern.

### Files to change

**`analysis-service/app/entry_summary_provider.py`** (same file as above)
- Update `_FULL_SYSTEM_PROMPT` theme instruction:
  - "2-4 words" instead of "3-6 words"
  - Add explicit bad examples: "reclaiming agency through creative work" (too long, drop the 'through X')
  - Good examples: "reclaiming agency", "solitude versus connection", "embodied presence", "processing grief"
- Update `_CHUNK_EXTRACT_SYSTEM_PROMPT`: same tighter theme instruction
- Update example JSON themes to be 2-4 words

## Execution Plan

1. Wait for v5 batch to finish (currently ~250/692)
2. Implement all changes above
3. Rsync to Star Space, rebuild Docker
4. Run `--skip-done --workers 6` (all entries will reprocess since v6 ≠ v5)
5. Validate with SQL queries while running
