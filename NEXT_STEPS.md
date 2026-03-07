# Next Steps

## Completed (March 2026)

### Claude Provider Integration
- Added `ClaudeEntrySummaryProvider` and `ClaudeStateLabelProvider` with single-call architecture
- Provider registry auto-resolves: prefers Claude when `ANTHROPIC_API_KEY` is set
- Batch processed 718/719 entries with Claude Haiku 4.5 (~50 min, 5 workers)
- Graph enrichment updated to accept Claude prompt versions

### Data Quality Fixes (from Post Batch v4)
- Resource fork entries cleaned (`._` files)
- Dict-format entities fixed via `_sanitize_typed_entities()` and `_extract_string_from_item()`
- Theme near-duplication handled via `_dedupe_themes_by_overlap()` (token overlap dedup)
- State label flatlines addressed with alternate key handling + dimension name normalization
- Entity coercion validator (`coerce_entities`) handles v4 plain strings alongside v6 TypedEntity dicts
- Truncation guards added: entity names (256), signal text (256), signal_id (256)

### Provider Passthrough Fix
- `entry_summary_service.py` now passes `provider` through to `StateLabelGenerateRequest`
- Previously state labels would fall back to Ollama even when Claude was specified

---

## Remaining

### Minor
- [ ] Reprocess 1 remaining Ollama entry (`4-25-2025`) — failed due to entity name >256 chars (now fixed with truncation)
- [ ] Fix `--skip-done` to correctly detect Claude-analyzed entries as complete

### Future
- [ ] Cross-pipeline integration: embeddings informing graph relationships
- [ ] Prompt v2 for Claude providers (fine-tune theme granularity, improve entity dedup)
- [ ] Batch API mode for cheaper processing (50% cost reduction at ~24h latency)
