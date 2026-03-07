import {
  searchWritings,
  getEntriesByDate,
  findRecurringThemes,
  getWritingStats,
  getRecentEntries,
  searchByKeyword,
} from "../embeddings-client.js";

import {
  graphSearch,
  findConnectedConcepts,
  getConceptFlows,
  findEntityRelationships,
  traceConceptEvolution,
  comparePeriods as graphComparePeriods,
  getDecisionContext,
  getArchetypePatterns,
  getThemeNetwork,
  getEntriesByState,
} from "../graph-client.js";

import {
  getEntrySummary,
  getEntrySummaries,
  type EntrySummaryRecord,
} from "../analysis-client.js";

import type { LLMMessage, ToolDef } from "../llm/index.js";

// --- Config ---

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

const MAX_HISTORY_MESSAGES = parsePositiveInt(
  process.env.MAX_HISTORY_MESSAGES,
  12
);
const MAX_HISTORY_CHARS = parsePositiveInt(process.env.MAX_HISTORY_CHARS, 12000);
const MAX_TOOL_RESULT_CHARS = parsePositiveInt(
  process.env.MAX_TOOL_RESULT_CHARS,
  4000
);
const MAX_PASSAGE_SNIPPET_CHARS = parsePositiveInt(
  process.env.MAX_PASSAGE_SNIPPET_CHARS,
  650
);
const MAX_ENTRY_SNIPPET_CHARS = parsePositiveInt(
  process.env.MAX_ENTRY_SNIPPET_CHARS,
  900
);
const MAX_ENRICHED_CONTEXT_CHARS = parsePositiveInt(
  process.env.MAX_ENRICHED_CONTEXT_CHARS,
  3500
);

// --- Tool result cache (LRU with TTL) ---

const TOOL_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
const TOOL_CACHE_MAX_SIZE = 100;
const UNCACHEABLE_TOOLS = new Set(["get_entries_by_date", "get_recent_entries"]);

const toolCache = new Map<string, { result: string; expires: number }>();

function toolCacheKey(name: string, args: Record<string, unknown>): string {
  return `${name}:${JSON.stringify(args)}`;
}

function toolCacheGet(key: string): string | undefined {
  const entry = toolCache.get(key);
  if (!entry) return undefined;
  if (Date.now() > entry.expires) {
    toolCache.delete(key);
    return undefined;
  }
  // Move to end (most recently used)
  toolCache.delete(key);
  toolCache.set(key, entry);
  return entry.result;
}

function toolCacheSet(key: string, result: string): void {
  toolCache.delete(key);
  if (toolCache.size >= TOOL_CACHE_MAX_SIZE) {
    const firstKey = toolCache.keys().next().value!;
    toolCache.delete(firstKey);
  }
  toolCache.set(key, { result, expires: Date.now() + TOOL_CACHE_TTL_MS });
}

// --- Tool definitions ---

export const TOOLS: ToolDef[] = [
  {
    type: "function" as const,
    function: {
      name: "search_writings",
      description:
        "Semantic search across the writing corpus. Finds passages most similar in meaning to your query. Use for questions about thoughts, feelings, experiences, or topics.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Natural language search query",
          },
          top_k: {
            type: "number",
            description: "Number of results to return (default 5)",
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_entries_by_date",
      description:
        "Retrieve all writing entries within a date range, sorted chronologically. Use when asked about a specific time period.",
      parameters: {
        type: "object",
        properties: {
          start_date: {
            type: "string",
            description: "Start date in YYYY-MM-DD format",
          },
          end_date: {
            type: "string",
            description: "End date in YYYY-MM-DD format",
          },
        },
        required: ["start_date", "end_date"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "find_recurring_themes",
      description:
        "Find how a topic or theme evolves over time across the writing corpus. Returns related passages sorted chronologically.",
      parameters: {
        type: "object",
        properties: {
          topic: {
            type: "string",
            description: "The theme or topic to trace",
          },
          top_k: {
            type: "number",
            description: "Number of passages to return (default 10)",
          },
        },
        required: ["topic"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_writing_stats",
      description:
        "Get statistics about the writing corpus: total word count, date range, number of entries, average words per entry.",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_recent_entries",
      description:
        "Get the most recent writing entries. Returns the last N entries sorted by date (newest first).",
      parameters: {
        type: "object",
        properties: {
          n: {
            type: "number",
            description: "Number of recent entries to return (default 7)",
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "search_by_keyword",
      description:
        "Exact text search across the writing corpus. Finds literal keyword matches (case-insensitive) with surrounding context.",
      parameters: {
        type: "object",
        properties: {
          keyword: {
            type: "string",
            description: "The exact word or phrase to search for",
          },
          context_words: {
            type: "number",
            description: "Words of surrounding context (default 100)",
          },
        },
        required: ["keyword"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_entry_analysis",
      description:
        "Get pre-computed analysis for a specific journal entry: summary, themes, entities, decisions, and full 8-dimension psychological state profile. Use after finding relevant entries to get deeper insight.",
      parameters: {
        type: "object",
        properties: {
          entry_id: {
            type: "string",
            description: "The entry ID (filename stem, e.g. '1-1-2025' or '08-18-2025 10-15 Expression @ Hix')",
          },
        },
        required: ["entry_id"],
      },
    },
  },
];

export const GRAPH_TOOLS: ToolDef[] = [
  {
    type: "function" as const,
    function: {
      name: "find_connected_concepts",
      description:
        "Find concepts, people, emotions, and entries connected to a given concept in the knowledge graph. Use to explore how ideas relate to each other.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "The concept name to look up",
          },
          limit: {
            type: "number",
            description: "Max connected nodes to return (default 30)",
          },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "trace_concept_evolution",
      description:
        "Trace how a concept appears over time — which entries mention it, what emotions and people co-occur. Shows the evolution of an idea.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "The concept to trace over time",
          },
          limit: {
            type: "number",
            description: "Max entries to return (default 20)",
          },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_concept_flows",
      description:
        "Find directed concept transitions (X -> Y) for a concept, including movement patterns like fear to action.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "The concept to inspect for incoming/outgoing flow edges",
          },
          limit: {
            type: "number",
            description: "Max flow transitions to return (default 20)",
          },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "find_entity_relationships",
      description:
        "Find all entries mentioning a person, along with the concepts and emotions associated with those entries. Maps a person's presence across the corpus.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "The person's name to look up",
          },
          limit: {
            type: "number",
            description: "Max entries to return (default 30)",
          },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "compare_periods",
      description:
        "Compare two time periods: what concepts, emotions, and archetypes appeared in each. Use for questions like 'how was January different from March?'",
      parameters: {
        type: "object",
        properties: {
          start1: {
            type: "string",
            description: "Start of first period (YYYY-MM-DD)",
          },
          end1: {
            type: "string",
            description: "End of first period (YYYY-MM-DD)",
          },
          start2: {
            type: "string",
            description: "Start of second period (YYYY-MM-DD)",
          },
          end2: {
            type: "string",
            description: "End of second period (YYYY-MM-DD)",
          },
        },
        required: ["start1", "end1", "start2", "end2"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_decision_context",
      description:
        "Find recorded decisions and their surrounding emotional and conceptual context. Optionally filter by keyword.",
      parameters: {
        type: "object",
        properties: {
          keyword: {
            type: "string",
            description: "Optional keyword to filter decisions",
          },
          limit: {
            type: "number",
            description: "Max decisions to return (default 10)",
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_archetype_patterns",
      description:
        "Get the archetypal patterns (Warrior, Sage, Creator, Healer, etc.) found in the writing, their frequency, strength, and associated emotions.",
      parameters: {
        type: "object",
        properties: {
          limit: {
            type: "number",
            description: "Max archetypes to return (default 10)",
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "search_themes",
      description:
        "Find entries linked to a specific theme and see which themes co-occur. Themes are 2-4 word patterns like 'cultivating resilience' or 'seeking clarity'.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "Theme to search for",
          },
          limit: {
            type: "number",
            description: "Max entries to return (default 30)",
          },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "search_by_state",
      description:
        "Find entries filtered by psychological state dimension. Dimensions: valence, activation, agency, certainty, relational_openness, self_trust, time_orientation, integration. Scores range -1 to 1.",
      parameters: {
        type: "object",
        properties: {
          dimension: {
            type: "string",
            description: "State dimension (e.g., 'valence', 'agency', 'self_trust')",
          },
          min_score: {
            type: "number",
            description: "Minimum score (-1 to 1, default -1)",
          },
          max_score: {
            type: "number",
            description: "Maximum score (-1 to 1, default 1)",
          },
          limit: {
            type: "number",
            description: "Max entries to return (default 20)",
          },
        },
        required: ["dimension"],
      },
    },
  },
];

// --- Tool execution helpers ---

export function truncateResult(
  text: string,
  maxChars: number = MAX_TOOL_RESULT_CHARS
): string {
  if (text.length <= maxChars) return text;
  return text.substring(0, maxChars) + "\n\n[...truncated]";
}

function clipSnippet(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return text.substring(0, maxChars) + " ...";
}

export function parseToolArguments(args: unknown): Record<string, unknown> {
  if (typeof args === "string") {
    try {
      const parsed = JSON.parse(args) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return {};
    }
    return {};
  }

  if (args && typeof args === "object" && !Array.isArray(args)) {
    return args as Record<string, unknown>;
  }

  return {};
}

// --- Graph center extraction ---

const GENERIC_GRAPH_CENTER_TOKENS = new Set([
  "a",
  "an",
  "and",
  "about",
  "are",
  "as",
  "by",
  "compare",
  "describe",
  "distinct",
  "examples",
  "for",
  "from",
  "give",
  "how",
  "i",
  "in",
  "is",
  "it",
  "me",
  "my",
  "of",
  "one",
  "or",
  "corpus",
  "entries",
  "entry",
  "sentence",
  "sentences",
  "show",
  "summarize",
  "the",
  "their",
  "three",
  "to",
  "two",
  "what",
  "where",
  "which",
  "with",
  "writing",
  "writings",
  "dates",
  "date",
  "year",
  "years",
]);

function normalizeGraphCenterCandidate(value: string): string | undefined {
  const cleaned = value.trim().replace(/^["'`]+|["'`]+$/g, "").replace(/[.,!?;:]+$/g, "");
  if (!cleaned || cleaned.length > 80) return undefined;

  const words = cleaned.split(/\s+/).filter(Boolean);
  if (words.length === 0) return undefined;

  const usefulWords = words.filter((word) => {
    const lower = word.toLowerCase();
    return !/^\d+$/.test(lower) && !GENERIC_GRAPH_CENTER_TOKENS.has(lower);
  });

  if (usefulWords.length === 0) return undefined;

  // Keep short proper names/concepts intact.
  if (words.length <= 3 && usefulWords.length === words.length) {
    return words.join(" ");
  }

  return usefulWords[0];
}

function pickFirstString(values: unknown): string | undefined {
  if (!Array.isArray(values)) return undefined;
  for (const value of values) {
    if (typeof value === "string") {
      const normalized = normalizeGraphCenterCandidate(value);
      if (normalized) return normalized;
    }
  }
  return undefined;
}

export function pickGraphCenterFromEnriched(
  enriched: Record<string, unknown>
): string | undefined {
  const extracted = enriched.extracted_entities;
  if (!extracted || typeof extracted !== "object" || Array.isArray(extracted)) {
    return undefined;
  }

  const entities = extracted as Record<string, unknown>;
  return (
    pickFirstString(entities.concepts) ||
    pickFirstString(entities.people) ||
    pickFirstString(entities.places)
  );
}

export function pickGraphCenterFromToolCall(
  toolName: string,
  args: Record<string, unknown>
): string | undefined {
  const name = args.name;
  if (typeof name === "string") {
    const normalized = normalizeGraphCenterCandidate(name);
    if (normalized) return normalized;
  }

  if (toolName === "get_decision_context") {
    const keyword = args.keyword;
    if (typeof keyword === "string") {
      const normalized = normalizeGraphCenterCandidate(keyword);
      if (normalized) return normalized;
    }
  }

  return undefined;
}

// --- History windowing ---

export function buildRecentHistory(history: LLMMessage[]): LLMMessage[] {
  const windowed = history.slice(-MAX_HISTORY_MESSAGES);
  const selected: LLMMessage[] = [];
  let totalChars = 0;

  for (let i = windowed.length - 1; i >= 0; i--) {
    const msg = windowed[i];
    const msgChars = msg.content.length;
    if (selected.length > 0 && totalChars + msgChars > MAX_HISTORY_CHARS) {
      break;
    }
    selected.push(msg);
    totalChars += msgChars;
  }

  return selected.reverse();
}

// --- GraphRAG pre-search ---

export { graphSearch };

export { MAX_ENRICHED_CONTEXT_CHARS };

// --- Tool execution ---

export interface ToolResult {
  result: string;
  duration_ms: number;
  cache_hit: boolean;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new Error("request aborted");
  }
}

export async function executeTool(
  name: string,
  args: Record<string, unknown>,
  signal?: AbortSignal,
  requestId?: string
): Promise<ToolResult> {
  throwIfAborted(signal);

  // Check cache for cacheable tools
  if (!UNCACHEABLE_TOOLS.has(name)) {
    const cKey = toolCacheKey(name, args);
    const cached = toolCacheGet(cKey);
    if (cached !== undefined) {
      return { result: cached, duration_ms: 0, cache_hit: true };
    }
  }

  const start = performance.now();
  const result = await executeToolUncached(name, args, signal, requestId);
  const duration_ms = performance.now() - start;

  // Store in cache
  if (!UNCACHEABLE_TOOLS.has(name)) {
    toolCacheSet(toolCacheKey(name, args), result);
  }

  return { result, duration_ms, cache_hit: false };
}

function formatAnalysisCompact(record: EntrySummaryRecord): string {
  const dims = record.state_profile.dimensions
    .map((d) => ({ ...d, abs: Math.abs(d.score) }))
    .sort((a, b) => b.abs - a.abs)
    .slice(0, 3);
  const stateStr = dims
    .map((d) => `${d.dimension}:${d.label}(${d.score.toFixed(1)})`)
    .join(", ");
  const summary = record.short_summary.length > 120
    ? record.short_summary.substring(0, 120) + "..."
    : record.short_summary;
  return `[Summary] ${summary}\n[State] ${stateStr}`;
}

async function enrichSearchResultsWithAnalysis(
  text: string,
  sourceFiles: string[],
  signal?: AbortSignal,
  requestId?: string
): Promise<string> {
  if (sourceFiles.length === 0) return text;

  const entryIds = sourceFiles.map((f) => f.replace(/\.md$/i, ""));
  let summaries: Map<string, EntrySummaryRecord>;
  try {
    summaries = await getEntrySummaries(entryIds, signal, requestId);
  } catch {
    return text;
  }

  if (summaries.size === 0) return text;

  const blocks = text.split("\n\n---\n\n");
  const enriched = blocks.map((block, i) => {
    const entryId = entryIds[i];
    if (!entryId) return block;
    const record = summaries.get(entryId);
    if (!record) return block;
    return block + "\n" + formatAnalysisCompact(record);
  });

  return enriched.join("\n\n---\n\n");
}

async function executeToolUncached(
  name: string,
  args: Record<string, unknown>,
  signal?: AbortSignal,
  requestId?: string
): Promise<string> {
  throwIfAborted(signal);

  try {
    switch (name) {
      case "search_writings": {
        const result = await searchWritings(
          args.query as string,
          (args.top_k as number) || 5,
          signal,
          requestId
        );
        if (result.results.length === 0) return "No results found.";
        const sourceFiles = result.results.map((r) => r.source_file);
        const text = result.results
          .map(
            (r, i) =>
              `[${i + 1}] Date: ${r.date} (relevance: ${r.relevance_score})\nSource: ${r.source_file}\n${clipSnippet(r.text, MAX_PASSAGE_SNIPPET_CHARS)}`
          )
          .join("\n\n---\n\n");
        const enriched = await enrichSearchResultsWithAnalysis(text, sourceFiles, signal, requestId);
        return truncateResult(enriched);
      }
      case "get_entries_by_date": {
        const result = await getEntriesByDate(
          args.start_date as string,
          args.end_date as string,
          signal,
          requestId
        );
        if (result.entries.length === 0)
          return `No entries found between ${args.start_date} and ${args.end_date}.`;
        const text = result.entries
          .map(
            (e) =>
              `Date: ${e.date} (${e.word_count} words)\n${clipSnippet(e.text, MAX_ENTRY_SNIPPET_CHARS)}`
          )
          .join("\n\n---\n\n");
        return truncateResult(text);
      }
      case "find_recurring_themes": {
        const result = await findRecurringThemes(
          args.topic as string,
          (args.top_k as number) || 10,
          signal,
          requestId
        );
        if (result.results.length === 0)
          return `No passages found for topic: ${args.topic}`;
        const text = result.results
          .map(
            (r, i) =>
              `[${i + 1}] Date: ${r.date} (relevance: ${r.relevance_score})\nSource: ${r.source_file}\n${clipSnippet(r.text, MAX_PASSAGE_SNIPPET_CHARS)}`
          )
          .join("\n\n---\n\n");
        return truncateResult(text);
      }
      case "get_writing_stats": {
        const stats = await getWritingStats(signal, requestId);
        return [
          `Total words: ${stats.total_words}`,
          `Total entries: ${stats.total_entries}`,
          `Total indexed chunks: ${stats.total_chunks}`,
          `Date range: ${stats.date_range.earliest || "N/A"} to ${stats.date_range.latest || "N/A"}`,
          `Avg words/entry: ${stats.avg_words_per_entry}`,
          `Entries per year: ${JSON.stringify(stats.entries_per_year)}`,
        ].join("\n");
      }
      case "get_recent_entries": {
        const result = await getRecentEntries((args.n as number) || 7, signal, requestId);
        if (result.entries.length === 0) return "No entries found.";
        const text = result.entries
          .map(
            (e) =>
              `Date: ${e.date} (${e.word_count} words)\n${clipSnippet(e.text, MAX_ENTRY_SNIPPET_CHARS)}`
          )
          .join("\n\n---\n\n");
        return truncateResult(text);
      }
      case "search_by_keyword": {
        const result = await searchByKeyword(
          args.keyword as string,
          (args.context_words as number) || 100,
          signal,
          requestId
        );
        if (result.results.length === 0)
          return `No matches found for: ${args.keyword}`;
        const text =
          `Found ${result.total_matches} matches:\n\n` +
          result.results
            .map(
              (r, i) =>
                `[${i + 1}] Date: ${r.date}\n${clipSnippet(r.context, MAX_PASSAGE_SNIPPET_CHARS)}`
            )
            .join("\n\n---\n\n");
        return truncateResult(text);
      }

      case "get_entry_analysis": {
        const entryId = args.entry_id as string;
        if (!entryId) return "Missing entry_id parameter.";
        const record = await getEntrySummary(entryId, signal, requestId);
        if (!record) return `No analysis found for entry: ${entryId}`;
        const dims = record.state_profile.dimensions
          .map((d) => `  ${d.dimension}: ${d.label} (${d.score.toFixed(2)}) [${d.low_anchor} <-> ${d.high_anchor}]`)
          .join("\n");
        return [
          `Entry: ${record.entry_id}${record.entry_date ? ` (${record.entry_date})` : ""}`,
          `Summary: ${record.short_summary}`,
          `Detail: ${record.detailed_summary}`,
          `Themes: ${record.themes.join(", ") || "none"}`,
          `Entities: ${record.entities.join(", ") || "none"}`,
          `Decisions: ${record.decisions_actions.join("; ") || "none"}`,
          `State Profile:`,
          dims,
          `Provider: ${record.processing.provider} (mock: ${record.processing.mock})`,
        ].join("\n");
      }

      // --- Graph tools ---

      case "find_connected_concepts": {
        const result = await findConnectedConcepts(
          args.name as string,
          (args.limit as number) || 30,
          signal,
          requestId
        );
        const network = (result as { network?: unknown[] }).network || [];
        if (network.length === 0)
          return `No connections found for concept: ${args.name}`;
        const text = (network as Record<string, unknown>[])
          .map(
            (n) =>
              `[${n.relationship}] ${n.node_type}: ${n.node_name}${n.date ? ` (${n.date})` : ""}`
          )
          .join("\n");
        return truncateResult(text);
      }
      case "trace_concept_evolution": {
        const result = await traceConceptEvolution(
          args.name as string,
          (args.limit as number) || 20,
          signal,
          requestId
        );
        const evolution =
          (result as { evolution?: unknown[] }).evolution || [];
        if (evolution.length === 0)
          return `No evolution data found for: ${args.name}`;
        const text = (evolution as Record<string, unknown>[])
          .map(
            (e) =>
              `${e.date} (${e.word_count} words) — emotions: ${JSON.stringify(e.emotions)}, people: ${JSON.stringify(e.people)}`
          )
          .join("\n");
        return truncateResult(text);
      }
      case "get_concept_flows": {
        const result = await getConceptFlows(
          args.name as string,
          (args.limit as number) || 20,
          signal,
          requestId
        );
        const flows = (result as { flows?: unknown[] }).flows || [];
        if (flows.length === 0)
          return `No concept flow data found for: ${args.name}`;
        const text = (flows as Record<string, unknown>[])
          .map(
            (flow) =>
              `${flow.source} -> ${flow.target} (weight: ${flow.weight}, direction: ${flow.direction})${flow.sample_phrase ? ` | sample: ${flow.sample_phrase}` : ""}`
          )
          .join("\n");
        return truncateResult(text);
      }
      case "find_entity_relationships": {
        const result = await findEntityRelationships(
          args.name as string,
          (args.limit as number) || 30,
          signal,
          requestId
        );
        const network = (result as { network?: unknown[] }).network || [];
        if (network.length === 0)
          return `No entries found mentioning: ${args.name}`;
        const text = (network as Record<string, unknown>[])
          .map(
            (e) =>
              `${e.date} — concepts: ${JSON.stringify(e.concepts)}, emotions: ${JSON.stringify(e.emotions)}`
          )
          .join("\n");
        return truncateResult(text);
      }
      case "compare_periods": {
        const result = await graphComparePeriods(
          args.start1 as string,
          args.end1 as string,
          args.start2 as string,
          args.end2 as string,
          signal,
          requestId
        );
        return truncateResult(JSON.stringify(result, null, 2));
      }
      case "get_decision_context": {
        const result = await getDecisionContext(
          args.keyword as string | undefined,
          (args.limit as number) || 10,
          signal,
          requestId
        );
        const decisions =
          (result as { decisions?: unknown[] }).decisions || [];
        if (decisions.length === 0)
          return args.keyword
            ? `No decisions found matching: ${args.keyword}`
            : "No decisions found.";
        const text = (decisions as Record<string, unknown>[])
          .map(
            (d) =>
              `${d.date}: "${d.decision}" — emotions: ${JSON.stringify(d.emotions)}, concepts: ${JSON.stringify(d.concepts)}`
          )
          .join("\n\n");
        return truncateResult(text);
      }
      case "get_archetype_patterns": {
        const result = await getArchetypePatterns(
          (args.limit as number) || 10,
          signal,
          requestId
        );
        const archetypes =
          (result as { archetypes?: unknown[] }).archetypes || [];
        if (archetypes.length === 0) return "No archetype patterns found.";
        const text = (archetypes as Record<string, unknown>[])
          .map(
            (a) =>
              `${a.archetype} (${a.entry_count} entries, avg strength: ${a.avg_strength}) — emotions: ${JSON.stringify(a.associated_emotions)}`
          )
          .join("\n");
        return truncateResult(text);
      }
      case "search_themes": {
        const result = await getThemeNetwork(
          args.name as string,
          (args.limit as number) || 30,
          signal,
          requestId
        );
        const r = result as {
          entries?: Record<string, unknown>[];
          cooccurrences?: Record<string, unknown>[];
        };
        const entries = r.entries || [];
        const cooccurrences = r.cooccurrences || [];
        const parts: string[] = [];
        if (cooccurrences.length > 0) {
          parts.push(
            "Co-occurring themes: " +
              (cooccurrences as Record<string, unknown>[])
                .map((c) => `${c.theme} (${c.weight})`)
                .join(", ")
          );
        }
        if (entries.length === 0) {
          parts.push(`No entries found for theme: ${args.name}`);
        } else {
          parts.push(
            ...entries.map(
              (e) =>
                `${e.date} — valence: ${e.valence ?? "n/a"}, emotions: ${JSON.stringify(e.emotions)}, co-themes: ${JSON.stringify((e.co_themes as string[])?.slice(0, 5))}`
            )
          );
        }
        return truncateResult(parts.join("\n"));
      }
      case "search_by_state": {
        const result = await getEntriesByState(
          args.dimension as string,
          (args.min_score as number) ?? -1,
          (args.max_score as number) ?? 1,
          (args.limit as number) || 20,
          signal,
          requestId
        );
        const stateEntries =
          (result as { entries?: unknown[] }).entries || [];
        if (stateEntries.length === 0)
          return `No entries found for ${args.dimension} in range.`;
        const text = (stateEntries as Record<string, unknown>[])
          .map(
            (e) =>
              `${e.date} (${args.dimension}: ${e.score}) — ${e.summary || e.filename}\n  themes: ${JSON.stringify(e.themes)}`
          )
          .join("\n\n");
        return truncateResult(text);
      }
      default:
        return `Unknown tool: ${name}`;
    }
  } catch (err) {
    if (signal?.aborted) {
      throw err;
    }
    return `Tool error: ${String(err)}`;
  }
}
