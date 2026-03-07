import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import {
  searchWritings,
  getEntriesByDate,
  findRecurringThemes,
  getWritingStats,
  getRecentEntries,
  searchByKeyword,
} from "./embeddings-client.js";
import {
  findConnectedConcepts,
  getConceptFlows,
  findEntityRelationships,
  traceConceptEvolution,
  comparePeriods,
  getDecisionContext,
  getArchetypePatterns,
  getThemeNetwork,
  getEntriesByState,
} from "./graph-client.js";
import {
  getEntrySummary,
  getEntrySummaries,
  type EntrySummaryRecord,
} from "./analysis-client.js";

function formatAnalysisFull(record: EntrySummaryRecord): string {
  const dims = record.state_profile.dimensions
    .map((d) => `  ${d.dimension}: ${d.label} (${d.score.toFixed(2)})`)
    .join("\n");
  return [
    `**Summary:** ${record.short_summary}`,
    `**Themes:** ${record.themes.join(", ") || "none"}`,
    `**State Profile:**`,
    dims,
  ].join("\n");
}

interface RegisterOptions {
  only?: string[];
}

export function registerTools(
  server: McpServer,
  opts?: RegisterOptions
): void {
  const allow = opts?.only ? new Set(opts.only) : null;

  // Wrap server to skip tools not in the allow list
  const gatedServer = allow
    ? new Proxy(server, {
        get(target, prop, receiver) {
          if (prop === "tool") {
            return (name: string, ...args: unknown[]) => {
              if (!allow.has(name)) return;
              return (target.tool as Function).call(target, name, ...args);
            };
          }
          return Reflect.get(target, prop, receiver);
        },
      }) as McpServer
    : server;

  // 1. search_writings
  gatedServer.tool(
    "search_writings",
    "Semantic search across the writing corpus. Finds passages most similar in meaning to your query. Use this for open-ended questions about thoughts, feelings, experiences, or topics discussed in the writing.",
    {
      query: z
        .string()
        .describe("Natural language search query — what are you looking for?"),
      top_k: z
        .number()
        .int()
        .min(1)
        .max(50)
        .default(5)
        .describe("Number of results to return (default 5)"),
    },
    async ({ query, top_k }) => {
      try {
        const result = await searchWritings(query, top_k);

        if (result.results.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: `No results found for: "${query}"`,
              },
            ],
          };
        }

        // Best-effort analysis enrichment
        const entryIds = result.results.map((r) => r.source_file.replace(/\.md$/i, ""));
        let summaries = new Map<string, EntrySummaryRecord>();
        try {
          summaries = await getEntrySummaries(entryIds);
        } catch {
          // Analysis service may not be available
        }

        const formatted = result.results
          .map((r, i) => {
            const entryId = r.source_file.replace(/\.md$/i, "");
            const record = summaries.get(entryId);
            const analysisBlock = record ? "\n\n" + formatAnalysisFull(record) : "";
            return [
              `### Result ${i + 1} — ${r.date} (relevance: ${r.relevance_score})`,
              `**Source:** ${r.source_file}`,
              ``,
              r.text,
              analysisBlock,
            ].join("\n");
          })
          .join("\n\n---\n\n");

        return {
          content: [{ type: "text" as const, text: formatted }],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // 2. get_entries_by_date
  gatedServer.tool(
    "get_entries_by_date",
    "Retrieve all writing entries within a date range. Returns full text of each entry, sorted chronologically. Useful for reviewing what was written during a specific period.",
    {
      start_date: z
        .string()
        .describe("Start date in YYYY-MM-DD format"),
      end_date: z
        .string()
        .describe("End date in YYYY-MM-DD format"),
    },
    async ({ start_date, end_date }) => {
      const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
      if (!dateRegex.test(start_date) || !dateRegex.test(end_date)) {
        return {
          content: [
            {
              type: "text" as const,
              text: "Invalid date format. Please use YYYY-MM-DD (e.g., 2025-01-15).",
            },
          ],
          isError: true,
        };
      }

      try {
        const result = await getEntriesByDate(start_date, end_date);

        if (result.entries.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: `No entries found between ${start_date} and ${end_date}.`,
              },
            ],
          };
        }

        const formatted = [
          `## Entries from ${start_date} to ${end_date} (${result.count} entries)`,
          "",
          ...result.entries.map((e) => {
            return [
              `### ${e.date} — ${e.filename} (${e.word_count} words)`,
              "",
              e.text,
            ].join("\n");
          }),
        ].join("\n\n---\n\n");

        return {
          content: [{ type: "text" as const, text: formatted }],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // 3. find_recurring_themes
  gatedServer.tool(
    "find_recurring_themes",
    "Find how a topic or theme evolves over time across the writing corpus. Returns passages related to the topic sorted chronologically, showing how thinking about this subject has changed. Great for tracking personal growth, recurring concerns, or evolving perspectives.",
    {
      topic: z
        .string()
        .describe(
          "The theme or topic to trace through the writing (e.g., 'recovery', 'relationships', 'work')"
        ),
      top_k: z
        .number()
        .int()
        .min(1)
        .max(50)
        .default(10)
        .describe("Number of passages to return (default 10)"),
    },
    async ({ topic, top_k }) => {
      try {
        const result = await findRecurringThemes(topic, top_k);

        if (result.results.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: `No passages found related to: "${topic}"`,
              },
            ],
          };
        }

        const formatted = [
          `## Theme: "${topic}" — Evolution over time (${result.results.length} passages)`,
          "",
          ...result.results.map((r, i) => {
            return [
              `### ${i + 1}. ${r.date} (relevance: ${r.relevance_score})`,
              `**Source:** ${r.source_file}`,
              "",
              r.text,
            ].join("\n");
          }),
        ].join("\n\n---\n\n");

        return {
          content: [{ type: "text" as const, text: formatted }],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // 4. get_writing_stats
  gatedServer.tool(
    "get_writing_stats",
    "Get statistics about the entire writing corpus: total word count, date range, number of entries, average words per entry, and entries per year. Useful for understanding the scope and consistency of the writing practice.",
    {},
    async () => {
      try {
        const stats = await getWritingStats();

        const yearLines = Object.entries(stats.entries_per_year)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([year, count]) => `  ${year}: ${count} chunks`)
          .join("\n");

        const text = [
          `## Writing Corpus Statistics`,
          "",
          `- **Total words:** ${stats.total_words.toLocaleString()}`,
          `- **Total entries:** ${stats.total_entries}`,
          `- **Total indexed chunks:** ${stats.total_chunks}`,
          `- **Date range:** ${stats.date_range.earliest || "N/A"} to ${stats.date_range.latest || "N/A"}`,
          `- **Average words per entry:** ${stats.avg_words_per_entry.toLocaleString()}`,
          "",
          `### Chunks per year`,
          yearLines,
        ].join("\n");

        return {
          content: [{ type: "text" as const, text }],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // 5. get_recent_entries
  gatedServer.tool(
    "get_recent_entries",
    "Get the most recent writing entries in full. Returns the last N entries sorted by date (newest first). Useful for understanding what's been on the writer's mind lately.",
    {
      n: z
        .number()
        .int()
        .min(1)
        .max(30)
        .default(7)
        .describe("Number of recent entries to return (default 7)"),
    },
    async ({ n }) => {
      try {
        const result = await getRecentEntries(n);

        if (result.entries.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: "No entries found in the corpus.",
              },
            ],
          };
        }

        const formatted = [
          `## ${result.count} Most Recent Entries`,
          "",
          ...result.entries.map((e) => {
            return [
              `### ${e.date} — ${e.filename} (${e.word_count} words)`,
              "",
              e.text,
            ].join("\n");
          }),
        ].join("\n\n---\n\n");

        return {
          content: [{ type: "text" as const, text: formatted }],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // 6. search_by_keyword
  gatedServer.tool(
    "search_by_keyword",
    "Exact text search across the writing corpus. Finds literal keyword matches (case-insensitive) and returns the surrounding context. Unlike semantic search, this finds exact words or phrases. Useful for finding specific names, places, or phrases.",
    {
      keyword: z
        .string()
        .describe("The exact word or phrase to search for (case-insensitive)"),
      context_words: z
        .number()
        .int()
        .min(10)
        .max(500)
        .default(100)
        .describe(
          "Number of words of surrounding context to include (default 100)"
        ),
    },
    async ({ keyword, context_words }) => {
      try {
        const result = await searchByKeyword(keyword, context_words);

        if (result.results.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: `No matches found for: "${keyword}"`,
              },
            ],
          };
        }

        const formatted = [
          `## Keyword: "${keyword}" — ${result.total_matches} matches`,
          "",
          ...result.results.map((r, i) => {
            return [
              `### Match ${i + 1} — ${r.date}`,
              `**File:** ${r.filename}`,
              "",
              r.context,
            ].join("\n");
          }),
        ].join("\n\n---\n\n");

        return {
          content: [{ type: "text" as const, text: formatted }],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: String(err) }],
          isError: true,
        };
      }
    }
  );

  // 7. get_entry_analysis
  gatedServer.tool(
    "get_entry_analysis",
    "Get pre-computed analysis for a specific journal entry: summary, themes, entities, decisions, and full 8-dimension psychological state profile (valence, activation, agency, certainty, relational openness, self-trust, time orientation, integration). Use after finding relevant entries to get deeper insight.",
    {
      entry_id: z
        .string()
        .describe("The entry ID (filename stem, e.g. '1-1-2025')"),
    },
    async ({ entry_id }) => {
      try {
        const record = await getEntrySummary(entry_id);
        if (!record) {
          return {
            content: [{ type: "text" as const, text: `No analysis found for entry: "${entry_id}"` }],
          };
        }

        const dims = record.state_profile.dimensions
          .map((d) => `- **${d.dimension}:** ${d.label} (${d.score.toFixed(2)}) — ${d.low_anchor} <-> ${d.high_anchor}`)
          .join("\n");

        const formatted = [
          `## Entry Analysis: ${record.entry_id}${record.entry_date ? ` (${record.entry_date})` : ""}`,
          "",
          `### Summary`,
          record.short_summary,
          "",
          `### Detail`,
          record.detailed_summary,
          "",
          `### Themes`,
          record.themes.length > 0 ? record.themes.map((t) => `- ${t}`).join("\n") : "None detected",
          "",
          `### Entities`,
          record.entities.length > 0 ? record.entities.map((e) => `- ${e.name} (${e.type})`).join("\n") : "None detected",
          "",
          `### Decisions & Actions`,
          record.decisions_actions.length > 0 ? record.decisions_actions.map((d) => `- ${d}`).join("\n") : "None detected",
          "",
          `### State Profile`,
          dims,
          "",
          `*Provider: ${record.processing.provider} | Mock: ${record.processing.mock}*`,
        ].join("\n");

        return { content: [{ type: "text" as const, text: formatted }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // --- Graph tools (require graph-service + neo4j) ---

  // 7. find_connected_concepts
  gatedServer.tool(
    "find_connected_concepts",
    "Find concepts, people, emotions, and entries connected to a given concept in the knowledge graph. Use to explore how ideas relate to each other across the writing.",
    {
      name: z
        .string()
        .describe("The concept name to look up (e.g., 'stillness', 'recovery', 'solitude')"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(100)
        .default(30)
        .describe("Max connected nodes to return (default 30)"),
    },
    async ({ name, limit }) => {
      try {
        const result = await findConnectedConcepts(name, limit) as { network?: Record<string, unknown>[] };
        const network = result.network || [];
        if (network.length === 0) {
          return { content: [{ type: "text" as const, text: `No connections found for concept: "${name}"` }] };
        }
        const formatted = network
          .map((n) => `- [${n.relationship}] ${n.node_type}: ${n.node_name}${n.date ? ` (${n.date})` : ""}`)
          .join("\n");
        return { content: [{ type: "text" as const, text: `## Connected to "${name}"\n\n${formatted}` }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // 8. trace_concept_evolution
  gatedServer.tool(
    "trace_concept_evolution",
    "Trace how a concept appears over time — which entries contain it, what emotions and people co-occur. Shows the evolution of an idea through the writing.",
    {
      name: z
        .string()
        .describe("The concept to trace over time"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(50)
        .default(20)
        .describe("Max entries to return (default 20)"),
    },
    async ({ name, limit }) => {
      try {
        const result = await traceConceptEvolution(name, limit) as { evolution?: Record<string, unknown>[] };
        const evolution = result.evolution || [];
        if (evolution.length === 0) {
          return { content: [{ type: "text" as const, text: `No evolution data for: "${name}"` }] };
        }
        const formatted = evolution
          .map((e) => `### ${e.date} (${e.word_count} words)\n- Emotions: ${(e.emotions as string[]).join(", ") || "none"}\n- People: ${(e.people as string[]).join(", ") || "none"}`)
          .join("\n\n");
        return { content: [{ type: "text" as const, text: `## Evolution of "${name}"\n\n${formatted}` }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // 9. find_entity_relationships
  gatedServer.tool(
    "find_entity_relationships",
    "Find all entries mentioning a person, along with concepts and emotions from those entries. Maps a person's presence across the corpus.",
    {
      name: z
        .string()
        .describe("The person's name to look up"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(100)
        .default(30)
        .describe("Max entries to return (default 30)"),
    },
    async ({ name, limit }) => {
      try {
        const result = await findEntityRelationships(name, limit) as { network?: Record<string, unknown>[] };
        const network = result.network || [];
        if (network.length === 0) {
          return { content: [{ type: "text" as const, text: `No entries found mentioning: "${name}"` }] };
        }
        const formatted = network
          .map((e) => `### ${e.date}\n- Concepts: ${(e.concepts as string[]).join(", ") || "none"}\n- Emotions: ${(e.emotions as string[]).join(", ") || "none"}`)
          .join("\n\n");
        return { content: [{ type: "text" as const, text: `## Entries mentioning "${name}"\n\n${formatted}` }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // 10. compare_periods
  gatedServer.tool(
    "compare_periods",
    "Compare concepts, emotions, and archetypes between two time periods. Great for questions like 'how was January different from March?'",
    {
      start1: z.string().describe("Start of first period (YYYY-MM-DD)"),
      end1: z.string().describe("End of first period (YYYY-MM-DD)"),
      start2: z.string().describe("Start of second period (YYYY-MM-DD)"),
      end2: z.string().describe("End of second period (YYYY-MM-DD)"),
    },
    async ({ start1, end1, start2, end2 }) => {
      try {
        const result = await comparePeriods(start1, end1, start2, end2) as Record<string, unknown>;
        return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // 11. get_decision_context
  gatedServer.tool(
    "get_decision_context",
    "Find recorded decisions and their surrounding emotional and conceptual context. Optionally filter by keyword.",
    {
      keyword: z
        .string()
        .optional()
        .describe("Optional keyword to filter decisions"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(50)
        .default(10)
        .describe("Max decisions to return (default 10)"),
    },
    async ({ keyword, limit }) => {
      try {
        const result = await getDecisionContext(keyword, limit) as { decisions?: Record<string, unknown>[] };
        const decisions = result.decisions || [];
        if (decisions.length === 0) {
          return { content: [{ type: "text" as const, text: keyword ? `No decisions matching: "${keyword}"` : "No decisions found." }] };
        }
        const formatted = decisions
          .map((d) => `### ${d.date}\n> ${d.decision}\n- Emotions: ${(d.emotions as string[]).join(", ") || "none"}\n- Concepts: ${(d.concepts as string[]).join(", ") || "none"}`)
          .join("\n\n");
        return { content: [{ type: "text" as const, text: `## Decisions\n\n${formatted}` }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // 12. get_archetype_patterns
  gatedServer.tool(
    "get_archetype_patterns",
    "Get archetypal patterns (Warrior, Sage, Creator, Healer, etc.) found in the writing, their frequency, strength, and associated emotions.",
    {
      limit: z
        .number()
        .int()
        .min(1)
        .max(20)
        .default(10)
        .describe("Max archetypes to return (default 10)"),
    },
    async ({ limit }) => {
      try {
        const result = await getArchetypePatterns(limit) as { archetypes?: Record<string, unknown>[] };
        const archetypes = result.archetypes || [];
        if (archetypes.length === 0) {
          return { content: [{ type: "text" as const, text: "No archetype patterns found." }] };
        }
        const formatted = archetypes
          .map((a) => `### ${a.archetype}\n- Entries: ${a.entry_count}\n- Avg strength: ${a.avg_strength}\n- Associated emotions: ${(a.associated_emotions as string[]).join(", ") || "none"}\n- Sample dates: ${(a.sample_dates as string[]).join(", ")}`)
          .join("\n\n");
        return { content: [{ type: "text" as const, text: `## Archetype Patterns\n\n${formatted}` }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // 13. get_concept_flows
  gatedServer.tool(
    "get_concept_flows",
    "Find directed flow transitions (X -> Y) connected to a concept. Useful for movement-pattern questions like 'fear to action' or mindset shifts over time.",
    {
      name: z
        .string()
        .describe("The concept name to inspect for incoming/outgoing flows"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(100)
        .default(20)
        .describe("Max flow transitions to return (default 20)"),
    },
    async ({ name, limit }) => {
      try {
        const result = await getConceptFlows(name, limit) as { flows?: Record<string, unknown>[] };
        const flows = result.flows || [];
        if (flows.length === 0) {
          return { content: [{ type: "text" as const, text: `No concept flows found for: "${name}"` }] };
        }
        const formatted = flows
          .map(
            (flow) =>
              `- ${flow.source} -> ${flow.target} (weight: ${flow.weight}, direction: ${flow.direction})${flow.sample_phrase ? `\n  sample: ${flow.sample_phrase}` : ""}`
          )
          .join("\n");
        return { content: [{ type: "text" as const, text: `## Concept Flows for "${name}"\n\n${formatted}` }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // 14. search_themes
  gatedServer.tool(
    "search_themes",
    "Find entries linked to a specific theme and see which themes co-occur with it. Themes are 2-4 word psychological/narrative patterns like 'cultivating resilience', 'seeking clarity', or 'struggling with discipline'. Use this to explore thematic patterns across the writing.",
    {
      name: z
        .string()
        .describe("Theme to search for (e.g., 'cultivating resilience', 'seeking clarity')"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(100)
        .default(30)
        .describe("Max entries to return (default 30)"),
    },
    async ({ name, limit }) => {
      try {
        const result = await getThemeNetwork(name, limit) as {
          theme?: string;
          entries?: Record<string, unknown>[];
          cooccurrences?: Record<string, unknown>[];
        };
        const entries = result.entries || [];
        const cooccurrences = result.cooccurrences || [];

        const parts: string[] = [`## Theme: "${result.theme || name}"\n`];

        if (cooccurrences.length > 0) {
          parts.push("### Co-occurring Themes");
          parts.push(
            cooccurrences
              .map((c) => `- ${c.theme} (weight: ${c.weight})`)
              .join("\n")
          );
          parts.push("");
        }

        if (entries.length === 0) {
          parts.push("No entries found for this theme.");
        } else {
          parts.push(`### Entries (${entries.length})`);
          parts.push(
            entries
              .map((e) => {
                const emotions = (e.emotions as string[])?.join(", ") || "none";
                const coThemes = (e.co_themes as string[])?.slice(0, 5).join(", ") || "none";
                return `- **${e.date}** ${e.filename}\n  Summary: ${e.summary || "(no summary)"}\n  Valence: ${e.valence ?? "n/a"} | Emotions: ${emotions}\n  Co-themes: ${coThemes}`;
              })
              .join("\n\n")
          );
        }

        return { content: [{ type: "text" as const, text: parts.join("\n") }] };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );

  // 15. search_by_state
  gatedServer.tool(
    "search_by_state",
    "Find entries filtered by psychological state dimension. Dimensions: valence (-1=heavy/+1=uplifted), activation (-1=calm/+1=activated), agency (-1=stuck/+1=empowered), certainty (-1=conflicted/+1=resolved), relational_openness (-1=guarded/+1=open), self_trust (-1=doubt/+1=trust), time_orientation (-1=past_looping/+1=future_building), integration (-1=fragmented/+1=coherent).",
    {
      dimension: z
        .string()
        .describe("State dimension to filter by (e.g., 'valence', 'agency', 'self_trust')"),
      min_score: z
        .number()
        .min(-1)
        .max(1)
        .default(-1)
        .describe("Minimum score (-1 to 1)"),
      max_score: z
        .number()
        .min(-1)
        .max(1)
        .default(1)
        .describe("Maximum score (-1 to 1)"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(100)
        .default(20)
        .describe("Max entries to return (default 20)"),
    },
    async ({ dimension, min_score, max_score, limit }) => {
      try {
        const result = await getEntriesByState(dimension, min_score, max_score, limit) as {
          dimension?: string;
          entries?: Record<string, unknown>[];
        };
        const entries = result.entries || [];
        if (entries.length === 0) {
          return {
            content: [{ type: "text" as const, text: `No entries found for ${dimension} between ${min_score} and ${max_score}.` }],
          };
        }
        const formatted = entries
          .map((e) => {
            const themes = (e.themes as string[])?.join(", ") || "none";
            return `- **${e.date}** (${dimension}: ${e.score})\n  ${e.summary || e.filename}\n  Themes: ${themes}`;
          })
          .join("\n\n");
        return {
          content: [{ type: "text" as const, text: `## Entries by ${dimension} [${min_score} to ${max_score}]\n\n${formatted}` }],
        };
      } catch (err) {
        return { content: [{ type: "text" as const, text: String(err) }], isError: true };
      }
    }
  );
}
