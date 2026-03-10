import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { orchestrate } from "./orchestrate.js";

// Size limits to prevent MCP response from exceeding buffer limits
const MAX_TOOL_RESULT_BYTES = 100_000; // 100KB per tool result
const MAX_TOTAL_BYTES = 800_000; // 800KB total (leave headroom below 1MB)

/**
 * Truncate a tool result to fit within size limits.
 * Tries to truncate JSON arrays intelligently by removing items from the end.
 */
function truncateResult(result: string, maxBytes: number): string {
  if (Buffer.byteLength(result, "utf8") <= maxBytes) {
    return result;
  }

  // Try to parse as JSON and truncate arrays intelligently
  try {
    const parsed = JSON.parse(result);

    // Find array fields and truncate them
    const truncateArrays = (obj: unknown, depth = 0): unknown => {
      if (depth > 3) return obj; // Don't recurse too deep

      if (Array.isArray(obj)) {
        // Keep halving the array until it fits
        let arr = obj;
        while (arr.length > 1) {
          const truncated = JSON.stringify(arr);
          if (Buffer.byteLength(truncated, "utf8") <= maxBytes * 0.8) {
            return arr;
          }
          arr = arr.slice(0, Math.ceil(arr.length / 2));
        }
        return arr;
      }

      if (obj && typeof obj === "object") {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
          result[key] = truncateArrays(value, depth + 1);
        }
        return result;
      }

      return obj;
    };

    const truncated = truncateArrays(parsed);
    const truncatedStr = JSON.stringify(truncated, null, 2);

    if (Buffer.byteLength(truncatedStr, "utf8") <= maxBytes) {
      return truncatedStr + "\n[... truncated for size]";
    }
  } catch {
    // Not valid JSON, fall through to simple truncation
  }

  // Simple byte-based truncation as fallback
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const bytes = encoder.encode(result);
  const truncatedBytes = bytes.slice(0, maxBytes - 50);
  return decoder.decode(truncatedBytes) + "\n[... truncated for size]";
}

export function registerGravityTools(server: McpServer): void {
  server.tool(
    "orchestrated_query",
    "Run a comprehensive multi-tool analysis of a natural language question about the writing corpus. " +
      "Automatically decomposes the query, identifies relevant tools via semantic gravity, " +
      "executes them in parallel, and returns assembled results. " +
      "Use this for complex questions that would benefit from multiple analytical perspectives.",
    {
      query: z
        .string()
        .min(1)
        .max(2000)
        .describe("Natural language question about the writing"),
    },
    async ({ query }) => {
      try {
        const result = await orchestrate(query);

        const sections: string[] = [
          `## Orchestrated Query: "${query}"`,
          `**Fragments:** ${result.fragments.map((f) => `[${f.type}] ${f.text}`).join(" | ")}`,
          `**Primary mass:** ${result.primary_mass}`,
          `**Tools activated:** ${result.activated_tools.join(", ")} (${result.total_ms.toFixed(0)}ms total)`,
          "",
        ];

        let totalBytes = Buffer.byteLength(sections.join("\n"), "utf8");

        for (const tr of result.results) {
          if (tr.error) {
            sections.push(
              `### ${tr.tool} (FAILED: ${tr.error})`
            );
          } else {
            sections.push(
              `### ${tr.tool} (score: ${tr.composite_score.toFixed(3)}, ${tr.duration_ms.toFixed(0)}ms)`
            );

            // Calculate remaining budget for this result
            const remainingBudget = MAX_TOTAL_BYTES - totalBytes;
            const maxForThisResult = Math.min(MAX_TOOL_RESULT_BYTES, remainingBudget - 1000);

            if (maxForThisResult <= 0) {
              sections.push("[... result omitted, output size limit reached]");
            } else {
              const truncated = truncateResult(tr.result, maxForThisResult);
              sections.push(truncated);
              totalBytes += Buffer.byteLength(truncated, "utf8");
            }
          }
          sections.push("");

          // Stop adding results if we're close to the limit
          if (totalBytes >= MAX_TOTAL_BYTES - 1000) {
            sections.push("[... additional results omitted, output size limit reached]");
            break;
          }
        }

        return {
          content: [{ type: "text" as const, text: sections.join("\n") }],
        };
      } catch (err) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Orchestration failed: ${String(err)}`,
            },
          ],
          isError: true,
        };
      }
    }
  );
}
