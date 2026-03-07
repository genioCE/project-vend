import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { orchestrate } from "./orchestrate.js";

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

        for (const tr of result.results) {
          if (tr.error) {
            sections.push(
              `### ${tr.tool} (FAILED: ${tr.error})`
            );
          } else {
            sections.push(
              `### ${tr.tool} (score: ${tr.composite_score.toFixed(3)}, ${tr.duration_ms.toFixed(0)}ms)`
            );
            sections.push(tr.result);
          }
          sections.push("");
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
