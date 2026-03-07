import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { registerTools } from "./tools.js";
import { registerTimeSeriesTools } from "./timeseries/tools.js";
import { registerGravityTools } from "./gravity/tools.js";

async function main() {
  const server = new McpServer({
    name: "corpus-intelligence",
    version: "1.0.0",
  });

  const gravityMode = process.env.GRAVITY_MODE !== "0";

  if (gravityMode) {
    // Curated tool set: orchestrator + direct-access follow-up tools
    registerGravityTools(server);
    registerTools(server, {
      only: ["get_entry_analysis", "get_entries_by_date", "get_recent_entries"],
    });
    console.error("[mcp] gravity mode: orchestrated_query + 3 follow-up tools");
  } else {
    // Full tool set: all individual tools exposed
    registerTools(server);
    registerTimeSeriesTools(server);
    registerGravityTools(server);
    console.error("[mcp] full mode: all tools exposed");
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  process.stderr.write(`Fatal error: ${err}\n`);
  process.exit(1);
});
