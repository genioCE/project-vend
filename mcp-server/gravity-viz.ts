/**
 * Gravity Orchestrator — Diagnostic Visualization Server
 *
 * Runs the full gravity pipeline and returns all intermediate data
 * for the orbital visualization.
 *
 * Run:  source ../.env && npx tsx gravity-viz.ts
 * Open: http://localhost:4000
 */

// Force localhost — override any container hostnames from .env
process.env.EMBEDDINGS_SERVICE_URL = "http://localhost:8000";
process.env.GRAPH_SERVICE_URL = "http://localhost:8001";
process.env.ANALYSIS_SERVICE_URL = "http://localhost:8002";
process.env.DUCKDB_PATH = process.env.DUCKDB_PATH || "./data/timeseries.duckdb";

import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { decompose, embedDecomposition } from "./src/gravity/decompose.js";
import { getIdentityVectors, TOOLS, TOOL_NAMES, META_TOOLS, ALWAYS_ACTIVE_TOOLS } from "./src/gravity/identities.js";
import {
  computePullMatrix,
  computeCompositeActivation,
  identifyPrimaryMass,
  findActivationCutoff,
  dotProduct,
} from "./src/gravity/field.js";
import { computeGravityField } from "./src/gravity/field.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.post("/api/diagnose", async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) return res.status(400).json({ error: "query required" });

    const start = performance.now();

    // Step 1: Decompose
    const decomposition = await decompose(query);
    const decomposeMs = performance.now() - start;

    // Step 2: Embed
    const embedStart = performance.now();
    await embedDecomposition(decomposition, query);
    const embedMs = performance.now() - embedStart;

    // Step 3: Identity vectors
    const idStart = performance.now();
    const identityVectors = await getIdentityVectors();
    const identityMs = performance.now() - idStart;

    // Step 4: Full gravity computation (for activated tools)
    const fieldStart = performance.now();
    const field = computeGravityField(identityVectors, decomposition);
    const fieldMs = performance.now() - fieldStart;

    // Step 4b: Get raw pull matrix for visualization
    const fragmentVectors = decomposition.fragments
      .map((f) => f.embedding!)
      .filter(Boolean);
    const allVectors = [...fragmentVectors, decomposition.query_embedding!];
    const pullMatrix = computePullMatrix(identityVectors, allVectors);
    const compositeScores = computeCompositeActivation(pullMatrix);

    // Step 5: Compute 2D positions using MDS-like projection from pull values
    // Use the pull matrix to create a 2D layout
    const activatedNames = new Set(field.activated.map((t) => t.name));

    const tools = TOOL_NAMES.map((name, i) => {
      const fragmentPulls = pullMatrix[i].slice(0, fragmentVectors.length);
      const queryPull = pullMatrix[i][fragmentVectors.length];

      return {
        name,
        composite_score: compositeScores[i],
        fragment_pulls: fragmentPulls,
        query_pull: queryPull,
        is_meta: META_TOOLS.has(name),
        is_always_active: ALWAYS_ACTIVE_TOOLS.has(name),
        activated: activatedNames.has(name),
        category: TOOLS[i].description.slice(0, 50),
      };
    });

    // Assign tool categories for coloring
    const toolCategories: Record<string, string> = {};
    for (const t of TOOLS) {
      if (["search_writings", "search_by_keyword", "get_entries_by_date", "get_recent_entries", "temporal_filter", "search_by_state"].includes(t.name))
        toolCategories[t.name] = "search";
      else if (["find_recurring_themes", "trace_concept_evolution", "get_concept_flows", "search_themes"].includes(t.name))
        toolCategories[t.name] = "pattern";
      else if (["find_connected_concepts", "find_entity_relationships"].includes(t.name))
        toolCategories[t.name] = "graph";
      else if (["get_entry_analysis", "get_archetype_patterns"].includes(t.name))
        toolCategories[t.name] = "psychological";
      else if (["query_time_series", "detect_anomalies", "correlate_metrics", "get_metric_summary", "compare_periods"].includes(t.name))
        toolCategories[t.name] = "quantitative";
      else if (["get_writing_stats", "list_available_metrics"].includes(t.name))
        toolCategories[t.name] = "meta";
      else if (["get_decision_context"].includes(t.name))
        toolCategories[t.name] = "psychological";
    }

    const totalMs = performance.now() - start;

    res.json({
      query,
      timing: {
        decompose_ms: Math.round(decomposeMs),
        embed_ms: Math.round(embedMs),
        identity_ms: Math.round(identityMs),
        field_ms: Math.round(fieldMs),
        total_ms: Math.round(totalMs),
      },
      decomposition: {
        fragments: decomposition.fragments.map((f) => ({
          type: f.type,
          text: f.text,
        })),
        primary_mass_index: decomposition.primary_mass_index,
        reasoning: decomposition.reasoning,
        extracted: decomposition.extracted,
      },
      tools: tools.map((t) => ({
        ...t,
        category: toolCategories[t.name] || "other",
      })),
      adaptive_cutoff: field.adaptive_cutoff,
      activated_count: field.activated.length,
    });
  } catch (err) {
    console.error("Diagnose error:", err);
    res.status(500).json({ error: String(err) });
  }
});

const PORT = 4000;
app.listen(PORT, () => {
  console.log(`\n  Gravity Visualization: http://localhost:${PORT}\n`);
});
