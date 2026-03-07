/**
 * Quick smoke test for the gravity orchestrator.
 * Run: EMBEDDINGS_SERVICE_URL=http://localhost:8000 GRAPH_SERVICE_URL=http://localhost:8001 npx tsx test-gravity.ts
 *
 * Requires docker compose services to be running.
 */

// Point to Docker services via localhost (not container names)
process.env.EMBEDDINGS_SERVICE_URL = process.env.EMBEDDINGS_SERVICE_URL || "http://localhost:8000";
process.env.GRAPH_SERVICE_URL = process.env.GRAPH_SERVICE_URL || "http://localhost:8001";
process.env.ANALYSIS_SERVICE_URL = process.env.ANALYSIS_SERVICE_URL || "http://localhost:8002";
process.env.DUCKDB_PATH = process.env.DUCKDB_PATH || "./data/timeseries.duckdb";

async function main() {
  // Step 1: Test /embed endpoint
  console.log("=== Step 1: Test /embed endpoint ===");
  const { embedTexts } = await import("./src/embeddings-client.js");
  const vectors = await embedTexts(["test query", "another test"]);
  console.log(`  Got ${vectors.length} vectors, dim=${vectors[0].length}`);

  // Step 2: Test decomposition
  console.log("\n=== Step 2: Test decomposition ===");
  const { decompose, embedDecomposition } = await import("./src/gravity/decompose.js");
  const query = "tell me about Kyle";
  const decomp = await decompose(query);
  console.log(`  Query: "${query}"`);
  console.log(`  Fragments: ${decomp.fragments.map(f => `[${f.type}] ${f.text}`).join(", ")}`);
  console.log(`  Primary mass: ${decomp.fragments[decomp.primary_mass_index]?.text}`);
  console.log(`  Extracted entities: ${decomp.extracted.entities.join(", ") || "(none)"}`);
  console.log(`  Extracted concepts: ${decomp.extracted.concepts.join(", ") || "(none)"}`);

  // Step 3: Embed decomposition
  console.log("\n=== Step 3: Embed decomposition ===");
  await embedDecomposition(decomp, query);
  console.log(`  Fragment vectors: ${decomp.fragments.filter(f => f.embedding).length}/${decomp.fragments.length}`);
  console.log(`  Query vector: ${decomp.query_embedding ? "yes" : "no"}`);

  // Step 4: Load identity vectors
  console.log("\n=== Step 4: Load identity vectors ===");
  const { getIdentityVectors } = await import("./src/gravity/identities.js");
  const idVecs = await getIdentityVectors();
  console.log(`  Identity vectors: ${idVecs.length} tools × ${idVecs[0].length} dims`);

  // Step 5: Compute gravity field
  console.log("\n=== Step 5: Compute gravity field ===");
  const { computeGravityField } = await import("./src/gravity/field.js");
  const field = computeGravityField(idVecs, decomp);
  console.log(`  Activated tools (${field.activated.length}):`);
  for (const t of field.activated) {
    const tag = t.is_always_active ? " [always]" : t.is_meta ? " [meta]" : "";
    console.log(`    ${t.name}: ${t.composite_score.toFixed(3)}${tag}`);
  }
  console.log(`  Adaptive cutoff: ${field.adaptive_cutoff.toFixed(3)}`);

  // Step 6: Full orchestration — multiple queries
  console.log("\n=== Step 6: Full orchestration ===");
  const { orchestrate } = await import("./src/gravity/orchestrate.js");

  const testQueries = [
    "tell me about Kyle",
    "how has my sense of agency changed over the last 3 months?",
    "what decisions have I made about recovery?",
  ];

  for (const q of testQueries) {
    console.log(`\n--- Query: "${q}" ---`);
    const result = await orchestrate(q);
    console.log(`  Fragments: ${result.fragments.map(f => `[${f.type}] ${f.text}`).join(", ")}`);
    console.log(`  Primary mass: "${result.primary_mass}"`);
    console.log(`  Tools activated (${result.activated_tools.length}): ${result.activated_tools.join(", ")}`);
    console.log(`  Total time: ${result.total_ms.toFixed(0)}ms`);
    for (const r of result.results) {
      if (r.error) {
        console.log(`    ${r.tool}: ERROR - ${r.error.slice(0, 100)}`);
      } else {
        const preview = r.result.length < 80 ? ` → ${r.result}` : "";
        console.log(`    ${r.tool}: ${r.result.length} chars, ${r.duration_ms.toFixed(0)}ms${preview}`);
      }
    }
  }
}

main().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
