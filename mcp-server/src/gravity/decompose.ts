import Anthropic from "@anthropic-ai/sdk";
import { embedTexts } from "../embeddings-client.js";
import type {
  DecompositionResult,
  Fragment,
  FragmentType,
  ExtractedParams,
} from "./types.js";

const DECOMPOSITION_PROMPT = `\
You are a query decomposition engine for a personal writing corpus analysis system.

Given a natural language query, decompose it into typed semantic fragments AND extract structured parameters for tool dispatch.

## Fragment Types

- **concept**: Abstract ideas, themes, philosophical constructs (e.g., silence, sovereignty, shame, trust, discipline, recovery)
- **entity**: Named people, places, practices, organizations (e.g., Kyle, Blocworks, climbing, StarSpace46, Mom)
- **temporal**: Time dimensions, change markers, period references (e.g., change over time, since January, last 3 months, recently, since I started climbing)
- **emotional**: Feelings, states, psychological dimensions (e.g., self-trust, integration, agency, valence, stuck, empowered, fragmented)
- **relational**: Connection structure, influence, tension (e.g., relationship with, tension between, influence of, how X connects to Y)
- **archetypal**: Patterns, roles, mythic structures (e.g., Creator, Healer, Warrior, Sovereign, Integrator)

## Rules

1. Extract ALL meaningful fragments from the query. A single query typically has 2-6 fragments.
2. Each fragment should be a short phrase (1-5 words), not a full sentence.
3. A word/phrase can only appear in ONE fragment — no duplicates.
4. If a fragment could fit multiple types, choose the most specific type.
5. Identify the PRIMARY MASS — the fragment the query is fundamentally about. This is the subject being investigated, not the lens through which it's being viewed.
6. Empty categories are signal — don't force fragments into categories where they don't belong.
7. Infer implicit fragments only when strongly implied (e.g., "how have I changed" implies temporal: "change over time").

## Extracted Parameters

In addition to fragments, extract structured parameters that tools will use as arguments:
- **entities**: Person names mentioned in the query (e.g., ["Kyle", "Matt"])
- **concepts**: Abstract ideas or themes (e.g., ["silence", "sovereignty"])
- **date_ranges**: Any date references, converted to YYYY-MM-DD format where possible. Use approximate dates for relative references like "last summer" or "since January". If no dates mentioned, use empty array.
- **metrics**: Psychological dimensions or measurable quantities mentioned (e.g., ["agency", "self_trust", "valence", "word_count"])
- **search_query**: The full query text reformulated for semantic search (may be the original query or a cleaned version)

## Output Format

Return ONLY valid JSON with this structure:
{
  "fragments": [
    {"type": "<fragment_type>", "text": "<fragment_text>"},
    ...
  ],
  "primary_mass_index": <index of primary mass fragment in the array>,
  "reasoning": "<1-2 sentences explaining why you chose this primary mass>",
  "extracted": {
    "entities": [],
    "concepts": [],
    "date_ranges": [{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}],
    "metrics": [],
    "search_query": "<query text for semantic search>"
  }
}`;

const VALID_FRAGMENT_TYPES = new Set<string>([
  "concept",
  "entity",
  "temporal",
  "emotional",
  "relational",
  "archetypal",
]);

export async function decompose(
  query: string,
  signal?: AbortSignal
): Promise<DecompositionResult> {
  const client = new Anthropic();

  const response = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 1024,
    system: DECOMPOSITION_PROMPT,
    messages: [{ role: "user", content: query }],
  });

  let text =
    response.content[0].type === "text" ? response.content[0].text.trim() : "";

  // Handle markdown code blocks
  if (text.startsWith("```")) {
    text = text.split("\n").slice(1).join("\n");
    if (text.endsWith("```")) {
      text = text.slice(0, -3).trim();
    }
  }

  const data = JSON.parse(text) as {
    fragments: Array<{ type: string; text: string }>;
    primary_mass_index: number;
    reasoning?: string;
    extracted?: {
      entities?: string[];
      concepts?: string[];
      date_ranges?: Array<{ start?: string; end?: string }>;
      metrics?: string[];
      search_query?: string;
    };
  };

  const fragments: Fragment[] = data.fragments.map((f) => ({
    type: (VALID_FRAGMENT_TYPES.has(f.type) ? f.type : "concept") as FragmentType,
    text: f.text,
  }));

  const ext = data.extracted || {};

  // Normalize metric names: hyphens → underscores, lowercase
  const normalizedMetrics = (ext.metrics || []).map((m) =>
    m.toLowerCase().trim().replace(/-/g, "_")
  );

  const extracted: ExtractedParams = {
    entities: ext.entities || [],
    concepts: ext.concepts || [],
    date_ranges: ext.date_ranges || [],
    metrics: normalizedMetrics,
    search_query: ext.search_query || query,
  };

  return {
    fragments,
    primary_mass_index: data.primary_mass_index,
    reasoning: data.reasoning || "",
    extracted,
  };
}

export async function embedDecomposition(
  result: DecompositionResult,
  query: string,
  signal?: AbortSignal
): Promise<DecompositionResult> {
  const texts = [...result.fragments.map((f) => f.text), query];
  const vectors = await embedTexts(texts, signal);

  result.fragments.forEach((f, i) => {
    f.embedding = vectors[i];
  });
  result.query_embedding = vectors[vectors.length - 1];

  return result;
}
