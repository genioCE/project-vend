import Anthropic from "@anthropic-ai/sdk";
import { STATE_LABEL_SYSTEM_PROMPT } from "./prompts.js";

const DIMENSIONS = [
  "valence", "activation", "agency", "certainty",
  "relational_openness", "self_trust", "time_orientation", "integration",
] as const;

export interface StateDimension {
  dimension: string;
  score: number;
  label: string;
  rationale: string;
}

export interface StateProvider {
  name: string;
  generate(entryId: string, text: string): Promise<StateDimension[]>;
}

function normalizeDimensionName(raw: string): string {
  return raw.toLowerCase().replace(/[\s-]+/g, "_");
}

function parseDimensions(json: Record<string, unknown>): StateDimension[] {
  const rawDims =
    (json.dimensions as unknown[]) ??
    (json.dimension_scores as unknown[]) ??
    (json.scores as unknown[]) ??
    [];

  // Handle array of dimension objects
  if (Array.isArray(rawDims) && rawDims.length > 0) {
    const parsed: StateDimension[] = [];
    for (const item of rawDims) {
      if (typeof item !== "object" || item === null) continue;
      const obj = item as Record<string, unknown>;
      const name = normalizeDimensionName(
        String(obj.dimension ?? obj.name ?? obj.dim ?? "")
      );
      if (!name) continue;
      parsed.push({
        dimension: name,
        score: Number(obj.score ?? 0),
        label: String(obj.label ?? ""),
        rationale: String(obj.rationale ?? ""),
      });
    }
    return parsed;
  }

  // Handle flat dict format: {"valence": 0.5, ...} or {"valence": {"score": 0.5}}
  const parsed: StateDimension[] = [];
  for (const dim of DIMENSIONS) {
    const val = json[dim];
    if (val == null) continue;
    if (typeof val === "number") {
      parsed.push({ dimension: dim, score: val, label: "", rationale: "" });
    } else if (typeof val === "object") {
      const obj = val as Record<string, unknown>;
      parsed.push({
        dimension: dim,
        score: Number(obj.score ?? 0),
        label: String(obj.label ?? ""),
        rationale: String(obj.rationale ?? ""),
      });
    }
  }
  return parsed;
}

function validateDimensions(dims: StateDimension[]): StateDimension[] {
  const byName = new Map(dims.map((d) => [d.dimension, d]));
  const result: StateDimension[] = [];

  for (const dim of DIMENSIONS) {
    const existing = byName.get(dim);
    if (existing) {
      // Clamp score to [-1, 1]
      existing.score = Math.max(-1, Math.min(1, existing.score));
      result.push(existing);
    } else {
      result.push({
        dimension: dim,
        score: 0.0,
        label: "between low and high",
        rationale: "No signal detected",
      });
    }
  }
  return result;
}

// ── Anthropic Provider ──────────────────────────────────────────────

export class AnthropicProvider implements StateProvider {
  name = "anthropic";
  private client: Anthropic;
  private model: string;

  constructor(model?: string) {
    this.client = new Anthropic();
    this.model = model ?? process.env.REPROCESS_MODEL ?? "claude-haiku-4-5-20251001";
  }

  async generate(entryId: string, text: string): Promise<StateDimension[]> {
    const maxRetries = 3;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const response = await this.client.messages.create({
          model: this.model,
          max_tokens: 2048,
          temperature: 0.2,
          system: STATE_LABEL_SYSTEM_PROMPT,
          messages: [
            {
              role: "user",
              content: `Entry ID: ${entryId}\n\n${text}`,
            },
          ],
        });

        const content = response.content[0];
        if (content.type !== "text") {
          throw new Error(`Unexpected response type: ${content.type}`);
        }

        const cleaned = stripCodeFences(content.text);
        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(cleaned);
        } catch {
          parsed = repairJson(cleaned);
        }
        return validateDimensions(parseDimensions(parsed));
      } catch (e: unknown) {
        const isRateLimit =
          e instanceof Error && (e.message.includes("429") || e.message.includes("rate_limit"));
        if (isRateLimit && attempt < maxRetries - 1) {
          const wait = (attempt + 1) * 15_000; // 15s, 30s, 45s
          process.stderr.write(`  ${entryId}: rate limited, waiting ${wait / 1000}s...\n`);
          await new Promise((r) => setTimeout(r, wait));
          continue;
        }
        throw e;
      }
    }
    throw new Error("Unreachable");
  }
}

function stripCodeFences(text: string): string {
  let s = text.trim();
  if (s.startsWith("```")) {
    s = s.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
  }
  return s;
}

function repairJson(text: string): Record<string, unknown> {
  // Try truncating at the last complete dimension object
  // Find the dimensions array and extract complete entries
  const dimMatch = text.match(/"dimensions"\s*:\s*\[/);
  if (!dimMatch || dimMatch.index === undefined) {
    throw new Error("Cannot find dimensions array in response");
  }

  const start = dimMatch.index + dimMatch[0].length;
  const dims: Record<string, unknown>[] = [];
  let depth = 0;
  let objStart = -1;

  for (let i = start; i < text.length; i++) {
    if (text[i] === "{") {
      if (depth === 0) objStart = i;
      depth++;
    } else if (text[i] === "}") {
      depth--;
      if (depth === 0 && objStart >= 0) {
        const objStr = text.slice(objStart, i + 1);
        try {
          dims.push(JSON.parse(objStr));
        } catch {
          // Try fixing common issues: single quotes, trailing commas
          try {
            const fixed = objStr
              .replace(/'/g, '"')
              .replace(/,\s*}/g, "}");
            dims.push(JSON.parse(fixed));
          } catch {
            // Skip this dimension
          }
        }
      }
    }
  }

  if (dims.length === 0) {
    throw new Error(`JSON repair failed — no dimensions extracted from response`);
  }

  return { dimensions: dims };
}

// ── OpenAI Provider ─────────────────────────────────────────────────

export class OpenAIProvider implements StateProvider {
  name = "openai";
  private apiKey: string;
  private model: string;
  private baseUrl: string;

  constructor(model?: string) {
    this.apiKey = process.env.OPENAI_API_KEY ?? "";
    this.model = model ?? process.env.REPROCESS_MODEL ?? "gpt-4o-mini";
    this.baseUrl = process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1";
    if (!this.apiKey) throw new Error("OPENAI_API_KEY not set");
  }

  async generate(entryId: string, text: string): Promise<StateDimension[]> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        temperature: 0.2,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: STATE_LABEL_SYSTEM_PROMPT },
          { role: "user", content: `Entry ID: ${entryId}\n\n${text}` },
        ],
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`OpenAI API error ${response.status}: ${err}`);
    }

    const data = (await response.json()) as {
      choices: Array<{ message: { content: string } }>;
    };
    const parsed = JSON.parse(data.choices[0].message.content);
    return validateDimensions(parseDimensions(parsed));
  }
}

// ── Factory ─────────────────────────────────────────────────────────

export function createProvider(name?: string): StateProvider {
  const providerName = name ?? process.env.REPROCESS_PROVIDER ?? "anthropic";

  switch (providerName) {
    case "anthropic":
      return new AnthropicProvider();
    case "openai":
      return new OpenAIProvider();
    default:
      throw new Error(
        `Unknown provider: "${providerName}". Use "anthropic" or "openai".`
      );
  }
}
