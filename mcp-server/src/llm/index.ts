export type { LLMProvider, LLMCallOptions, LLMStreamOptions } from "./provider.js";
export type { LLMMessage, ToolDef, ToolCall, LLMResponse, MessageRole } from "./types.js";

import type { LLMProvider } from "./provider.js";
import { OllamaProvider } from "./ollama-provider.js";
import { AnthropicProvider } from "./anthropic-provider.js";

function parseBoundedFloat(
  value: string | undefined,
  fallback: number,
  min: number,
  max: number
): number {
  if (!value) return fallback;
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

export function createProvider(): LLMProvider {
  const name = (process.env.LLM_PROVIDER || "ollama").toLowerCase().trim();

  switch (name) {
    case "anthropic":
    case "claude": {
      const apiKey = process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        throw new Error(
          "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic"
        );
      }
      return new AnthropicProvider({
        apiKey,
        model: process.env.ANTHROPIC_MODEL || "claude-sonnet-4-20250514",
        maxTokens: Math.max(
          1,
          parseInt(process.env.ANTHROPIC_MAX_TOKENS || "4096", 10) || 4096
        ),
        temperature: parseBoundedFloat(process.env.ANTHROPIC_TEMPERATURE, 0.3, 0, 1),
        topP: parseBoundedFloat(process.env.ANTHROPIC_TOP_P, 0.9, 0, 1),
      });
    }
    case "ollama":
    default:
      return new OllamaProvider({
        url: process.env.OLLAMA_URL || "http://host.docker.internal:11434",
        model: process.env.OLLAMA_MODEL || "llama3.2",
        keepAlive: process.env.OLLAMA_KEEP_ALIVE || "2h",
        temperature: parseBoundedFloat(process.env.OLLAMA_TEMPERATURE, 0.3, 0, 2),
        topP: parseBoundedFloat(process.env.OLLAMA_TOP_P, 0.9, 0, 1),
        repeatPenalty: parseBoundedFloat(process.env.OLLAMA_REPEAT_PENALTY, 1.18, 0.5, 2),
      });
  }
}
