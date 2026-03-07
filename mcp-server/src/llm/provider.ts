import type { LLMMessage, ToolDef, LLMResponse } from "./types.js";

export interface LLMCallOptions {
  messages: LLMMessage[];
  tools: ToolDef[];
  systemPrompt?: string;
  signal?: AbortSignal;
  modelOverride?: string;
}

export interface LLMStreamOptions extends LLMCallOptions {
  onToken: (content: string) => void;
}

export interface LLMProvider {
  readonly name: string;
  chat(options: LLMCallOptions): Promise<LLMResponse>;
  chatStreaming(options: LLMStreamOptions): Promise<LLMResponse>;
  healthCheck(): Promise<boolean>;
}
