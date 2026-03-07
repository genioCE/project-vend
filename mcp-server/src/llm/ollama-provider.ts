import type { LLMProvider, LLMCallOptions, LLMStreamOptions } from "./provider.js";
import type { LLMMessage, ToolDef, LLMResponse, ToolCall } from "./types.js";

interface OllamaConfig {
  url: string;
  model: string;
  keepAlive: string;
  temperature: number;
  topP: number;
  repeatPenalty: number;
}

interface OllamaChatResponse {
  model: string;
  message: {
    role: string;
    content: string;
    tool_calls?: ToolCall[];
  };
  done: boolean;
}

export class OllamaProvider implements LLMProvider {
  readonly name = "ollama";
  private config: OllamaConfig;

  constructor(config: OllamaConfig) {
    this.config = config;
  }

  async chat(options: LLMCallOptions): Promise<LLMResponse> {
    this.throwIfAborted(options.signal);
    const url = `${this.config.url}/api/chat`;
    const body = {
      model: options.modelOverride || this.config.model,
      messages: options.messages,
      tools: options.tools,
      stream: false,
      keep_alive: this.config.keepAlive,
      options: {
        temperature: this.config.temperature,
        top_p: this.config.topP,
        repeat_penalty: this.config.repeatPenalty,
      },
    };

    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: options.signal,
      });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw err;
      }
      throw new Error(
        `Ollama unavailable at ${this.config.url}. Is Ollama running on the host?`
      );
    }

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Ollama error (${response.status}): ${text}`);
    }

    const data = (await response.json()) as OllamaChatResponse;
    return {
      content: data.message.content || "",
      tool_calls: data.message.tool_calls,
      done: data.done,
    };
  }

  async chatStreaming(options: LLMStreamOptions): Promise<LLMResponse> {
    this.throwIfAborted(options.signal);
    const url = `${this.config.url}/api/chat`;
    const body = {
      model: options.modelOverride || this.config.model,
      messages: options.messages,
      tools: options.tools,
      stream: true,
      keep_alive: this.config.keepAlive,
      options: {
        temperature: this.config.temperature,
        top_p: this.config.topP,
        repeat_penalty: this.config.repeatPenalty,
      },
    };

    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: options.signal,
      });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw err;
      }
      throw new Error(
        `Ollama unavailable at ${this.config.url}. Is Ollama running on the host?`
      );
    }

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Ollama error (${response.status}): ${text}`);
    }

    // Read NDJSON stream line-by-line
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullContent = "";
    let toolCalls: ToolCall[] | undefined;
    let done = false;

    while (true) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep the last potentially incomplete line in the buffer
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        let chunk: OllamaChatResponse;
        try {
          chunk = JSON.parse(trimmed) as OllamaChatResponse;
        } catch {
          continue;
        }

        done = chunk.done || done;

        if (chunk.message?.content) {
          fullContent += chunk.message.content;
          options.onToken(chunk.message.content);
        }

        if (chunk.message?.tool_calls && chunk.message.tool_calls.length > 0) {
          toolCalls = chunk.message.tool_calls;
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      try {
        const chunk = JSON.parse(buffer.trim()) as OllamaChatResponse;
        done = chunk.done || done;
        if (chunk.message?.content) {
          fullContent += chunk.message.content;
          options.onToken(chunk.message.content);
        }
        if (chunk.message?.tool_calls && chunk.message.tool_calls.length > 0) {
          toolCalls = chunk.message.tool_calls;
        }
      } catch {
        // ignore partial trailing data
      }
    }

    return {
      content: fullContent,
      tool_calls: toolCalls,
      done,
    };
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.url}/api/tags`);
      return response.ok;
    } catch {
      return false;
    }
  }

  private throwIfAborted(signal?: AbortSignal): void {
    if (signal?.aborted) {
      throw new Error("request aborted");
    }
  }
}
