import Anthropic from "@anthropic-ai/sdk";
import type {
  MessageParam,
  ContentBlockParam,
  Tool,
  ToolResultBlockParam,
  ToolUseBlockParam,
  TextBlockParam,
} from "@anthropic-ai/sdk/resources/messages/messages.js";
import type { LLMProvider, LLMCallOptions, LLMStreamOptions } from "./provider.js";
import type { LLMMessage, ToolDef, LLMResponse, ToolCall } from "./types.js";

interface AnthropicConfig {
  apiKey: string;
  model: string;
  maxTokens: number;
  temperature: number;
  topP: number;
}

export class AnthropicProvider implements LLMProvider {
  readonly name = "anthropic";
  private client: Anthropic;
  private config: AnthropicConfig;

  constructor(config: AnthropicConfig) {
    this.config = config;
    this.client = new Anthropic({ apiKey: config.apiKey });
  }

  async chat(options: LLMCallOptions): Promise<LLMResponse> {
    const { systemPrompt, messages } = this.extractSystem(options.messages);
    const tools = this.translateTools(options.tools);

    const response = await this.client.messages.create(
      {
        model: this.config.model,
        max_tokens: this.config.maxTokens,
        system: systemPrompt || undefined,
        messages,
        tools: tools.length > 0 ? tools : undefined,
        temperature: this.config.temperature,
        top_p: this.config.topP,
        stream: false,
      },
      { signal: options.signal },
    );

    return this.translateResponse(response);
  }

  async chatStreaming(options: LLMStreamOptions): Promise<LLMResponse> {
    const { systemPrompt, messages } = this.extractSystem(options.messages);
    const tools = this.translateTools(options.tools);

    const stream = this.client.messages.stream(
      {
        model: this.config.model,
        max_tokens: this.config.maxTokens,
        system: systemPrompt || undefined,
        messages,
        tools: tools.length > 0 ? tools : undefined,
        temperature: this.config.temperature,
        top_p: this.config.topP,
      },
      { signal: options.signal },
    );

    stream.on("text", (text) => {
      options.onToken(text);
    });

    const finalMessage = await stream.finalMessage();
    return this.translateResponse(finalMessage);
  }

  async healthCheck(): Promise<boolean> {
    try {
      // Use a minimal API call to verify the key is valid
      await this.client.messages.create({
        model: this.config.model,
        max_tokens: 1,
        messages: [{ role: "user", content: "hi" }],
      });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Extract system-role messages from the canonical message array and
   * return them as a single concatenated system prompt string.
   * Non-system messages are translated to Anthropic's MessageParam format.
   */
  private extractSystem(messages: LLMMessage[]): {
    systemPrompt: string;
    messages: MessageParam[];
  } {
    const systemParts: string[] = [];
    const apiMessages: MessageParam[] = [];

    // First pass: collect system messages and build conversation messages
    // We need to track tool_use IDs for correlation with tool results
    let toolUseIdCounter = 0;
    // Map from position to the IDs we assigned to tool_calls in that assistant message
    const toolCallIds: string[] = [];

    for (const msg of messages) {
      if (msg.role === "system") {
        systemParts.push(msg.content);
        continue;
      }

      if (msg.role === "assistant") {
        if (msg.tool_calls && msg.tool_calls.length > 0) {
          // Assistant message with tool calls -> content blocks
          const contentBlocks: (TextBlockParam | ToolUseBlockParam)[] = [];
          if (msg.content) {
            contentBlocks.push({ type: "text" as const, text: msg.content });
          }
          for (const tc of msg.tool_calls) {
            const id = `call_${toolUseIdCounter++}`;
            toolCallIds.push(id);
            contentBlocks.push({
              type: "tool_use" as const,
              id,
              name: tc.function.name,
              input: tc.function.arguments ?? {},
            });
          }
          apiMessages.push({
            role: "assistant",
            content: contentBlocks,
          });
        } else {
          apiMessages.push({
            role: "assistant",
            content: msg.content,
          });
        }
        continue;
      }

      if (msg.role === "tool") {
        // Tool result -> must be part of a user message with tool_result blocks
        const toolId = toolCallIds.shift() || `call_unknown_${toolUseIdCounter++}`;
        const toolResultBlock: ToolResultBlockParam = {
          type: "tool_result" as const,
          tool_use_id: toolId,
          content: msg.content,
        };

        // Anthropic requires tool_result blocks to be inside user messages.
        // If the previous API message is already a user message with content blocks,
        // append to it (for multiple tool results from parallel calls).
        const lastMsg = apiMessages[apiMessages.length - 1];
        if (lastMsg && lastMsg.role === "user" && Array.isArray(lastMsg.content)) {
          (lastMsg.content as ContentBlockParam[]).push(toolResultBlock);
        } else {
          apiMessages.push({
            role: "user",
            content: [toolResultBlock],
          });
        }
        continue;
      }

      // user role
      apiMessages.push({
        role: "user",
        content: msg.content,
      });
    }

    return {
      systemPrompt: systemParts.join("\n\n"),
      messages: apiMessages,
    };
  }

  /**
   * Translate canonical tool definitions to Anthropic's Tool format.
   */
  private translateTools(tools: ToolDef[]): Tool[] {
    return tools.map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: {
        type: "object" as const,
        properties: t.function.parameters.properties,
        required: t.function.parameters.required,
      },
    }));
  }

  /**
   * Translate an Anthropic Message response to the canonical LLMResponse.
   */
  private translateResponse(message: Anthropic.Message): LLMResponse {
    let textContent = "";
    const toolCalls: ToolCall[] = [];

    for (const block of message.content) {
      if (block.type === "text") {
        textContent += block.text;
      } else if (block.type === "tool_use") {
        toolCalls.push({
          function: {
            name: block.name,
            arguments: block.input,
          },
        });
      }
    }

    return {
      content: textContent,
      tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
      done: message.stop_reason === "end_turn" || message.stop_reason === "max_tokens",
    };
  }
}
