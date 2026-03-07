export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface ToolCall {
  function: { name: string; arguments: unknown };
}

export interface LLMMessage {
  role: MessageRole;
  content: string;
  tool_calls?: ToolCall[];
}

export interface ToolDef {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: string;
      properties: Record<string, { type: string; description: string }>;
      required: string[];
    };
  };
}

export interface LLMResponse {
  content: string;
  tool_calls?: ToolCall[];
  done: boolean;
}
