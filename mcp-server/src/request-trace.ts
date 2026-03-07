/**
 * Request trace collector — captures timing spans for a single chat request
 * and stores the most recent trace for the /debug/last-query endpoint.
 */

export interface ToolSpan {
  tool: string;
  args: Record<string, unknown>;
  duration_ms: number;
  cache_hit: boolean;
  result_chars: number;
}

export interface LLMSpan {
  provider: string;
  turn: number;
  duration_ms: number;
  tool_call_count: number;
}

export interface RequestTrace {
  request_id: string | undefined;
  conversation_id: string;
  user_message: string;
  started_at: string;
  total_ms: number;
  chat_mode: string;
  graphrag: boolean;
  system_prompt_chars: number;
  graphrag_pre_search_ms: number | null;
  llm_spans: LLMSpan[];
  tool_spans: ToolSpan[];
}

export interface TraceCollector {
  addToolSpan(
    tool: string,
    args: Record<string, unknown>,
    duration_ms: number,
    cache_hit: boolean,
    result_chars: number,
  ): void;
  addLLMSpan(
    provider: string,
    turn: number,
    duration_ms: number,
    tool_call_count: number,
  ): void;
  setGraphRagPreSearch(ms: number): void;
  setSystemPromptChars(n: number): void;
  finish(): RequestTrace;
}

let lastTrace: RequestTrace | null = null;

export function startTrace(
  requestId: string | undefined,
  conversationId: string,
  userMessage: string,
  chatMode: string,
  graphrag: boolean,
): TraceCollector {
  const startedAt = new Date().toISOString();
  const startMs = performance.now();
  const toolSpans: ToolSpan[] = [];
  const llmSpans: LLMSpan[] = [];
  let graphragPreSearchMs: number | null = null;
  let systemPromptChars = 0;

  return {
    addToolSpan(tool, args, duration_ms, cache_hit, result_chars) {
      toolSpans.push({ tool, args, duration_ms, cache_hit, result_chars });
      if (cache_hit) {
        console.log(`[cache hit] ${tool}`);
      } else {
        console.log(`[tool] ${tool} ${duration_ms.toFixed(0)}ms`);
      }
    },

    addLLMSpan(provider, turn, duration_ms, tool_call_count) {
      llmSpans.push({ provider, turn, duration_ms, tool_call_count });
      console.log(
        `[${provider}] Turn ${turn} ${duration_ms.toFixed(0)}ms (${tool_call_count} tool calls)`,
      );
    },

    setGraphRagPreSearch(ms) {
      graphragPreSearchMs = ms;
      console.log(`[graphrag] Pre-search ${ms.toFixed(0)}ms`);
    },

    setSystemPromptChars(n) {
      systemPromptChars = n;
    },

    finish() {
      const trace: RequestTrace = {
        request_id: requestId,
        conversation_id: conversationId,
        user_message: userMessage,
        started_at: startedAt,
        total_ms: Math.round(performance.now() - startMs),
        chat_mode: chatMode,
        graphrag,
        system_prompt_chars: systemPromptChars,
        graphrag_pre_search_ms: graphragPreSearchMs,
        llm_spans: llmSpans,
        tool_spans: toolSpans,
      };
      lastTrace = trace;
      return trace;
    },
  };
}

export function getLastTrace(): RequestTrace | null {
  return lastTrace;
}
