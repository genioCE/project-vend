import {
  createProvider,
  type LLMMessage,
} from "./llm/index.js";

import {
  classifyTurn,
  buildInterpretationProtocolInstruction,
  responseEndsWithQuestion,
  buildFallbackFollowUpQuestion,
  type TurnClassification,
  type FollowUpResolution,
} from "./agent/classify.js";

import { postProcessInterpretiveResponse } from "./agent/post-process.js";

import {
  getHistory,
  buildGraphRagQueryKey,
  getGraphRagCachedContext,
  setGraphRagCachedContext,
} from "./agent/conversation.js";

import {
  TOOLS,
  GRAPH_TOOLS,
  executeTool,
  parseToolArguments,
  truncateResult,
  buildRecentHistory,
  pickGraphCenterFromToolCall,
  pickGraphCenterFromEnriched,
  graphSearch,
  MAX_ENRICHED_CONTEXT_CHARS,
} from "./agent/tool-executor.js";

import { startTrace, type TraceCollector } from "./request-trace.js";

// Re-exports consumed by tests and api-server.ts
export { buildGraphRagQueryKey } from "./agent/conversation.js";
export { postProcessInterpretiveResponse } from "./agent/post-process.js";
export { getHistory } from "./agent/conversation.js";
export type { TurnClassification, FollowUpResolution } from "./agent/classify.js";

const llmProvider = createProvider();

export type ChatMode = "classic" | "converse";

const MAX_TOOL_TURNS = 8;

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

const MAX_STORED_HISTORY_MESSAGES = parsePositiveInt(
  process.env.MAX_STORED_HISTORY_MESSAGES,
  40
);

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new Error("request aborted");
  }
}

// --- System prompts ---

const SYSTEM_PROMPT = `You are a thoughtful assistant with access to a personal writing corpus — approximately 1 million words of daily journal entries. You have tools to search this corpus semantically, find recurring themes, retrieve entries by date, and more.

Core behavior:
- Always search the corpus when answering questions about personal history, thinking patterns, or past experiences.
- Be direct, specific, and grounded in retrieved evidence.
- Cite dates when referencing specific entries.
- After finding relevant entries, use get_entry_analysis to retrieve pre-computed psychological state profiles and summaries for deeper insight.

Conversation style:
- Sound conversational and human, not templated.
- Vary response shape naturally across turns instead of repeating the same header pattern.
- Use bullets only when they add clarity; otherwise use concise prose.
- Avoid repetitive restatements and boilerplate conclusions.`;

const GRAPHRAG_SYSTEM_PROMPT = `You are a thoughtful assistant with access to a personal writing corpus — approximately 1 million words of daily journal entries. You have both semantic search tools AND a knowledge graph with extracted entities (people, places, concepts, emotions, decisions, archetypes) and relationships.

Tool usage:
- Use standard search tools for broad semantic questions.
- Use graph tools when the question is about relationships, co-occurrence, evolution over time, comparisons, decisions, archetypes, or concept flows.
- After finding relevant entries, use get_entry_analysis to retrieve pre-computed psychological state profiles and summaries for deeper insight.

Answer quality:
- Be direct, insightful, and evidence-grounded.
- Cite dates when available.
- Draw explicit connections between entities when graph context supports them.

Conversation style:
- Keep responses conversational, not rigidly templated.
- Choose format dynamically (brief narrative, narrative + bullets, or concise synthesis) based on the user request.
- Do not repeat the same structure every turn unless the user explicitly asks for a fixed format.

Anti-redundancy rules:
- Do not restate the same claim in multiple paragraphs.
- Merge overlapping evidence instead of repeating similar points.
- Do not add a conclusion that just repeats the opening claim.`;

const CONVERSE_SYSTEM_PROMPT = `You are a grounded conversational partner for a personal writing corpus.

Conversation behavior:
- Hold a natural back-and-forth conversation, not rigid Q&A.
- Respond warmly and directly to social turns (for example greetings, check-ins).
- Keep the tone human and reflective, without sounding robotic or scripted.

Grounding behavior:
- When the user asks about their history, patterns, themes, entries, or changes over time, use corpus tools before making claims.
- When the turn is simple social conversation, do not force retrieval.
- If you do use retrieval, keep evidence integrated into the conversation naturally rather than over-formatting.
- Cite dates when referencing specific entries.
- After finding relevant entries, use get_entry_analysis to retrieve pre-computed psychological state profiles and summaries for deeper insight.`;

const CONVERSE_GRAPHRAG_SYSTEM_PROMPT = `You are a grounded conversational partner for a personal writing corpus with both semantic search tools and a knowledge graph.

Conversation behavior:
- Keep a flowing conversational style, not rigid Q&A.
- Social turns should feel natural and brief, without unnecessary analysis.

Grounding behavior:
- Use retrieval/graph tools when the user asks about relationships, patterns, people, concepts, decisions, archetypes, or changes over time.
- Do not force graph retrieval on pure social turns.
- Blend grounded evidence into natural language and cite dates when referencing specific entries.
- After finding relevant entries, use get_entry_analysis to retrieve pre-computed psychological state profiles and summaries for deeper insight.`;

// --- SSE event types ---

type SSEEventMeta = {
  request_id?: string;
};

export type SSEEvent =
  | ({ type: "text_delta"; content: string } & SSEEventMeta)
  | ({ type: "tool_call"; tool: string; input: Record<string, unknown> } & SSEEventMeta)
  | ({ type: "tool_result"; tool: string; preview: string; center?: string } & SSEEventMeta)
  | ({ type: "done" } & SSEEventMeta)
  | ({ type: "error"; message: string } & SSEEventMeta);

interface RunAgentOptions {
  abortSignal?: AbortSignal;
  replaceLastUserMessage?: boolean;
  requestId?: string;
  modelOverride?: string;
}

// --- Agent loop ---

export async function runAgent(
  conversationId: string,
  userMessage: string,
  onEvent: (event: SSEEvent) => void,
  graphRagMode: boolean = false,
  chatMode: ChatMode = "classic",
  options: RunAgentOptions = {}
): Promise<void> {
  const abortSignal = options.abortSignal;
  const replaceLastUserMessage = options.replaceLastUserMessage === true;
  const requestId = options.requestId;
  const modelOverride = options.modelOverride;
  if (abortSignal?.aborted) return;

  const trace = startTrace(requestId, conversationId, userMessage, chatMode, graphRagMode);

  const history = getHistory(conversationId);
  const previousAssistantContent = [...history]
    .reverse()
    .find((message) => message.role === "assistant")?.content;

  const turn = classifyTurn(userMessage, previousAssistantContent, graphRagMode, chatMode);

  if (turn.followUpResolution) {
    console.log(
      `[follow-up] Resolved "${userMessage}" -> "${turn.effectiveUserMessage}"`
    );
  }

  if (replaceLastUserMessage) {
    let lastUserIndex = -1;
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].role === "user") {
        lastUserIndex = i;
        break;
      }
    }
    if (lastUserIndex >= 0) {
      history.splice(lastUserIndex);
    }
  }
  history.push({ role: "user", content: userMessage });
  if (history.length > MAX_STORED_HISTORY_MESSAGES) {
    history.splice(0, history.length - MAX_STORED_HISTORY_MESSAGES);
  }

  let systemPrompt =
    chatMode === "converse"
      ? CONVERSE_SYSTEM_PROMPT
      : SYSTEM_PROMPT;
  const availableTools = turn.skipCorpusRetrieval ? [] : [...TOOLS];

  if (graphRagMode && !turn.skipCorpusRetrieval) {
    availableTools.push(...GRAPH_TOOLS);
    systemPrompt =
      chatMode === "converse"
        ? CONVERSE_GRAPHRAG_SYSTEM_PROMPT
        : GRAPHRAG_SYSTEM_PROMPT;

    // Pre-search: combined vector + graph context injection (cached per conversation)
    const queryKey = buildGraphRagQueryKey(turn.effectiveUserMessage);
    const cachedContext = getGraphRagCachedContext(conversationId, queryKey);
    if (cachedContext) {
      console.log(`[graphrag] Using cached context for ${conversationId}/${queryKey}`);
      systemPrompt +=
        "\n\n--- Enriched Context (graph + vector search) ---\n" +
        cachedContext;
    } else {
      try {
        throwIfAborted(abortSignal);
        onEvent({
          type: "tool_call",
          tool: "graph_search",
          input: { query: turn.effectiveUserMessage },
        });
        const gStart = performance.now();
        const enriched = await graphSearch(turn.effectiveUserMessage, 5, abortSignal, requestId);
        const gMs = performance.now() - gStart;
        trace.setGraphRagPreSearch(gMs);
        const context =
          (enriched as Record<string, string>).formatted_context || "";
        const graphCenter = pickGraphCenterFromEnriched(
          enriched as Record<string, unknown>
        );
        if (context) {
          const truncated = truncateResult(context, MAX_ENRICHED_CONTEXT_CHARS);
          setGraphRagCachedContext(conversationId, queryKey, truncated);
          onEvent({
            type: "tool_result",
            tool: "graph_search",
            preview: "Graph + vector context loaded",
            center: graphCenter,
          });
          systemPrompt +=
            "\n\n--- Enriched Context (graph + vector search) ---\n" +
            truncated;
        }
      } catch {
        if (abortSignal?.aborted) return;
        // Graph service may not be available — proceed without it
      }
    }
  }

  if (turn.skipCorpusRetrieval) {
    systemPrompt +=
      "\n\nTurn instruction: The user is making a simple social check-in. Respond conversationally in 1-3 sentences. " +
      "Do not call tools or cite corpus entries unless they explicitly ask about their writing.";
  }

  if (turn.requireFollowUpQuestion) {
    systemPrompt +=
      "\n\nInteraction instruction: End this reply with exactly one short, specific follow-up question that would improve the next retrieval step. Avoid generic prompts like 'Want me to continue?'.";
  } else if (turn.suppressFollowUpQuestion) {
    systemPrompt +=
      "\n\nInteraction instruction: Do not add a follow-up question in this reply.";
  }

  if (turn.isFollowUpContinuation) {
    systemPrompt +=
      "\n\nContinuation instruction: The user accepted your prior follow-up. Execute it now with deeper, non-redundant evidence. Do not restate the same summary.";
  }

  if (turn.interpretationMode) {
    systemPrompt += buildInterpretationProtocolInstruction(
      turn.requireTemporalContrast
    );
  }

  const historyForPrompt = buildRecentHistory(history).map((message) => ({
    ...message,
  }));

  if (turn.followUpResolution) {
    for (let i = historyForPrompt.length - 1; i >= 0; i--) {
      if (historyForPrompt[i].role === "user") {
        historyForPrompt[i].content = turn.effectiveUserMessage;
        break;
      }
    }
  }

  const messagesForLLM: LLMMessage[] = [
    { role: "system", content: systemPrompt },
    ...(turn.followUpResolution
      ? [
          {
            role: "system" as const,
            content:
              `${turn.followUpResolution.contextHint}\n` +
              `For this turn, treat the user's intent as: "${turn.effectiveUserMessage}".`,
          },
        ]
      : []),
    ...historyForPrompt,
  ];

  let turns = 0;

  while (turns < MAX_TOOL_TURNS) {
    throwIfAborted(abortSignal);
    turns++;

    const llmStart = performance.now();
    const response = turn.streamAssistantText
      ? await llmProvider.chatStreaming({
          messages: messagesForLLM,
          tools: availableTools,
          onToken: (token) => {
            onEvent({ type: "text_delta", content: token });
          },
          signal: abortSignal,
          modelOverride,
        })
      : await llmProvider.chat({
          messages: messagesForLLM,
          tools: availableTools,
          signal: abortSignal,
          modelOverride,
        });
    const llmMs = performance.now() - llmStart;
    trace.addLLMSpan(llmProvider.name, turns, llmMs, response.tool_calls?.length || 0);

    if (response.tool_calls && response.tool_calls.length > 0) {
      // Add assistant message with tool calls to history
      messagesForLLM.push({
        role: "assistant",
        content: response.content || "",
        tool_calls: response.tool_calls,
      });

      const pendingCalls = response.tool_calls.map((tc) => {
        const toolName = tc.function.name;
        const toolArgs = parseToolArguments(tc.function.arguments);
        onEvent({ type: "tool_call", tool: toolName, input: toolArgs });
        return { toolName, toolArgs };
      });

      const toolResults = await Promise.all(
        pendingCalls.map(async ({ toolName, toolArgs }) => {
          const toolResult = await executeTool(toolName, toolArgs, abortSignal, requestId);
          trace.addToolSpan(toolName, toolArgs, toolResult.duration_ms, toolResult.cache_hit, toolResult.result.length);
          return { toolName, toolResult, toolArgs };
        })
      );

      for (const { toolName, toolResult, toolArgs } of toolResults) {
        const preview =
          toolResult.result.length > 200 ? toolResult.result.substring(0, 200) + "..." : toolResult.result;
        onEvent({
          type: "tool_result",
          tool: toolName,
          preview,
          center: pickGraphCenterFromToolCall(toolName, toolArgs),
        });
        messagesForLLM.push({ role: "tool", content: toolResult.result });
      }
    } else {
      // Final text response — streamed live when enabled, otherwise emitted after post-processing.
      let content = response.content || "";
      if (turn.interpretationMode) {
        content = postProcessInterpretiveResponse(
          content,
          turn.requireTemporalContrast
        );
      }
      if (
        turn.requireFollowUpQuestion &&
        !turn.suppressFollowUpQuestion &&
        !responseEndsWithQuestion(content)
      ) {
        const followUp = buildFallbackFollowUpQuestion(
          turn.effectiveUserMessage,
          graphRagMode
        );
        const separator = content.trim().length > 0 ? "\n\n" : "";
        const appended = `${separator}${followUp}`;
        content += appended;
        if (turn.streamAssistantText) {
          onEvent({ type: "text_delta", content: appended });
        }
      }

      if (!turn.streamAssistantText && content) {
        onEvent({ type: "text_delta", content });
      }

      // Store in persistent history
      history.push({ role: "assistant", content });
      if (history.length > MAX_STORED_HISTORY_MESSAGES) {
        history.splice(0, history.length - MAX_STORED_HISTORY_MESSAGES);
      }
      break;
    }
  }

  if (turns >= MAX_TOOL_TURNS) {
    if (abortSignal?.aborted) return;
    onEvent({
      type: "text_delta",
      content:
        "[Reached maximum tool call limit. Here is what I found so far based on the tool results above.]",
    });
  }

  if (abortSignal?.aborted) return;
  trace.setSystemPromptChars(systemPrompt.length);
  trace.finish();
  onEvent({ type: "done" });
}

// --- Health check ---

export async function checkOllamaHealth(): Promise<boolean> {
  return llmProvider.healthCheck();
}
