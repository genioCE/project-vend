import type { LLMMessage } from "../llm/index.js";

// --- GraphRAG pre-search cache (per conversation) ---

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

const GRAPHRAG_CACHE_TTL_MS = 2 * 60 * 1000; // 2 minutes
const GRAPHRAG_CACHE_MAX_QUERIES = parsePositiveInt(
  process.env.GRAPHRAG_CACHE_MAX_QUERIES,
  8
);
const graphRagCache = new Map<string, Map<string, { context: string; timestamp: number }>>();

export function buildGraphRagQueryKey(userMessage: string): string {
  return userMessage.trim().toLowerCase().replace(/\s+/g, " ").slice(0, 512);
}

export function getGraphRagCachedContext(
  conversationId: string,
  queryKey: string
): string | undefined {
  const conversationCache = graphRagCache.get(conversationId);
  if (!conversationCache) return undefined;

  const entry = conversationCache.get(queryKey);
  if (!entry) return undefined;

  if (Date.now() - entry.timestamp > GRAPHRAG_CACHE_TTL_MS) {
    conversationCache.delete(queryKey);
    if (conversationCache.size === 0) {
      graphRagCache.delete(conversationId);
    }
    return undefined;
  }

  // Touch entry for simple per-conversation LRU behavior.
  conversationCache.delete(queryKey);
  conversationCache.set(queryKey, entry);
  return entry.context;
}

export function setGraphRagCachedContext(
  conversationId: string,
  queryKey: string,
  context: string
): void {
  let conversationCache = graphRagCache.get(conversationId);
  if (!conversationCache) {
    conversationCache = new Map<string, { context: string; timestamp: number }>();
    graphRagCache.set(conversationId, conversationCache);
  }

  conversationCache.delete(queryKey);
  if (conversationCache.size >= GRAPHRAG_CACHE_MAX_QUERIES) {
    const oldestKey = conversationCache.keys().next().value;
    if (oldestKey) {
      conversationCache.delete(oldestKey);
    }
  }

  conversationCache.set(queryKey, { context, timestamp: Date.now() });
}

// --- Conversation store (with access-time tracking + pruning) ---

interface ConversationEntry {
  messages: LLMMessage[];
  lastAccess: number;
}

const conversations = new Map<string, ConversationEntry>();
const CONVERSATION_TTL_MS = 30 * 60 * 1000; // 30 minutes

export function getHistory(conversationId: string): LLMMessage[] {
  let entry = conversations.get(conversationId);
  if (!entry) {
    entry = { messages: [], lastAccess: Date.now() };
    conversations.set(conversationId, entry);
  }
  entry.lastAccess = Date.now();
  return entry.messages;
}

// Prune stale conversations every 10 minutes
setInterval(() => {
  const now = Date.now();
  let pruned = 0;
  for (const [id, entry] of conversations) {
    if (now - entry.lastAccess > CONVERSATION_TTL_MS) {
      conversations.delete(id);
      graphRagCache.delete(id);
      pruned++;
    }
  }
  if (pruned > 0) {
    console.log(`[prune] Removed ${pruned} stale conversations (${conversations.size} active)`);
  }
}, 10 * 60 * 1000).unref();
