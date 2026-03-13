/**
 * Agent identity vectors — Project Vend.
 *
 * Each agent has a natural language identity description that gets embedded
 * into the same 768-dim vector space as event fragments. The gravity engine
 * computes cosine similarity between fragments and these identities to
 * determine which agents activate.
 */

export interface AgentIdentity {
  name: string;
  description: string;
  is_meta: boolean;
  always_active: boolean;
}

export const AGENTS: AgentIdentity[] = [
  // ── Core Agents ─────────────────────────────────────────────────
  {
    name: "inventory_agent",
    description:
      "Manages product stock levels in vending machines. Monitors inventory " +
      "depletion rates, predicts demand patterns based on time of day, day of " +
      "week, and campus context. Triggers restocking workflows when stock falls " +
      "below thresholds. Tracks expiration dates for perishable items and flags " +
      "waste risk. Handles supply chain coordination, delivery scheduling, and " +
      "slot allocation. Activated by stock changes, low inventory alerts, " +
      "restock needs, waste risk, demand forecasting, and product expiration.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "customer_agent",
    description:
      "Handles the customer-facing experience at the vending machine. Generates " +
      "product recommendations based on browsing behavior, purchase history, and " +
      "available inventory. Manages the purchase flow from product selection " +
      "through payment completion. Provides conversational support and handles " +
      "customer queries. Accommodates accessibility needs including screen reader " +
      "support and language preferences. Personalizes the experience based on " +
      "student ID recognition and past interactions. Activated by customer " +
      "presence, browsing behavior, purchase events, abandoned transactions, " +
      "interaction quality signals, and accessibility requests.",
    is_meta: false,
    always_active: false,
  },
  {
    name: "pricing_agent",
    description:
      "Dynamically adjusts product pricing based on real-time demand signals, " +
      "current inventory levels, time of day, day of week, and competitive " +
      "positioning. Runs A/B tests on pricing strategies to optimize revenue. " +
      "Creates bundle deals, implements expiration-based discounts to reduce " +
      "waste, and manages promotional campaigns. Integrates with campus events " +
      "and student discount programs. Analyzes pricing performance and adjusts " +
      "strategy based on historical data. Activated by demand shifts, inventory " +
      "changes, time-based triggers, promotional events, pricing performance " +
      "reviews, and revenue optimization opportunities.",
    is_meta: false,
    always_active: false,
  },

  // ── Meta / Dashboard Agent ──────────────────────────────────────
  {
    name: "analytics_agent",
    description:
      "Collects and reports on system-wide metrics, agent decisions, and " +
      "vending machine performance. Generates dashboards, logs agent decision " +
      "chains, and tracks KPIs like revenue, waste rate, customer satisfaction, " +
      "and restock efficiency. Provides operational summaries and alerts on " +
      "anomalies. Activated by reporting requests, performance review triggers, " +
      "and system health monitoring.",
    is_meta: true,
    always_active: false,
  },
];

export const AGENT_NAMES = AGENTS.map((a) => a.name);

export const ALWAYS_ACTIVE_AGENTS = new Set(
  AGENTS.filter((a) => a.always_active).map((a) => a.name)
);

export const META_AGENTS = new Set(
  AGENTS.filter((a) => a.is_meta).map((a) => a.name)
);

// ─── Identity Vector Cache ────────────────────────────────────────

let _cachedVectors: number[][] | null = null;

/**
 * Get embedded identity vectors for all agents.
 * Calls the embeddings service once, then caches in memory.
 */
export async function getIdentityVectors(
  embedFn: (texts: string[]) => Promise<number[][]>,
  signal?: AbortSignal
): Promise<number[][]> {
  if (_cachedVectors) return _cachedVectors;

  const descriptions = AGENTS.map((a) => a.description);
  _cachedVectors = await embedFn(descriptions);
  return _cachedVectors;
}
