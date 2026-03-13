/**
 * Parallel agent dispatch — Project Vend.
 *
 * Maps activated agent names to actual agent handler functions.
 * Each agent runs concurrently with per-agent timeouts.
 *
 * TODO: Implement actual agent logic. Currently returns placeholder responses.
 */

import type { ActivatedAgent, ExtractedParams, AgentResult } from "./types.js";

const AGENT_TIMEOUT_MS = 10_000;

type AgentHandler = (
  params: ExtractedParams,
  eventDescription: string,
  signal?: AbortSignal
) => Promise<string>;

const AGENT_TABLE: Record<string, AgentHandler> = {
  inventory_agent: async (params, event) => {
    // TODO: Implement inventory agent logic
    return JSON.stringify({
      agent: "inventory",
      action: "evaluate_stock",
      products: params.products,
      signals: params.signals,
      recommendation: "placeholder — implement inventory logic",
    });
  },

  customer_agent: async (params, event) => {
    // TODO: Implement customer interaction agent logic
    return JSON.stringify({
      agent: "customer",
      action: "evaluate_interaction",
      context: params.context_tags,
      recommendation: "placeholder — implement customer logic",
    });
  },

  pricing_agent: async (params, event) => {
    // TODO: Implement pricing agent logic
    return JSON.stringify({
      agent: "pricing",
      action: "evaluate_pricing",
      products: params.products,
      signals: params.signals,
      recommendation: "placeholder — implement pricing logic",
    });
  },

  analytics_agent: async (params, event) => {
    // TODO: Implement analytics/dashboard agent
    return JSON.stringify({
      agent: "analytics",
      action: "log_event",
      event_summary: event.slice(0, 200),
      recommendation: "placeholder — implement analytics logic",
    });
  },
};

export async function dispatchAgents(
  activated: ActivatedAgent[],
  params: ExtractedParams,
  eventDescription: string,
  signal?: AbortSignal
): Promise<AgentResult[]> {
  const promises = activated.map(async (agent): Promise<AgentResult> => {
    const handler = AGENT_TABLE[agent.name];
    if (!handler) {
      return {
        agent: agent.name,
        result: "",
        composite_score: agent.composite_score,
        duration_ms: 0,
        error: `no handler for ${agent.name}`,
      };
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), AGENT_TIMEOUT_MS);

    let mergedSignal: AbortSignal;
    if (signal) {
      mergedSignal = AbortSignal.any([signal, controller.signal]);
    } else {
      mergedSignal = controller.signal;
    }

    const start = performance.now();
    try {
      const result = await handler(params, eventDescription, mergedSignal);
      return {
        agent: agent.name,
        result,
        composite_score: agent.composite_score,
        duration_ms: performance.now() - start,
      };
    } catch (err) {
      return {
        agent: agent.name,
        result: "",
        composite_score: agent.composite_score,
        duration_ms: performance.now() - start,
        error: String(err),
      };
    } finally {
      clearTimeout(timer);
    }
  });

  const settled = await Promise.allSettled(promises);
  return settled.map((r) =>
    r.status === "fulfilled"
      ? r.value
      : {
          agent: "unknown",
          result: "",
          composite_score: 0,
          duration_ms: 0,
          error: String(r.reason),
        }
  );
}
