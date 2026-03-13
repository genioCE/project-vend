/**
 * Event decomposition for Project Vend.
 *
 * Decomposes vending system events into typed semantic fragments:
 * Product, Signal, Temporal, Context, Customer, Machine.
 *
 * Primarily rule-based (no LLM cost for standard events).
 * Falls back to Claude Haiku for ambiguous natural language inputs.
 */

import type {
  DecompositionResult,
  Fragment,
  FragmentType,
  ExtractedParams,
} from "./types.js";

// ─── Known vocabulary ─────────────────────────────────────────────

const SIGNAL_KEYWORDS = new Map<string, string>([
  ["low_stock", "low_stock"],
  ["out_of_stock", "out_of_stock"],
  ["stock_updated", "stock_updated"],
  ["restock_needed", "restock_needed"],
  ["expiring_soon", "expiring_soon"],
  ["expired", "expired"],
  ["purchase_completed", "purchase_completed"],
  ["purchase_failed", "purchase_failed"],
  ["payment_failed", "payment_failed"],
  ["price_changed", "price_changed"],
  ["promotion_started", "promotion_started"],
  ["promotion_ended", "promotion_ended"],
  ["machine_offline", "machine_offline"],
  ["machine_online", "machine_online"],
  ["slot_jammed", "slot_jammed"],
  ["temperature_alert", "temperature_alert"],
  ["maintenance_needed", "maintenance_needed"],
]);

const TEMPORAL_PATTERNS: Array<{ pattern: RegExp; label: string }> = [
  { pattern: /morning|6am|7am|8am|9am|10am/i, label: "morning" },
  { pattern: /afternoon|noon|12pm|1pm|2pm|3pm|4pm/i, label: "afternoon_peak" },
  { pattern: /evening|5pm|6pm|7pm|8pm|9pm/i, label: "evening" },
  { pattern: /night|late|10pm|11pm|midnight/i, label: "night" },
  { pattern: /weekend|saturday|sunday/i, label: "weekend" },
  { pattern: /weekday|monday|tuesday|wednesday|thursday|friday/i, label: "weekday" },
  { pattern: /finals?\s*week/i, label: "finals_week" },
  { pattern: /peak\s*hours?/i, label: "peak_hours" },
  { pattern: /off[\s-]?peak/i, label: "off_peak" },
  { pattern: /end[\s-]?of[\s-]?day/i, label: "end_of_day" },
  { pattern: /start[\s-]?of[\s-]?day/i, label: "start_of_day" },
];

const CONTEXT_KEYWORDS = new Set([
  "campus_event",
  "homecoming",
  "graduation",
  "orientation",
  "football_game",
  "basketball_game",
  "finals_week",
  "spring_break",
  "holiday",
  "weather_hot",
  "weather_cold",
  "weather_rain",
  "maintenance_window",
  "power_outage",
]);

const CUSTOMER_SIGNALS = new Set([
  "browsing",
  "abandoned_cart",
  "repeat_buyer",
  "first_visit",
  "student_id_scanned",
  "payment_initiated",
  "help_requested",
  "accessibility_mode",
]);

// ─── Rule-Based Decomposition ─────────────────────────────────────

/**
 * Decompose a structured vending system event into typed fragments.
 *
 * Handles both structured event objects (JSON) and free-text descriptions.
 */
export function decomposeEvent(
  input: string | Record<string, unknown>
): DecompositionResult {
  const fragments: Fragment[] = [];
  const extracted: ExtractedParams = {
    products: [],
    signals: [],
    machine_ids: [],
    date_ranges: [],
    context_tags: [],
    search_query: "",
  };

  // Normalize input to string for pattern matching
  const text =
    typeof input === "string" ? input : JSON.stringify(input).toLowerCase();
  const textLower = text.toLowerCase();

  extracted.search_query = typeof input === "string" ? input : text;

  // ── Extract structured fields if JSON event ──
  if (typeof input === "object" && input !== null) {
    const event = input as Record<string, unknown>;

    // Product
    if (event.product && typeof event.product === "string") {
      fragments.push({ type: "product", text: event.product });
      extracted.products.push(event.product);
    }
    if (event.product_id && typeof event.product_id === "string") {
      fragments.push({ type: "product", text: event.product_id });
      extracted.products.push(event.product_id);
    }

    // Signal / event type
    if (event.event_type && typeof event.event_type === "string") {
      fragments.push({ type: "signal", text: event.event_type });
      extracted.signals.push(event.event_type);
    }
    if (event.signal && typeof event.signal === "string") {
      fragments.push({ type: "signal", text: event.signal });
      extracted.signals.push(event.signal);
    }

    // Machine
    if (event.machine_id && typeof event.machine_id === "string") {
      fragments.push({ type: "machine", text: event.machine_id });
      extracted.machine_ids.push(event.machine_id);
    }

    // Context
    const context = event.context as Record<string, unknown> | undefined;
    if (context && typeof context === "object") {
      if (Array.isArray(context.campus_events)) {
        for (const ce of context.campus_events) {
          fragments.push({ type: "context", text: String(ce) });
          extracted.context_tags.push(String(ce));
        }
      }
      if (context.time_of_day && typeof context.time_of_day === "string") {
        fragments.push({ type: "temporal", text: context.time_of_day });
      }
      if (context.day_type && typeof context.day_type === "string") {
        fragments.push({ type: "temporal", text: context.day_type });
      }
      if (context.weather && typeof context.weather === "string") {
        fragments.push({ type: "context", text: `weather_${context.weather}` });
        extracted.context_tags.push(`weather_${context.weather}`);
      }
    }

    // Customer
    if (event.customer_action && typeof event.customer_action === "string") {
      fragments.push({ type: "customer", text: event.customer_action });
    }
    if (event.customer_id && typeof event.customer_id === "string") {
      fragments.push({ type: "customer", text: "identified_customer" });
    }
  }

  // ── Free-text pattern matching (supplements structured extraction) ──

  // Signals from text
  for (const [keyword, label] of SIGNAL_KEYWORDS) {
    if (textLower.includes(keyword.replace(/_/g, " ")) || textLower.includes(keyword)) {
      if (!extracted.signals.includes(label)) {
        fragments.push({ type: "signal", text: label });
        extracted.signals.push(label);
      }
    }
  }

  // Temporal patterns
  for (const { pattern, label } of TEMPORAL_PATTERNS) {
    if (pattern.test(text)) {
      const existing = fragments.find(
        (f) => f.type === "temporal" && f.text === label
      );
      if (!existing) {
        fragments.push({ type: "temporal", text: label });
      }
    }
  }

  // Context keywords
  for (const keyword of CONTEXT_KEYWORDS) {
    if (
      textLower.includes(keyword.replace(/_/g, " ")) ||
      textLower.includes(keyword)
    ) {
      if (!extracted.context_tags.includes(keyword)) {
        fragments.push({ type: "context", text: keyword });
        extracted.context_tags.push(keyword);
      }
    }
  }

  // Customer signals
  for (const signal of CUSTOMER_SIGNALS) {
    if (
      textLower.includes(signal.replace(/_/g, " ")) ||
      textLower.includes(signal)
    ) {
      const existing = fragments.find(
        (f) => f.type === "customer" && f.text === signal
      );
      if (!existing) {
        fragments.push({ type: "customer", text: signal });
      }
    }
  }

  // ── Primary mass: default to first signal, fallback to first fragment ──
  const signalIdx = fragments.findIndex((f) => f.type === "signal");
  const primaryMassIndex = signalIdx >= 0 ? signalIdx : 0;

  return {
    fragments,
    primary_mass_index: primaryMassIndex,
    reasoning: `Rule-based decomposition: ${fragments.length} fragments extracted`,
    extracted,
    token_usage: {
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      source: "rule-based",
    },
  };
}

/**
 * Embed all fragments and the full event description.
 * Populates the `embedding` field on each fragment and sets `query_embedding`.
 */
export async function embedDecomposition(
  decomposition: DecompositionResult,
  fullEventText: string,
  embedFn: (texts: string[]) => Promise<number[][]>
): Promise<void> {
  const textsToEmbed = [
    ...decomposition.fragments.map((f) => f.text),
    fullEventText,
  ];

  const embeddings = await embedFn(textsToEmbed);

  // Assign fragment embeddings
  for (let i = 0; i < decomposition.fragments.length; i++) {
    decomposition.fragments[i].embedding = embeddings[i];
  }

  // Last embedding is the full event
  decomposition.query_embedding = embeddings[embeddings.length - 1];
}
