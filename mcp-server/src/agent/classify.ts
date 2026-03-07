import type { ChatMode } from "../agent.js";

// --- Regex patterns ---

const FOLLOW_UP_BLOCK_PATTERNS = [
  /\bno follow[- ]?up\b/i,
  /\bdon['']?t ask\b/i,
  /\bdo not ask\b/i,
  /\bone sentence\b/i,
  /\bone line\b/i,
  /\byes or no\b/i,
  /\bjust answer\b/i,
  /\bonly answer\b/i,
  /\bexactly\s+\d+/i,
  /\b\d+\s+bullets?\b/i,
];

const FOLLOW_UP_ENABLE_PATTERNS = [
  /^\s*why\b/i,
  /^\s*how\b/i,
  /\bhelp me understand\b/i,
  /\bwhat do you notice\b/i,
  /\bpatterns?\b/i,
  /\bthemes?\b/i,
  /\bevolv\w*\b/i,
  /\brelationships?\b/i,
  /\bconnected\b/i,
  /\bmeaning\b/i,
  /\bvoice\b/i,
  /\btone\b/i,
  /\bconflict\b/i,
  /\btension\b/i,
  /\buncertain\b/i,
];

const INTERPRETIVE_QUERY_PATTERNS = [
  /\bwhat does\b/i,
  /\bwhat do(es)? .* mean\b/i,
  /\bmean(ing)?\b/i,
  /\binterpret\w*\b/i,
  /\bassociated\b/i,
  /\brelationship(s)?\b/i,
  /\bpatterns?\b/i,
  /\bthemes?\b/i,
  /\barchetype(s)?\b/i,
  /\bcore\b/i,
  /\bprimary\b/i,
  /\bwho am i\b/i,
  /\bhow do i\b/i,
];

const TEMPORAL_CONTRAST_PATTERNS = [
  /\bover time\b/i,
  /\bevolv\w*\b/i,
  /\bchanged?\b/i,
  /\bshift\w*\b/i,
  /\bcompare\b/i,
  /\bvs\.?\b/i,
  /\bbefore\b.*\bafter\b/i,
  /\bearlier\b.*\blater\b/i,
  /\bperiod\b/i,
];

const YEAR_RE = /\b(19|20)\d{2}\b/g;

const DIRECTIVE_QUERY_RE =
  /^(map|show|list|give|summarize|compare|find|extract|trace|break down|analyze)\b/i;

const SHORT_AFFIRMATIVE_RE =
  /^(yes|yeah|yep|yup|sure|ok|okay|please|please do|go ahead|do it|sounds good|let['']?s do it|lets do it)[.! ]*$/i;

const SHORT_NEGATIVE_RE =
  /^(no|nope|nah|not now|skip|pass|no thanks|don't|do not)[.! ]*$/i;

const CORPUS_RETRIEVAL_CUE_RE =
  /\b(journal|writing|wrote|entries|entry|pattern|patterns|theme|themes|history|compare|evolution|over time|date|dates|search|find|recurring|archetype|decision|concept|people|relationship)\b/i;

const SOCIAL_TURN_RE =
  /\b(hi|hey|hello|how are you|how's it going|whats up|what's up|good morning|good afternoon|good evening|thanks|thank you)\b/i;

// --- Classification functions ---

function isSocialCheckInTurn(userMessage: string): boolean {
  const text = userMessage.trim().toLowerCase();
  if (!text) return false;
  if (CORPUS_RETRIEVAL_CUE_RE.test(text)) return false;

  const words = text.split(/\s+/).filter(Boolean);
  if (words.length > 12) return false;
  return SOCIAL_TURN_RE.test(text);
}

function shouldSkipCorpusRetrievalForTurn(
  userMessage: string,
  chatMode: ChatMode
): boolean {
  if (chatMode !== "converse") return false;
  return isSocialCheckInTurn(userMessage);
}

function shouldAskFollowUpQuestion(userMessage: string): boolean {
  const text = userMessage.trim();
  if (!text) return false;
  if (FOLLOW_UP_BLOCK_PATTERNS.some((pattern) => pattern.test(text))) return false;
  if (DIRECTIVE_QUERY_RE.test(text) && text.split(/\s+/).length >= 5) return false;
  if (FOLLOW_UP_ENABLE_PATTERNS.some((pattern) => pattern.test(text))) return true;
  if (text.endsWith("?") && text.split(/\s+/).length >= 6) return true;
  return false;
}

function shouldSuppressFollowUpQuestion(userMessage: string): boolean {
  const text = userMessage.trim();
  if (!text) return false;
  return FOLLOW_UP_BLOCK_PATTERNS.some((pattern) => pattern.test(text));
}

function shouldUseInterpretationProtocol(
  userMessage: string,
  graphRagMode: boolean,
  chatMode: ChatMode
): boolean {
  if (chatMode === "converse") {
    const text = userMessage.trim();
    if (!text) return false;
    return (
      INTERPRETIVE_QUERY_PATTERNS.some((pattern) => pattern.test(text)) ||
      requiresTemporalContrast(text)
    );
  }
  if (graphRagMode) return true;
  const text = userMessage.trim();
  if (!text) return false;
  return INTERPRETIVE_QUERY_PATTERNS.some((pattern) => pattern.test(text));
}

function requiresTemporalContrast(userMessage: string): boolean {
  const text = userMessage.trim();
  if (!text) return false;
  if (TEMPORAL_CONTRAST_PATTERNS.some((pattern) => pattern.test(text))) {
    return true;
  }
  const years = text.match(YEAR_RE);
  return years ? new Set(years).size >= 2 : false;
}

// --- Follow-up resolution ---

export interface FollowUpResolution {
  kind: "affirmative" | "negative";
  resolvedQuery: string;
  contextHint: string;
}

function isShortAffirmative(userMessage: string): boolean {
  return SHORT_AFFIRMATIVE_RE.test(userMessage.trim());
}

function isShortNegative(userMessage: string): boolean {
  return SHORT_NEGATIVE_RE.test(userMessage.trim());
}

function extractTrailingQuestion(text: string): string | undefined {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  for (let i = lines.length - 1; i >= 0; i--) {
    const candidate = lines[i]
      .replace(/^[-*]\s*/, "")
      .replace(/^next steps:\s*/i, "")
      .trim();
    if (candidate.endsWith("?") && candidate.length <= 240) {
      return candidate;
    }
  }

  const matches = text.match(/[^?\n]{4,240}\?/g);
  if (!matches || matches.length === 0) return undefined;
  return matches[matches.length - 1].trim();
}

function toContinuationQuery(question: string): string {
  let value = question.trim();
  value = value.replace(/^["'`]+|["'`]+$/g, "");
  value = value.replace(/^next steps:\s*/i, "");
  value = value.replace(/^would you like to\s+/i, "");
  value = value.replace(/^do you want (?:me )?to\s+/i, "");
  value = value.replace(/^want me to\s+/i, "");
  value = value.replace(/^should i\s+/i, "");
  value = value.replace(/^would it help to\s+/i, "");
  value = value.replace(/\?+$/g, "");
  value = value.replace(/\byour\b/gi, "my");
  value = value.replace(/\byou\b/gi, "I");
  value = value.trim();
  if (!value) return question.trim();
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function resolveFollowUpReply(
  userMessage: string,
  previousAssistantContent?: string
): FollowUpResolution | undefined {
  const text = userMessage.trim();
  if (!text) return undefined;

  const embedded = text.match(
    /^(.*\?)\s*(yes|yeah|yep|yup|sure|ok|okay|please|go ahead|do it|let['']?s do it|lets do it)[.! ]*$/i
  );
  if (embedded) {
    const priorQuestion = embedded[1].trim();
    const resolved = toContinuationQuery(priorQuestion);
    return {
      kind: "affirmative",
      resolvedQuery: resolved,
      contextHint:
        `The user's latest message confirms this follow-up request: "${priorQuestion}". ` +
        `Proceed with that request now and add new detail rather than repeating prior wording.`,
    };
  }

  if (!previousAssistantContent) return undefined;
  const lastQuestion = extractTrailingQuestion(previousAssistantContent);
  if (!lastQuestion) return undefined;

  if (isShortAffirmative(text)) {
    const resolved = toContinuationQuery(lastQuestion);
    return {
      kind: "affirmative",
      resolvedQuery: resolved,
      contextHint:
        `The user replied "${text}" to your prior follow-up question "${lastQuestion}". ` +
        `Treat this as confirmation and continue with that analysis. Do not repeat the same summary.`,
    };
  }

  if (isShortNegative(text)) {
    return {
      kind: "negative",
      resolvedQuery: text,
      contextHint:
        `The user declined your prior follow-up question "${lastQuestion}". ` +
        `Do not continue that branch; briefly acknowledge and invite a new direction.`,
    };
  }

  return undefined;
}

// --- Exported helpers used in agent.ts finalization ---

export function responseEndsWithQuestion(content: string): boolean {
  const trimmed = content.trim();
  if (!trimmed) return false;
  const lines = trimmed
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) return false;
  return lines[lines.length - 1].endsWith("?");
}

export function buildFallbackFollowUpQuestion(
  userMessage: string,
  graphRagMode: boolean
): string {
  const text = userMessage.toLowerCase();
  if (/shift|change|evolv|over time|this year/.test(text)) {
    return "Want me to break that into time periods with 3 dated examples?";
  }
  if (/relationship|connected|associated|linked|co-occur/.test(text)) {
    return "Want me to map the strongest relationships and where they diverge?";
  }
  if (/fear to action|from .* to .*|transition|flow/.test(text)) {
    return "Want me to pull explicit transition excerpts with dates?";
  }
  if (/people|person|who/.test(text)) {
    return "Want me to focus on the top people and how their context differs over time?";
  }
  if (graphRagMode) {
    return "Want me to drill into one part of this with dated evidence?";
  }
  return "Want me to narrow this to one specific angle next?";
}

export function buildInterpretationProtocolInstruction(
  requireTemporalContrast: boolean
): string {
  const lines = [
    "Interpretation protocol:",
    "- Build an internal claim->evidence map first (do not expose JSON or chain-of-thought).",
    "- Write 2-5 non-overlapping claims only.",
    "- Anchor each claim with at least one dated reference from retrieved evidence when available.",
    "- If date evidence is missing or weak, say so explicitly instead of inferring.",
    "- Avoid repeated summaries: do not restate the opening claim in the closing lines.",
    "- Prefer concise evidence bullets where each bullet adds a distinct point.",
  ];

  if (requireTemporalContrast) {
    lines.push(
      "- Include a 'Shift over time' comparison with at least two periods when dated evidence supports it."
    );
  }

  return `\n\n${lines.join("\n")}`;
}

// --- Turn classification facade ---

export interface TurnClassification {
  effectiveUserMessage: string;
  followUpResolution: FollowUpResolution | undefined;
  skipCorpusRetrieval: boolean;
  interpretationMode: boolean;
  requireTemporalContrast: boolean;
  streamAssistantText: boolean;
  requireFollowUpQuestion: boolean;
  suppressFollowUpQuestion: boolean;
  isFollowUpContinuation: boolean;
  isFollowUpDecline: boolean;
}

export function classifyTurn(
  userMessage: string,
  previousAssistantContent: string | undefined,
  graphRagMode: boolean,
  chatMode: ChatMode,
): TurnClassification {
  const followUpResolution = resolveFollowUpReply(
    userMessage,
    previousAssistantContent
  );
  const effectiveUserMessage = followUpResolution?.resolvedQuery || userMessage;
  const skipCorpusRetrieval = shouldSkipCorpusRetrievalForTurn(
    effectiveUserMessage,
    chatMode
  );
  const interpretationMode = shouldUseInterpretationProtocol(
    effectiveUserMessage,
    graphRagMode,
    chatMode
  );
  const reqTemporalContrast = requiresTemporalContrast(effectiveUserMessage);
  const streamAssistantText = !interpretationMode;
  const isFollowUpContinuation = followUpResolution?.kind === "affirmative";
  const isFollowUpDecline = followUpResolution?.kind === "negative";
  const requireFollowUpQuestion =
    chatMode === "classic" &&
    !isFollowUpContinuation &&
    !isFollowUpDecline &&
    !skipCorpusRetrieval &&
    shouldAskFollowUpQuestion(effectiveUserMessage);
  const suppressFollowUpQuestion =
    chatMode === "converse" ||
    shouldSuppressFollowUpQuestion(userMessage) ||
    isFollowUpContinuation ||
    isFollowUpDecline;

  return {
    effectiveUserMessage,
    followUpResolution,
    skipCorpusRetrieval,
    interpretationMode,
    requireTemporalContrast: reqTemporalContrast,
    streamAssistantText,
    requireFollowUpQuestion,
    suppressFollowUpQuestion,
    isFollowUpContinuation,
    isFollowUpDecline,
  };
}
