const REDUNDANCY_STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "has",
  "have",
  "in",
  "is",
  "it",
  "its",
  "of",
  "on",
  "or",
  "that",
  "the",
  "their",
  "there",
  "these",
  "this",
  "to",
  "was",
  "were",
  "with",
  "your",
  "you",
  "my",
  "i",
]);

const TEMPORAL_SIGNAL_RE =
  /\b(over time|shift|earlier|later|before|after|compared|contrast|period)\b/i;
const YEAR_RE = /\b(19|20)\d{2}\b/g;

function normalizeForRedundancy(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_>#~]/g, " ")
    .replace(/\[[^\]]+\]\([^)]+\)/g, " ")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function toRedundancyTokenSet(text: string): Set<string> {
  const normalized = normalizeForRedundancy(text);
  if (!normalized) return new Set<string>();
  const tokens = normalized
    .split(" ")
    .filter(
      (token) => token.length >= 3 && !REDUNDANCY_STOPWORDS.has(token)
    );
  return new Set(tokens);
}

function tokenOverlapRatio(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let overlap = 0;
  for (const token of a) {
    if (b.has(token)) overlap++;
  }
  return overlap / Math.min(a.size, b.size);
}

function isRedundantText(candidate: string, prior: string[]): boolean {
  const candNorm = normalizeForRedundancy(candidate);
  if (!candNorm) return false;
  const candTokens = toRedundancyTokenSet(candidate);

  for (const prev of prior) {
    const prevNorm = normalizeForRedundancy(prev);
    if (!prevNorm) continue;

    if (candNorm === prevNorm) return true;
    if (
      Math.min(candNorm.length, prevNorm.length) >= 50 &&
      (candNorm.includes(prevNorm) || prevNorm.includes(candNorm))
    ) {
      return true;
    }

    const prevTokens = toRedundancyTokenSet(prev);
    if (
      candTokens.size >= 6 &&
      prevTokens.size >= 6 &&
      tokenOverlapRatio(candTokens, prevTokens) >= 0.85
    ) {
      return true;
    }
  }

  return false;
}

function dedupeBulletLines(content: string): string {
  const lines = content.split("\n");
  const kept: string[] = [];
  const seenBullets: string[] = [];

  for (const line of lines) {
    if (/^\s*[-*]\s+/.test(line)) {
      const body = line.replace(/^\s*[-*]\s+/, "").trim();
      if (body && isRedundantText(body, seenBullets)) {
        continue;
      }
      if (body) {
        seenBullets.push(body);
      }
    }
    kept.push(line);
  }

  return kept.join("\n");
}

function dedupeParagraphs(content: string): string {
  const blocks = content
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
  const kept: string[] = [];

  for (const block of blocks) {
    if (isRedundantText(block, kept)) {
      continue;
    }
    kept.push(block);
  }

  return kept.join("\n\n");
}

function countDistinctYears(text: string): number {
  const matches = text.match(YEAR_RE);
  if (!matches) return 0;
  return new Set(matches).size;
}

export function postProcessInterpretiveResponse(
  content: string,
  requireTemporalContrast: boolean
): string {
  let value = content.replace(/\r\n/g, "\n").trim();
  if (!value) return value;

  value = dedupeBulletLines(value);
  value = dedupeParagraphs(value);
  value = value.replace(/\n{3,}/g, "\n\n").trim();

  if (
    requireTemporalContrast &&
    !TEMPORAL_SIGNAL_RE.test(value) &&
    countDistinctYears(value) < 2
  ) {
    value +=
      "\n\nShift over time: I need more distinct dated evidence in this retrieval set to compare periods reliably.";
  }

  return value;
}
