"""Re-label decisions across all corpus entries using Claude.

Uses the v7 decision prompt (active-decisions-only) with configurable model.
Writes entry-level results to a JSONL file, then a second pass converts
to sentence-level training data.

Features:
- Resume capability: skips entries already in output file
- Cost tracking: reports input/output tokens and cost
- Rate limiting: respects rate limits with backoff
- Progress: tqdm progress bar
- Model selection: --model haiku or sonnet (default: sonnet for better instruction following)

Usage:
  python -m scripts.relabel_decisions_haiku \
    --corpus /path/to/corpus \
    --output decision_labels_v7.jsonl \
    [--model sonnet] \
    [--max-entries 0]  # 0 = all entries
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

# Add parent directory for corpus_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.corpus_utils.text_processing import strip_markdown

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ── Prompt v6 ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You identify concrete decisions and commitments in personal journal entries.

A DECISION is a significant choice the writer is ACTIVELY MAKING in the moment of writing. The act of deciding must be happening NOW, in this entry, not reported after the fact.

## THE CRITICAL DISTINCTION — timing:
- "I'm going to quit this job" → DECISION (actively deciding right now in writing)
- "I quit my job last week" → NOT a decision (reporting something already done)
- "I choose to let her go" → DECISION (the choosing is happening as they write)
- "I asked her out at the end of January" → NOT a decision (narrating a past event)
- "I'm committing 5-10 hours a week to this" → DECISION (locking in right now)
- "I went to Best Buy and bought a MacBook" → NOT a decision (telling a story about what happened)

The writing IS the decision — the writer is processing and committing to something through the act of writing itself.

## What IS a decision (include these):
- Active commitments being made NOW: "I'm going to quit", "I choose to let her go"
- Firm plans being locked in: "I'll text Carly today to see if she wants brunch"
- Strategic pivots declared in the moment: "Drop the startup and sit on the sidelines"
- Deliberate surrenders happening now: "I surrender this feeling"
- Choosing to step through a process: "Just going to step through this interview process"

## What is NOT a decision (exclude ALL of these):
- REPORTING PAST ACTIONS: "I bought a MacBook", "I asked her out", "I finished 2 pieces", "I went to a meeting", "I showed up and shared my inventory", "I ordered nigiri", "I started building a document organizer" — these are narratives about things already done, NOT decisions being made in writing
- REPORTING PAST DECISIONS: "I decided to call Luke", "she decided to go out with me", "when I decided I needed a change", "I remember when I decided to leave that job" — recounting a decision already made is NOT a current decision
- Routine daily plans: "I'm going to go stand up that RAG model today"
- Casual social plans/invitations: "I'm going to invite her to madder this friday"
- Meta-reflections about decisions: "it all happened as soon as I made the decision"
- ANY "gotta"/"need to"/"should" language: "I gotta stop smoking weed", "I gotta let go"
- Aspirations/goals/wants: "reducing smoking is a goal", "I want to break 175k"
- Predictions/future projections: "we will be in a relationship in less than 30 days"
- Affirmations/declarations of current state: "I am abstaining from drugs", "I'm on Day 7", "I haven't drank in over a year", "I've chosen this life" (past tense = reporting, not deciding)
- Prayers/wishes: "I pray for good news"
- Observations/reflections: "I think 100mg is my sweet spot", "Removing rosin has been profound"
- Telling OTHERS what to do: "I told him they need to give each other space"
- Questions/deliberation: "Should I just commit to that?", "I think I might ask Adina"
- Reporting events: "I reconnected with Kearby today", "Sending Ruth that amends opened me up"
- Vague directives: "Just eat edibles"
- Task lists/logistics: "get back to the gym", "Office by 630am"
- Hedged/tentative: "I should probably go to a meeting tonight"

## The key test:
Is the writer ACTIVELY DECIDING right now as they write this sentence? Or are they TELLING YOU about something that already happened?
- "I'm going to quit my 120k job" → YES, deciding now
- "I just quit a 120k job" → NO, reporting what happened
- "I choose to not leave my brother behind" → YES, active choosing
- "I got to chair a meeting yesterday" → NO, reporting
- "I will find a team and build a proof of concept ASAP" → YES, committing now
- "I went over all my AI ideas with ChatGPT" → NO, narrating

## Instructions:
Return a JSON array of the EXACT sentences containing decisions. Return [] if none. Be VERY selective — most entries have 0-1 real decisions. Many entries have zero.

Return strict JSON only."""

USER_TEMPLATE = """Entry ID: {entry_id}

Entry text:
{entry_text}
"""


def find_all_corpus_files(corpus_dir: Path) -> list[tuple[str, Path]]:
    """Find all .md corpus files, returning (entry_id, path) pairs."""
    entries = []
    for year_dir in sorted(corpus_dir.iterdir()):
        if not year_dir.is_dir() or year_dir.name.startswith("."):
            continue
        for md_file in sorted(year_dir.glob("*.md")):
            if md_file.name.startswith("._"):
                continue
            entry_id = md_file.stem
            entries.append((entry_id, md_file))
    return entries


def load_already_done(output_path: Path) -> set[str]:
    """Load entry IDs already processed from output file."""
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["entry_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


MODEL_MAP = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-20250514",
}

# Cost per million tokens by model
MODEL_COSTS = {
    "haiku": {"input": 0.80, "output": 4.00},
    "sonnet": {"input": 3.00, "output": 15.00},
}


def call_model(
    client: anthropic.Anthropic,
    entry_id: str,
    entry_text: str,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 3,
) -> tuple[list[str], int, int]:
    """Send entry to Claude. Returns (decisions, input_tokens, output_tokens)."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(
                    entry_id=entry_id, entry_text=entry_text
                )}],
            )
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            result = json.loads(content)
            if isinstance(result, list):
                return [str(s) for s in result], input_tokens, output_tokens
            return [], input_tokens, output_tokens

        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except json.JSONDecodeError:
            # Haiku sometimes wraps empty arrays in explanation text
            if "[]" in content:
                return [], input_tokens, output_tokens
            if attempt < max_retries - 1:
                print(f"  JSON parse error for {entry_id}, retrying...")
                time.sleep(1)
            else:
                print(f"  WARNING: Failed to parse JSON for {entry_id}: {content[:100]}")
                return [], input_tokens, output_tokens
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Error for {entry_id}: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ERROR: Failed after {max_retries} attempts for {entry_id}: {e}")
                return [], 0, 0

    return [], 0, 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-label decisions with v7 prompt")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory")
    parser.add_argument("--output", default="decision_labels_v7.jsonl", help="Output JSONL file")
    parser.add_argument("--model", default="sonnet", choices=["haiku", "sonnet"],
                        help="Model to use (default: sonnet for better instruction following)")
    parser.add_argument("--max-entries", type=int, default=0, help="Max entries to process (0 = all)")
    parser.add_argument("--env-file", default=None, help="Path to .env file with ANTHROPIC_API_KEY")
    args = parser.parse_args()

    # Load API key
    if args.env_file:
        env_path = Path(args.env_file)
    else:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                break

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: No ANTHROPIC_API_KEY found")
        sys.exit(1)

    client = anthropic.Anthropic()
    corpus_dir = Path(args.corpus)
    output_path = Path(args.output)
    model_id = MODEL_MAP[args.model]
    costs = MODEL_COSTS[args.model]
    print(f"Using model: {args.model} ({model_id})")

    # Find all corpus files
    all_entries = find_all_corpus_files(corpus_dir)
    print(f"Found {len(all_entries)} corpus entries")

    # Load already-done entries for resume
    done = load_already_done(output_path)
    if done:
        print(f"Resuming: {len(done)} entries already processed")

    # Filter to remaining entries
    remaining = [(eid, path) for eid, path in all_entries if eid not in done]
    if args.max_entries > 0:
        remaining = remaining[:args.max_entries]

    print(f"Processing {len(remaining)} entries...")

    # Process entries
    total_input_tokens = 0
    total_output_tokens = 0
    total_decisions = 0
    entries_with_decisions = 0
    errors = 0

    with open(output_path, "a") as f:
        for entry_id, file_path in tqdm(remaining, desc="Labeling"):
            try:
                raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"  ERROR reading {file_path}: {e}")
                errors += 1
                continue

            text = strip_markdown(raw_text)
            normalized = " ".join(text.split())
            if not normalized.strip():
                # Write empty result for empty entries
                rec = {"entry_id": entry_id, "decisions": []}
                f.write(json.dumps(rec) + "\n")
                continue

            decisions, in_tok, out_tok = call_model(client, entry_id, normalized, model=model_id)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            rec = {
                "entry_id": entry_id,
                "decisions": decisions,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

            total_decisions += len(decisions)
            if decisions:
                entries_with_decisions += 1

    # Final report
    processed = len(remaining) - errors
    input_cost = total_input_tokens / 1_000_000 * costs["input"]
    output_cost = total_output_tokens / 1_000_000 * costs["output"]
    total_cost = input_cost + output_cost

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Processed:    {processed} entries ({errors} errors)")
    print(f"  Decisions:    {total_decisions} total")
    print(f"  With decisions: {entries_with_decisions}/{processed} entries "
          f"({100*entries_with_decisions/max(processed,1):.0f}%)")
    print(f"  Avg decisions/entry: {total_decisions/max(processed,1):.2f}")
    print(f"  Input tokens:  {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Model: {args.model}")
    print(f"  Cost: ${total_cost:.2f} "
          f"(input: ${input_cost:.2f}, output: ${output_cost:.2f})")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
