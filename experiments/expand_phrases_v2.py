#!/usr/bin/env python3
"""
expand_phrases_v2.py — Generate 8 trigger phrases per skill with deliberate
implicit coverage.

Required mix per skill (8 total):
  2 explicit    — directly names the tool/format/skill
  2 paraphrase  — describes what it does without casual phrasing
  2 colloquial  — casual/informal phrasing of the same need
  2 implicit    — goal/outcome only, NO tool or skill name at all

The implicit phrases are the key addition: they bridge the gap for prompts
that describe an outcome without naming a tool (the dominant MISS_STAGE1
failure mode).

Output field: trigger_phrases_v2 (does NOT overwrite trigger_phrases)
Progress: data/expand_v2_progress.json  (incremental; re-run is free)
Output: data/skills_corpus_expanded.json (trigger_phrases_v2 field added)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT          = Path(__file__).resolve().parent.parent
EXPANDED_PATH = ROOT / "data" / "skills_corpus_expanded.json"
PROGRESS_PATH = ROOT / "data" / "expand_v2_progress.json"
COST_PATH     = ROOT / "data" / ".cost_tracker.json"

load_dotenv(ROOT / ".env")

MODEL      = "openai/gpt-4o-mini"
IN_PRICE   = 0.15 / 1_000_000
OUT_PRICE  = 0.60 / 1_000_000
HALT_AT    = 3.00

SYSTEM_PROMPT = """\
You are a test-case generator for a skill router.
Given a skill name and description, output a JSON array of exactly 8 short user \
requests (one sentence each, ≤20 words) that would naturally route to this skill.

Required mix — output in this exact order:
1. Explicit 1 — directly names the tool, API, or format (e.g. "Use PhantomBuster to extract LinkedIn leads")
2. Explicit 2 — another explicit request naming the tool or format
3. Paraphrase 1 — describes what the skill does, no casual tone
4. Paraphrase 2 — different angle on the same capability
5. Colloquial 1 — casual/informal phrasing ("can you just...", "quick question about...")
6. Colloquial 2 — another informal phrasing
7. Implicit 1 — describes ONLY the end goal or outcome; NO tool name, API name, \
skill name, or format name at all (e.g. "gather follower counts from multiple social accounts", \
"figure out where my team spent their hours last sprint")
8. Implicit 2 — another goal/outcome phrase from a different scenario; still no tool names

The implicit phrases (7 & 8) are the most important. They must describe what \
someone needs to accomplish without revealing which tool or service they want.

Output ONLY the JSON array of 8 strings, nothing else."""


def call_expand(skill: dict, dry_run: bool = False):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set.")

    user_msg = (
        f"Skill name: {skill['name']}\n"
        f"Description: {skill['description']}\n\n"
        "Generate exactly 8 example user requests following the required mix. "
        "Output ONLY a JSON array of 8 strings."
    )

    if dry_run:
        name = skill["name"]
        return [
            f"Use {name} for my task",
            f"Set up {name} integration",
            f"Help me automate with {name}",
            f"Configure {name} for our workflow",
            f"Can you just hook up {name} quickly?",
            f"I need that {name} thing",
            f"automate the data collection from external sources",
            f"track outcomes and report on performance metrics",
        ], 0, 0, 0.0

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/skillrouter",
            "X-Title":       "SkillRouter-ExpandV2",
        },
        json={
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0.7,
            "max_tokens":  400,
        },
        timeout=30,
    )
    resp.raise_for_status()
    body  = resp.json()
    usage = body.get("usage", {})
    pt    = usage.get("prompt_tokens",     0)
    ct    = usage.get("completion_tokens", 0)
    cost  = usage.get("cost") or (pt * IN_PRICE + ct * OUT_PRICE)
    raw   = body["choices"][0]["message"]["content"].strip()

    try:
        phrases = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\[[\s\S]*?\]', raw)
        phrases = json.loads(m.group()) if m else None

    if not isinstance(phrases, list) or len(phrases) < 6:
        print(f"  [WARN] parse failed for {skill['name']!r}: {raw[:80]!r}")
        return None

    phrases = [str(p).strip() for p in phrases if str(p).strip()][:8]
    return phrases, pt, ct, cost


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {}


def save_progress(p: dict) -> None:
    with open(PROGRESS_PATH, "w") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)


def load_cost() -> dict:
    if COST_PATH.exists():
        with open(COST_PATH) as f:
            return json.load(f)
    return {"total_usd": 0.0, "calls": 0}


def save_cost(t: dict) -> None:
    with open(COST_PATH, "w") as f:
        json.dump(t, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(EXPANDED_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    progress = load_progress()
    tracker  = load_cost()

    already = len([s for s in corpus if s["id"] in progress])
    print(f"Corpus: {len(corpus)} skills")
    print(f"Already done: {already}  Remaining: {len(corpus) - already}")
    print(f"Running cost: ${tracker['total_usd']:.4f}  (halt at ${HALT_AT:.2f})")

    if tracker["total_usd"] >= HALT_AT:
        sys.exit(f"[HALT] Budget ${HALT_AT:.2f} reached.")

    to_process = [s for s in corpus if s["id"] not in progress]
    if args.limit:
        to_process = to_process[:args.limit]

    total_cost = 0.0

    for i, skill in enumerate(to_process, 1):
        result = call_expand(skill, dry_run=args.dry_run)
        if result is None:
            progress[skill["id"]] = []
            save_progress(progress)
            continue

        phrases, pt, ct, cost = result
        total_cost += cost

        if not args.dry_run:
            tracker["total_usd"] = round(tracker["total_usd"] + cost, 6)
            tracker["calls"]    += 1
            save_cost(tracker)
            if tracker["total_usd"] >= HALT_AT:
                print(f"\n[HALT] Budget reached at ${tracker['total_usd']:.4f}")
                progress[skill["id"]] = phrases
                save_progress(progress)
                break

        progress[skill["id"]] = phrases
        save_progress(progress)

        if i % 50 == 0 or i == len(to_process):
            print(f"  [{i}/{len(to_process)}] {skill['name']!r}  "
                  f"phrases={len(phrases)}  run_cost=${total_cost:.4f}")

    # Merge v2 phrases back into corpus (add trigger_phrases_v2 field)
    phrase_map = dict(progress)
    n_with = 0
    for s in corpus:
        phrases = phrase_map.get(s["id"], [])
        if phrases:
            s["trigger_phrases_v2"] = phrases
            n_with += 1

    with open(EXPANDED_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"\nDone.  Skills with trigger_phrases_v2: {n_with}/{len(corpus)}")
    print(f"Run cost: ${total_cost:.4f}  Running total: ${tracker['total_usd']:.4f}")
    print(f"Saved → {EXPANDED_PATH}")


if __name__ == "__main__":
    main()
