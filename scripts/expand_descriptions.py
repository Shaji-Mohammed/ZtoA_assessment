#!/usr/bin/env python3
"""
expand_descriptions.py — Lever D: generate trigger phrases per skill

For each of the 534 skills in skills_corpus.json, asks gpt-4o-mini to write
3-5 short example user requests that would naturally route to that skill.
Phrases are stored in a new 'trigger_phrases' field; descriptions are
NOT modified.

Output:
  data/skills_corpus_expanded.json   — corpus + trigger_phrases field
  data/expand_progress.json          — incremental save so re-runs are free

Estimated cost: ~$0.05 for 534 skills.
Usage:
    python scripts/expand_descriptions.py [--dry-run] [--limit N]
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
CORPUS_PATH   = ROOT / "data" / "skills_corpus.json"
EXPANDED_PATH = ROOT / "data" / "skills_corpus_expanded.json"
PROGRESS_PATH = ROOT / "data" / "expand_progress.json"
COST_PATH     = ROOT / "data" / ".cost_tracker.json"

load_dotenv(ROOT / ".env")

MODEL       = "openai/gpt-4o-mini"
IN_PRICE    = 0.15 / 1_000_000
OUT_PRICE   = 0.60 / 1_000_000
HALT_AT_USD = 3.00

SYSTEM_PROMPT = (
    "You are a test-case generator for a skill router. "
    "Given a skill name and description, output a JSON array of 4 short "
    "user requests (one sentence each, ≤20 words) that would naturally route "
    "to this skill. Vary the phrasing: include at least one goal-oriented request "
    "(what the user wants to achieve) and one action-oriented request (what the "
    "user wants to do). Output ONLY the JSON array, nothing else."
)


def call_expand(skill: dict, dry_run: bool = False) -> list[str] | None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set.")

    user_msg = (
        f"Skill name: {skill['name']}\n"
        f"Description: {skill['description']}\n\n"
        "Generate 4 short example user requests for this skill. "
        "Output ONLY a JSON array of 4 strings."
    )

    if dry_run:
        print(f"  [dry-run] would call API for: {skill['name']}")
        return [
            f"Help me use {skill['name']}",
            f"I need {skill['name']} functionality",
            f"Can you do {skill['name']} for me?",
            f"Apply {skill['name']} to my task",
        ]

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/skillrouter",
            "X-Title":       "SkillRouter-Expand",
        },
        json={
            "model":       MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0.7,
            "max_tokens":  200,
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

    # Parse the JSON array
    try:
        phrases = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\[[\s\S]*?\]', raw)
        if m:
            try:
                phrases = json.loads(m.group())
            except json.JSONDecodeError:
                phrases = None
        else:
            phrases = None

    if not isinstance(phrases, list) or not phrases:
        print(f"  [WARN] parse failed for {skill['name']!r}: {raw[:80]!r}")
        return None

    # Keep only strings, strip whitespace
    phrases = [str(p).strip() for p in phrases if str(p).strip()][:5]
    return phrases, pt, ct, cost


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_cost() -> dict:
    if COST_PATH.exists():
        with open(COST_PATH) as f:
            return json.load(f)
    return {"total_usd": 0.0, "calls": 0}


def save_cost(tracker: dict) -> None:
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COST_PATH, "w") as f:
        json.dump(tracker, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate fake phrases without calling the API")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N skills (0 = all)")
    args = parser.parse_args()

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    progress = load_progress()
    tracker  = load_cost()

    already_done = len([s for s in corpus if s["id"] in progress])
    print(f"Corpus: {len(corpus)} skills")
    print(f"Already done: {already_done}  Remaining: {len(corpus) - already_done}")

    if tracker["total_usd"] >= HALT_AT_USD:
        sys.exit(f"[HALT] Budget ${HALT_AT_USD:.2f} reached. Reset .cost_tracker.json.")

    to_process = [s for s in corpus if s["id"] not in progress]
    if args.limit:
        to_process = to_process[:args.limit]

    total_cost = 0.0
    n_done     = 0

    for i, skill in enumerate(to_process, 1):
        result = call_expand(skill, dry_run=args.dry_run)
        if result is None:
            progress[skill["id"]] = []   # mark as attempted; leave empty
            continue

        if args.dry_run:
            phrases = result
            pt = ct = 0
            cost = 0.0
        else:
            phrases, pt, ct, cost = result

        total_cost += cost
        n_done     += 1

        if not args.dry_run:
            tracker["total_usd"] = round(tracker["total_usd"] + cost, 6)
            tracker["calls"]    += 1
            save_cost(tracker)

            if tracker["total_usd"] >= HALT_AT_USD:
                print(f"\n[HALT] Budget reached at ${tracker['total_usd']:.4f}")
                progress[skill["id"]] = phrases
                save_progress(progress)
                break

        progress[skill["id"]] = phrases
        save_progress(progress)

        if i % 50 == 0 or i == len(to_process):
            print(f"  [{i}/{len(to_process)}] {skill['name']!r}  "
                  f"phrases={len(phrases)}  cost=${total_cost:.4f}")

    # Merge phrases back into corpus
    phrase_map = {sid: phrases for sid, phrases in progress.items()}
    expanded   = []
    n_with     = 0
    for s in corpus:
        s2 = dict(s)
        phrases = phrase_map.get(s["id"], [])
        if phrases:
            s2["trigger_phrases"] = phrases
            n_with += 1
        expanded.append(s2)

    with open(EXPANDED_PATH, "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)

    print(f"\nDone.  Skills with phrases: {n_with}/{len(corpus)}")
    print(f"Total expand cost: ${total_cost:.4f}")
    print(f"Running total    : ${tracker['total_usd']:.4f}")
    print(f"Saved → {EXPANDED_PATH}")


if __name__ == "__main__":
    main()
