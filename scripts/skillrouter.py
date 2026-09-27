#!/usr/bin/env python3
"""
skillrouter.py — Skill Router CLI

Two-stage retrieve-then-decide pipeline:
  Stage 1 (local, free): embed query with all-MiniLM-L6-v2, cosine-similarity
    against cached skill vectors, return every skill above THRESHOLD.
  Stage 2 (OpenRouter, cheap): send prompt + Stage-1 candidates to an LLM,
    have it select the final subset. Skipped when Stage 1 produces a single
    clearly-dominant match (gap to second-best >= UNAMBIGUOUS_GAP).

Usage:
    python scripts/skillrouter.py query "<prompt>" [options]
    python scripts/skillrouter.py cost

Options (query):
    --threshold FLOAT   Stage-1 cosine-similarity cut-off  (default: 0.30)
    --no-stage2         Skip Stage-2 LLM rerank entirely
    --dry-run           Print Stage-2 payload without making an API call
    --show-top INT      Debug: also print top-N scores below threshold

Env (required for Stage 2):
    OPENROUTER_API_KEY  Set in .env or shell environment

Budget guard:
    Running cost is accumulated in data/.cost_tracker.json.
    Execution halts if cumulative spend reaches HALT_AT_USD ($3.00).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
VECTORS_PATH = ROOT / "data" / "skills_index.npz"
META_PATH    = ROOT / "data" / "skills_index_meta.json"
COST_PATH    = ROOT / "data" / ".cost_tracker.json"

load_dotenv(ROOT / ".env")

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_NAME      = "all-MiniLM-L6-v2"
STAGE2_MODEL    = "openai/gpt-4o-mini"   # cheap, reliable JSON output
# Pricing for STAGE2_MODEL (USD per token)
S2_INPUT_PRICE  = 0.15  / 1_000_000     # $0.15 / M input tokens
S2_OUTPUT_PRICE = 0.60  / 1_000_000     # $0.60 / M output tokens
HALT_AT_USD     = 3.00                  # hard spend cap
UNAMBIGUOUS_GAP = 0.15                  # top-1 vs top-2 score gap → skip Stage 2


# ── module-level caches ───────────────────────────────────────────────────────
_vectors = None   # np.ndarray (N, 384) float32, L2-normalised
_meta    = None   # list of {id, name, description, source_repo}
_model   = None   # SentenceTransformer instance


def _load_index():
    global _vectors, _meta
    if _vectors is None:
        if not VECTORS_PATH.exists():
            sys.exit(
                f"Index not found at {VECTORS_PATH}\n"
                "Run:  python scripts/build_index.py"
            )
        data = np.load(VECTORS_PATH)
        _vectors = data["vectors"]
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
        if len(_vectors) != len(_meta):
            sys.exit("Index corrupted: vectors/meta length mismatch. Rebuild with build_index.py")
    return _vectors, _meta


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Stage 1: local embedding + cosine similarity ──────────────────────────────
def stage1(prompt: str, threshold: float) -> tuple[list[dict], float, float]:
    """
    Embed prompt, score all skills, return candidates above threshold.

    Returns:
        candidates    — list of skill dicts with added 'score' key, sorted desc
        top_score     — cosine sim of the best-matching skill
        second_score  — cosine sim of the second-best skill (may be < threshold)
    """
    model = _get_model()
    vec = model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)[0]
    vec = vec.astype(np.float32)

    vectors, meta = _load_index()
    scores = vectors @ vec                     # dot product == cosine sim (normalised)

    order = np.argsort(scores)[::-1]
    top_score    = float(scores[order[0]])
    second_score = float(scores[order[1]]) if len(order) > 1 else 0.0

    candidates = []
    for idx in order:
        s = float(scores[idx])
        if s < threshold:
            break
        candidates.append({"score": s, **meta[idx]})

    return candidates, top_score, second_score


def is_unambiguous(candidates: list[dict], top_score: float, second_score: float) -> bool:
    """
    True when Stage 1 has a single dominant match and Stage 2 would add no value.
    Condition: exactly one candidate above threshold, and the gap to the
    second-best skill (wherever it sits) is >= UNAMBIGUOUS_GAP.
    """
    return len(candidates) == 1 and (top_score - second_score) >= UNAMBIGUOUS_GAP


# ── Stage 2: LLM rerank (OpenRouter) ─────────────────────────────────────────
def _load_cost() -> dict:
    if COST_PATH.exists():
        with open(COST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"total_usd": 0.0, "calls": 0}


def _save_cost(tracker: dict) -> None:
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COST_PATH, "w", encoding="utf-8") as f:
        json.dump(tracker, f, indent=2)


def _check_budget(tracker: dict) -> None:
    if tracker["total_usd"] >= HALT_AT_USD:
        sys.exit(
            f"\n[HALT] Cumulative spend ${tracker['total_usd']:.4f} has reached "
            f"the ${HALT_AT_USD:.2f} safety limit.\n"
            f"Reset data/.cost_tracker.json manually to continue."
        )


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * S2_INPUT_PRICE + completion_tokens * S2_OUTPUT_PRICE


def _build_stage2_prompt(task: str, candidates: list[dict]) -> str:
    lines = [
        f"Task: {task}",
        "",
        "Candidate skills (numbered):",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. {c['name']}: {c['description']}")
    lines += [
        "",
        "Return a JSON array of the numbers of the skills that are actually needed "
        "to complete the task. Include a skill only if it is genuinely required. "
        "You may return fewer than all candidates. Return ONLY the JSON array, "
        "nothing else. Example: [1, 3]",
    ]
    return "\n".join(lines)


def _parse_stage2_response(raw: str, candidates: list[dict]) -> list[dict] | None:
    """
    Parse LLM output into a list of chosen candidates.
    Returns None if parsing fails.
    """
    raw = raw.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back: extract first JSON array from anywhere in the response
        m = re.search(r'\[[\s\S]*?\]', raw)
        if not m:
            return None
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    # parsed may be a list of ints, or {"skills": [1, 2]}, etc.
    if isinstance(parsed, dict):
        nums = next((v for v in parsed.values() if isinstance(v, list)), [])
    elif isinstance(parsed, list):
        nums = parsed
    else:
        return None

    # Extract integers (1-based indices into candidates)
    chosen = []
    for n in nums:
        try:
            idx = int(n) - 1          # convert 1-based → 0-based
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates):
            chosen.append(candidates[idx])

    return chosen if chosen else None


def stage2(prompt: str, candidates: list[dict], dry_run: bool = False) -> list[dict]:
    """
    Call the Stage-2 LLM to select the final subset from Stage-1 candidates.
    Returns the chosen subset; falls back to candidates if anything goes wrong.
    """
    import requests as req

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not dry_run:
        print(
            "[WARN] OPENROUTER_API_KEY not set — skipping Stage 2 "
            "(set it in .env or environment)",
            file=sys.stderr,
        )
        return candidates

    tracker = _load_cost()
    _check_budget(tracker)

    user_msg = _build_stage2_prompt(prompt, candidates)

    if dry_run:
        print("\n[DRY RUN] Stage-2 payload:")
        print(f"  model:           {STAGE2_MODEL}")
        print(f"  candidates:      {len(candidates)}")
        print(f"  user_msg chars:  {len(user_msg)}")
        print(f"  ~prompt tokens:  ~{len(user_msg) // 4} (rough estimate)")
        return candidates

    resp = req.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/skillrouter",
            "X-Title":       "SkillRouter",
        },
        json={
            "model":       STAGE2_MODEL,
            "messages": [
                {
                    "role":    "system",
                    "content": (
                        "You are a skill router. Your only job is to select which "
                        "skills from a numbered list are needed for a given task. "
                        "Always respond with a JSON array of integers and nothing else."
                    ),
                },
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0,
            "max_tokens":  128,
        },
        timeout=30,
    )

    resp.raise_for_status()
    body = resp.json()

    # ── cost accounting ───────────────────────────────────────────────────────
    usage = body.get("usage", {})
    prompt_tokens     = usage.get("prompt_tokens",     0)
    completion_tokens = usage.get("completion_tokens", 0)
    # OpenRouter may include a 'cost' field (in USD); fall back to estimate
    call_cost = usage.get("cost") or _estimate_cost(prompt_tokens, completion_tokens)

    tracker["total_usd"] = round(tracker["total_usd"] + call_cost, 6)
    tracker["calls"]    += 1
    _save_cost(tracker)

    print(
        f"  [Stage 2] model={STAGE2_MODEL}  "
        f"tokens={prompt_tokens}+{completion_tokens}  "
        f"cost=${call_cost:.5f}  running=${tracker['total_usd']:.4f}",
        file=sys.stderr,
    )

    _check_budget(tracker)

    # ── parse & validate ──────────────────────────────────────────────────────
    raw = body["choices"][0]["message"]["content"]
    chosen = _parse_stage2_response(raw, candidates)

    if chosen is None:
        print(
            f"[WARN] Stage-2 returned unparseable output: {raw!r}\n"
            "       Falling back to Stage-1 results.",
            file=sys.stderr,
        )
        return candidates

    if not chosen:
        print(
            "[WARN] Stage-2 selected zero skills — falling back to Stage-1 results.",
            file=sys.stderr,
        )
        return candidates

    return chosen


# ── CLI commands ──────────────────────────────────────────────────────────────
def cmd_query(args):
    prompt    = args.prompt
    threshold = args.threshold
    no_stage2 = args.no_stage2
    dry_run   = args.dry_run
    show_top  = args.show_top

    print(f"\nQuery:     {prompt!r}")
    print(f"Threshold: {threshold}")

    candidates, top_score, second_score = stage1(prompt, threshold)

    # ── Stage 1 output ────────────────────────────────────────────────────────
    print(f"\nStage 1 — {len(candidates)} candidate(s) above {threshold}:")
    if candidates:
        for c in candidates:
            print(f"  [{c['score']:.4f}]  {c['name']}")
            print(f"           {c['description'][:90]}")
    else:
        print(f"  (none — top score was {top_score:.4f})")

    # Optional: show more scores below threshold for debugging
    if show_top and show_top > len(candidates):
        model = _get_model()
        vec = model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        vectors, meta = _load_index()
        scores = vectors @ vec
        order = np.argsort(scores)[::-1]
        n_extra = show_top - len(candidates)
        print(f"\n  Next {n_extra} below threshold:")
        shown = 0
        for idx in order:
            s = float(scores[idx])
            if s >= threshold:
                continue
            print(f"  [{s:.4f}]  {meta[idx]['name']}")
            shown += 1
            if shown >= n_extra:
                break

    if not candidates:
        return

    # ── decide whether to run Stage 2 ────────────────────────────────────────
    skip_reason = None
    if no_stage2:
        skip_reason = "--no-stage2 flag"
    elif is_unambiguous(candidates, top_score, second_score):
        skip_reason = (
            f"single dominant match "
            f"(gap={top_score - second_score:.4f} >= {UNAMBIGUOUS_GAP})"
        )

    if skip_reason:
        print(f"\nStage 2: SKIPPED ({skip_reason})")
        final = candidates
    else:
        print(f"\nStage 2: calling {STAGE2_MODEL} …")
        final = stage2(prompt, candidates, dry_run=dry_run)

    # ── final output ──────────────────────────────────────────────────────────
    print(f"\nFinal skill set ({len(final)}):")
    for s in final:
        score_tag = f"[{s['score']:.4f}] " if "score" in s else ""
        print(f"  {score_tag}{s['name']}")
        print(f"    {s['description'][:100]}")

    print("\nJSON:")
    print(json.dumps([{"id": s["id"], "name": s["name"]} for s in final], indent=2))


def cmd_cost(_args):
    tracker = _load_cost()
    print(f"Running cost: ${tracker['total_usd']:.5f}  ({tracker['calls']} Stage-2 calls)")
    print(f"Halt at:      ${HALT_AT_USD:.2f}")
    remaining = max(0.0, HALT_AT_USD - tracker["total_usd"])
    print(f"Remaining:    ${remaining:.5f}")


def main():
    parser = argparse.ArgumentParser(
        prog="skillrouter",
        description="Two-stage skill router: embed → threshold → LLM rerank",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    # query subcommand
    qp = sub.add_parser("query", help="Route a prompt to matching skills")
    qp.add_argument("prompt", help="Task description to route")
    qp.add_argument(
        "--threshold", type=float, default=0.30,
        help="Stage-1 cosine-similarity cut-off (default: 0.30)",
    )
    qp.add_argument(
        "--no-stage2", action="store_true",
        help="Skip Stage-2 LLM rerank entirely",
    )
    qp.add_argument(
        "--dry-run", action="store_true",
        help="Show Stage-2 payload without calling the API",
    )
    qp.add_argument(
        "--show-top", type=int, default=0, metavar="N",
        help="Also print top-N scores below threshold (debug)",
    )

    # cost subcommand
    cp = sub.add_parser("cost", help="Show current running cost")
    cp.add_argument("_unused", nargs="*")   # absorb accidental extra args

    args = parser.parse_args()
    if args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "cost":
        cmd_cost(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
