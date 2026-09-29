#!/usr/bin/env python3
"""
build_eval.py — Synthetic eval set generator

Generates task prompts with ground-truth skill sets for offline evaluation.
Always run --dry-run first; it reports REAL per-call costs from the OpenRouter
response and extrapolates to the full run before committing anything to disk.

Usage:
    # Step 1: dry run — 8 skills, ~6 API calls, no files written
    python scripts/build_eval.py --dry-run

    # Step 2: full run — only after confirming dry-run cost is safe
    python scripts/build_eval.py

Output schema (data/eval_set.json):
    [
      {
        "prompt_id": "eval_001",
        "prompt": "...",
        "ground_truth_ids":   ["id1", "id2"],
        "ground_truth_names": ["skill-a", "skill-b"],
        "set_size": 2,
        "trigger_type": "explicit"   # or "implied"
      },
      ...
    ]

Full-run plan (used for extrapolation):
    FULL_N_SKILLS      = 80   (stratified across repos)
    FULL_SINGLE_CALLS  = 45   → 90 explicit single-skill prompts
    FULL_MULTI2_CALLS  = 15   → up to 30 two-skill prompts (similarity-guided)
    FULL_MULTI3_CALLS  = 5    → up to 10 three-skill prompts (similarity-guided)
    FULL_IMPLIED_CALLS = 15   → 30 implied-need single-skill prompts
    TOTAL API CALLS    = 80   → target 120-150 eval prompts

Multi-skill pairing strategy:
    Pairs/triples are selected by cosine similarity in [0.20, 0.45] among
    the sampled skills, ensuring related-but-distinct co-occurrence rather
    than random cross-domain groupings that always return [].
    If fewer same-similarity-band pairs exist than needed, remaining groups
    fall back to random pairing within the same source repo.

trigger_type field:
    "explicit" — task directly names the deliverable format, platform, or action
    "implied"  — task expresses the underlying goal; skill need is inferred from context
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

import requests
from dotenv import load_dotenv

ROOT        = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "skills_corpus.json"
EVAL_PATH   = ROOT / "data" / "eval_set.json"
COST_PATH   = ROOT / "data" / ".cost_tracker.json"

load_dotenv(ROOT / ".env")

GEN_MODEL       = "openai/gpt-4o-mini"
INPUT_PRICE     = 0.15 / 1_000_000   # $ per token
OUTPUT_PRICE    = 0.60 / 1_000_000
HALT_AT_USD     = 3.00
N_VARIANTS      = 2    # prompts generated per API call

# ── full-run plan constants ────────────────────────────────────────────────────
# Target: 120-150 prompts from 80 API calls (2 variants each).
# Corpus v4 is 534 skills; 80-skill sample gives good cross-repo coverage.
FULL_N_SKILLS      = 80
FULL_SINGLE_CALLS  = 45   # → 90 explicit single-skill prompts
FULL_MULTI2_CALLS  = 15   # → up to 30 two-skill prompts (similarity-guided pairs)
FULL_MULTI3_CALLS  = 5    # → up to 10 three-skill prompts
FULL_IMPLIED_CALLS = 15   # → 30 implied-need single-skill prompts
FULL_TOTAL_CALLS   = (FULL_SINGLE_CALLS + FULL_MULTI2_CALLS
                      + FULL_MULTI3_CALLS + FULL_IMPLIED_CALLS)

# Similarity band for multi-skill pairing: related-but-distinct
MULTI_SIM_LO = 0.20
MULTI_SIM_HI = 0.45

# ── dry-run plan ──────────────────────────────────────────────────────────────
DRY_N_SKILLS      = 12
DRY_SINGLE_CALLS  = 3
DRY_MULTI2_CALLS  = 2
DRY_MULTI3_CALLS  = 1
DRY_IMPLIED_CALLS = 2
DRY_TOTAL_CALLS   = (DRY_SINGLE_CALLS + DRY_MULTI2_CALLS
                     + DRY_MULTI3_CALLS + DRY_IMPLIED_CALLS)


# ── cost helpers ──────────────────────────────────────────────────────────────
def load_cost() -> dict:
    if COST_PATH.exists():
        with open(COST_PATH) as f:
            return json.load(f)
    return {"total_usd": 0.0, "calls": 0}


def save_cost(tracker: dict) -> None:
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COST_PATH, "w") as f:
        json.dump(tracker, f, indent=2)


def check_budget(tracker: dict) -> None:
    if tracker["total_usd"] >= HALT_AT_USD:
        sys.exit(
            f"\n[HALT] Cumulative spend ${tracker['total_usd']:.4f} has reached "
            f"the ${HALT_AT_USD:.2f} safety limit.\n"
            f"Reset data/.cost_tracker.json manually to continue."
        )


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * INPUT_PRICE + completion_tokens * OUTPUT_PRICE


# ── corpus sampling ───────────────────────────────────────────────────────────
def sample_skills(corpus: list[dict], n: int, seed: int = 42) -> list[dict]:
    """
    Stratified sample of n skills.

    Corpus v3 composition: ~19 anthropics, ~382 nvidia, ~432 alirezarezvani, ~83 composio.

    Strategy:
      - Always include ALL anthropics_skills (official, highest confidence).
      - Remaining budget split proportionally across alirezarezvani and composio.
    """
    rng = random.Random(seed)
    by_repo = {}
    for s in corpus:
        by_repo.setdefault(s["source_repo"], []).append(s)

    # Always keep all anthropics_skills
    kept      = list(by_repo.get("anthropics_skills", []))
    remaining = n - len(kept)

    if remaining <= 0:
        return rng.sample(kept, n)

    # Proportional split across community repos (nvidia excluded from corpus)
    COMMUNITY_REPOS = ["alirezarezvani_skills", "composio_awesome"]
    pools = {repo: list(by_repo.get(repo, [])) for repo in COMMUNITY_REPOS}
    community_total = sum(len(v) for v in pools.values())

    taken = 0
    shares = {}
    for repo in COMMUNITY_REPOS[:-1]:   # all but last get rounded share
        share = round(remaining * len(pools[repo]) / community_total) if community_total else 0
        share = min(share, len(pools[repo]))
        shares[repo] = share
        taken += share
    # Last repo gets the leftover to avoid rounding drift
    last = COMMUNITY_REPOS[-1]
    shares[last] = min(len(pools[last]), remaining - taken)

    for repo in COMMUNITY_REPOS:
        rng.shuffle(pools[repo])
        kept += pools[repo][:shares[repo]]

    rng.shuffle(kept)
    return kept[:n]


# ── prompt generation ─────────────────────────────────────────────────────────
def build_generation_prompt(skills: list[dict], n_variants: int) -> str:
    """
    Build the LLM prompt that generates eval task descriptions.

    The 'required vs helpful' rule is encoded explicitly:
      REQUIRED  = the task explicitly mentions or clearly implies the specific
                  deliverable format, platform, or action the skill provides,
                  AND the task cannot be completed to spec without that skill.
      HELPFUL   = skill is adjacent / would improve quality, but not explicitly
                  triggered and not strictly necessary.

    For single-skill prompts: only ONE skill should be triggered.
    For multi-skill prompts: each skill must have its own distinct explicit trigger.
    """
    if len(skills) == 1:
        s = skills[0]
        skill_block = f"Skill: {s['name']}\nDescription: {s['description']}"
        rules = (
            "Rules:\n"
            "- Do NOT mention the skill name or any technical system term in the task — "
            "write only what the user wants to accomplish.\n"
            "- The task must explicitly reference the specific deliverable format, platform, "
            "or action this skill provides (e.g. the file type, the service, the output).\n"
            "- Keep it focused: the task should not also require other unrelated capabilities.\n"
            "- Sound like a real user request, not a system instruction."
        )
        instruction = (
            f"Write exactly {n_variants} different, realistic task descriptions that a user "
            "would give to an AI assistant and that REQUIRE this skill to complete — "
            "the deliverable cannot be produced without it.\n\n"
            f"{rules}"
        )
    else:
        lines = [f"{i+1}. {s['name']}: {s['description']}" for i, s in enumerate(skills)]
        skill_block = "Skills required:\n" + "\n".join(lines)
        rules = (
            "Rules:\n"
            "- Do NOT mention skill names or technical system terms — write only what "
            "the user wants to accomplish.\n"
            "- Each skill must be explicitly triggered by a specific element in the task "
            "(a named format, platform, service, or output type) — not merely helpful.\n"
            "- The task cannot be completed to its stated specification without every skill.\n"
            "- Sound like a real user request, not a system instruction.\n"
            "- IMPORTANT: If these skills cannot naturally co-occur in a single realistic "
            "user task, output [] instead of forcing an artificial one."
        )
        instruction = (
            f"Write exactly {n_variants} different, realistic task descriptions that a user "
            "would give to an AI assistant and that REQUIRE ALL of the listed skills. "
            "Each skill must have its own explicit trigger in the task wording.\n\n"
            f"{rules}"
        )

    return (
        "You are building a test dataset for an AI skill routing system.\n\n"
        f"{skill_block}\n\n"
        f"{instruction}\n\n"
        "Return a JSON array of strings and nothing else.\n"
        f'Example: ["Do task A here", "Do task B here"]'
    )


def build_implied_prompt(skill: dict, n_variants: int) -> str:
    """
    Build the LLM prompt for implied-need prompts (single-skill only).

    The generated task expresses the user's underlying goal rather than naming
    the deliverable format, platform, or action the skill provides.  The skill
    need must be inferred from context — this is the harder, semantic-routing
    case as opposed to keyword-triggered matching.
    """
    skill_block = f"Skill: {skill['name']}\nDescription: {skill['description']}"
    rules = (
        "Rules:\n"
        "- Do NOT name the deliverable format, platform, service, or technical action "
        "the skill provides (e.g. do not say 'PowerPoint', 'PDF', 'Slack', 'git diff').\n"
        "- Express the user's underlying need or business goal instead — what they want "
        "to achieve, not how or in what format.\n"
        "- The skill should be the natural solution to the stated goal, but the user "
        "has not explicitly triggered it by naming the format or action.\n"
        "- Sound like a real user request.\n"
        "- Example pattern: instead of 'create a PowerPoint presentation' → "
        "'put together something I can walk the team through on screen'."
    )
    instruction = (
        f"Write exactly {n_variants} different, realistic task descriptions where this "
        "skill is the most appropriate solution, but where the user expresses their goal "
        "rather than naming the specific output format or tool. The need for this skill "
        "should be implied by context, not explicitly stated.\n\n"
        f"{rules}"
    )

    return (
        "You are building a test dataset for an AI skill routing system.\n\n"
        f"{skill_block}\n\n"
        f"{instruction}\n\n"
        "Return a JSON array of strings and nothing else.\n"
        f'Example: ["Do task A here", "Do task B here"]'
    )


def parse_prompts(raw: str) -> list[str] | None:
    """Extract a JSON array of strings from LLM output."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(p) for p in parsed if p]
    except json.JSONDecodeError:
        pass
    # Fallback: find first [...] block
    m = re.search(r'\[[\s\S]*?\]', raw)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [str(p) for p in parsed if p]
        except json.JSONDecodeError:
            pass
    return None


def call_api(user_msg: str) -> dict:
    """
    Call OpenRouter. Returns dict with keys:
        prompts       — list[str] generated prompts
        prompt_tokens
        completion_tokens
        cost_from_api — float from response body, or None if absent
        cost_estimate — float calculated from token counts * price
        cost_used     — whichever was available (api > estimate)
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set. Add it to .env")

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/skillrouter",
            "X-Title":       "SkillRouter-EvalGen",
        },
        json={
            "model":       GEN_MODEL,
            "messages":    [{"role": "user", "content": user_msg}],
            "temperature": 0.7,   # some variety across prompts
            "max_tokens":  300,
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()

    usage            = body.get("usage", {})
    prompt_tokens    = usage.get("prompt_tokens",     0)
    completion_tokens = usage.get("completion_tokens", 0)
    cost_from_api    = usage.get("cost")             # None if not present
    cost_estimate    = estimate_cost(prompt_tokens, completion_tokens)
    # Prefer API-reported cost; fall back to estimate; never use 0 if tokens > 0
    cost_used = cost_from_api if (cost_from_api is not None and cost_from_api > 0) else cost_estimate

    raw = body["choices"][0]["message"]["content"]
    prompts = parse_prompts(raw) or []

    return {
        "prompts":           prompts,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_from_api":     cost_from_api,
        "cost_estimate":     cost_estimate,
        "cost_used":         cost_used,
        "raw":               raw,
    }


# ── batch generation ──────────────────────────────────────────────────────────
def run_generation(
    skill_groups: list[list[dict]],
    label: str,
    dry_run: bool,
    cost_tracker: dict,
    trigger_type: str = "explicit",
) -> list[dict]:
    """
    Run one generation call per skill group.
    Returns list of eval records (without prompt_id yet).

    trigger_type: "explicit" or "implied"
    """
    records = []
    for i, skills in enumerate(skill_groups, 1):
        names = " + ".join(s["name"] for s in skills)
        print(f"\n  Call {i}/{len(skill_groups)} [{label}]: {names}")

        if trigger_type == "implied":
            gen_prompt = build_implied_prompt(skills[0], N_VARIANTS)
        else:
            gen_prompt = build_generation_prompt(skills, N_VARIANTS)
        result = call_api(gen_prompt)  # always makes a real call

        pt   = result["prompt_tokens"]
        ct   = result["completion_tokens"]
        cost = result["cost_used"]
        src  = "API" if result["cost_from_api"] is not None else "estimate"

        print(f"    prompt_tokens={pt}  completion_tokens={ct}  "
              f"cost=${cost:.6f} ({src})")

        if result["prompts"]:
            for p in result["prompts"]:
                print(f"    → {p!r}")
        else:
            print(f"    raw: {result['raw'][:120]!r}")

        # Always enforce budget; only persist to disk on a real (non-dry) run
        check_budget(cost_tracker)
        if not dry_run:
            cost_tracker["total_usd"] = round(cost_tracker["total_usd"] + cost, 6)
            cost_tracker["calls"] += 1
            save_cost(cost_tracker)

        for prompt_text in result["prompts"]:
            records.append({
                "_skills":       skills,
                "_prompt":       prompt_text,
                "_trigger_type": trigger_type,
                "_pt":           pt,
                "_ct":           ct,
                "_cost":         cost,
            })

        # accumulate for summary even in dry run
        records.append({
            "_skills":  skills,
            "_pt":      pt,
            "_ct":      ct,
            "_cost":    cost,
            "_prompts": result["prompts"],
            "_is_meta": True,   # call-level record, not a prompt record
        })

    return records


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic eval prompts")
    parser.add_argument(
        "--dry-run", action="store_true",
        help=f"Run {DRY_TOTAL_CALLS} calls on {DRY_N_SKILLS} skills, "
             "report real costs, extrapolate — no files written",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for skill sampling (default: 42)")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output file path (default: data/eval_set.json). "
                             "Use a different path to generate supplementary sets, "
                             "e.g. data/eval_dev_extra.json or data/eval_final_test.json")
    parser.add_argument("--n-skills", type=int, default=0,
                        help="Override number of skills to sample (0 = use built-in default)")
    parser.add_argument("--prompt-id-prefix", default="eval",
                        help="Prefix for prompt_id fields (default: 'eval'). "
                             "Change to 'dev' or 'ft' to avoid ID collisions.")
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        sys.exit(f"Corpus not found: {CORPUS_PATH}")
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set. Add it to .env")

    cost_tracker = load_cost()
    check_budget(cost_tracker)

    rng = random.Random(args.seed)

    # ── select skills ─────────────────────────────────────────────────────────
    if args.n_skills:
        n_skills = args.n_skills
    else:
        n_skills = DRY_N_SKILLS if args.dry_run else FULL_N_SKILLS
    skills   = sample_skills(corpus, n_skills, seed=args.seed)

    print(f"\n{'=== DRY RUN ===' if args.dry_run else '=== FULL RUN ==='}")
    print(f"Skills sampled: {n_skills}")
    from collections import Counter
    repo_counts = Counter(s["source_repo"] for s in skills)
    for repo, cnt in repo_counts.most_common():
        print(f"  {repo}: {cnt}")

    # ── build call groups ─────────────────────────────────────────────────────
    n_single  = DRY_SINGLE_CALLS  if args.dry_run else FULL_SINGLE_CALLS
    n_multi2  = DRY_MULTI2_CALLS  if args.dry_run else FULL_MULTI2_CALLS
    n_multi3  = DRY_MULTI3_CALLS  if args.dry_run else FULL_MULTI3_CALLS
    n_implied = DRY_IMPLIED_CALLS if args.dry_run else FULL_IMPLIED_CALLS

    shuffled = list(skills)
    rng.shuffle(shuffled)

    single_groups = [[s] for s in shuffled[:n_single]]

    # Multi-skill: similarity-guided pairing
    # Load the skill vectors so we can pair skills that are related-but-distinct.
    idx_data    = np.load(ROOT / "data" / "skills_index.npz")
    all_vectors = idx_data["vectors"]
    with open(ROOT / "data" / "skills_index_meta.json") as _f:
        _meta = json.load(_f)
    _id_to_idx = {m["id"]: i for i, m in enumerate(_meta)}

    def _get_similarity_guided_groups(pool_skills, group_size, n_groups, rng_):
        """
        Build groups of `group_size` skills where every pair in the group
        has cosine similarity in [MULTI_SIM_LO, MULTI_SIM_HI].  Falls back
        to same-repo pairing if the similarity band yields too few candidates.
        """
        # Filter to skills that have a vector entry; deduplicate by ID
        seen_ids: set = set()
        eligible = []
        for s in pool_skills:
            if s["id"] in _id_to_idx and s["id"] not in seen_ids:
                eligible.append(s)
                seen_ids.add(s["id"])
        if len(eligible) < group_size:
            return []

        vecs   = np.array([all_vectors[_id_to_idx[s["id"]]] for s in eligible])
        sim    = vecs @ vecs.T                # cosine (L2-normed)
        n      = len(eligible)
        groups = []
        used   = set()

        if group_size == 2:
            # Collect all in-band pairs, sorted by closeness to band centre (0.325)
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    s = float(sim[i, j])
                    if MULTI_SIM_LO <= s <= MULTI_SIM_HI:
                        pairs.append((abs(s - 0.325), i, j))
            pairs.sort()
            for _, i, j in pairs:
                if i not in used and j not in used:
                    groups.append([eligible[i], eligible[j]])
                    used.update([i, j])
                if len(groups) >= n_groups:
                    break

        elif group_size == 3:
            # Greedy: for each unused skill, find the 2 closest in-band companions
            order = list(range(n))
            rng_.shuffle(order)
            for anchor in order:
                if anchor in used:
                    continue
                companions = sorted(
                    [j for j in range(n) if j != anchor and j not in used
                     and MULTI_SIM_LO <= float(sim[anchor, j]) <= MULTI_SIM_HI],
                    key=lambda j: abs(float(sim[anchor, j]) - 0.325),
                )
                if len(companions) >= 2:
                    i2, i3 = companions[0], companions[1]
                    groups.append([eligible[anchor], eligible[i2], eligible[i3]])
                    used.update([anchor, i2, i3])
                if len(groups) >= n_groups:
                    break

        # Fallback: if not enough groups, add same-repo random groups
        if len(groups) < n_groups:
            remaining = [s for i, s in enumerate(eligible) if i not in used]
            by_repo = {}
            for s in remaining:
                by_repo.setdefault(s["source_repo"], []).append(s)
            repo_pool = [s for pool in by_repo.values()
                         for s in pool if len(by_repo[s["source_repo"]]) >= group_size]
            rng_.shuffle(repo_pool)
            same_repo_by_repo = {r: list(ss) for r, ss in by_repo.items()
                                  if len(ss) >= group_size}
            for repo_skills in same_repo_by_repo.values():
                rng_.shuffle(repo_skills)
                while len(repo_skills) >= group_size and len(groups) < n_groups:
                    groups.append(repo_skills[:group_size])
                    repo_skills = repo_skills[group_size:]

        return groups[:n_groups]

    # Use all sampled skills as multi-skill pool (no duplication — the
    # similarity-guided pairer doesn't need wrap-around sequential access).
    multi2_groups = _get_similarity_guided_groups(shuffled, 2, n_multi2, rng)
    multi3_groups = _get_similarity_guided_groups(shuffled, 3, n_multi3, rng)

    # Implied-need: fresh shuffle, distinct from single pool
    rng.shuffle(shuffled)
    implied_groups = [[s] for s in shuffled[:n_implied]]

    total_calls = n_single + len(multi2_groups) + len(multi3_groups) + n_implied
    print(f"\nGeneration plan:")
    print(f"  Single-skill (explicit) : {n_single}")
    print(f"  Two-skill (explicit)    : {len(multi2_groups)}  (requested {n_multi2})")
    print(f"  Three-skill (explicit)  : {len(multi3_groups)}  (requested {n_multi3})")
    print(f"  Single-skill (implied)  : {n_implied}")
    print(f"  Total API calls         : {total_calls}")
    print(f"  Expected prompts        : {total_calls * N_VARIANTS}")

    # ── run generation ────────────────────────────────────────────────────────
    all_meta = []
    for groups, label, ttype in [
        (single_groups,  "single",      "explicit"),
        (multi2_groups,  "two-skill",   "explicit"),
        (multi3_groups,  "three-skill", "explicit"),
        (implied_groups, "implied",     "implied"),
    ]:
        print(f"\n--- {label} ({ttype}) ---")
        meta = run_generation(
            groups, label, dry_run=args.dry_run,
            cost_tracker=cost_tracker, trigger_type=ttype,
        )
        all_meta.extend(meta)

    # ── cost summary ──────────────────────────────────────────────────────────
    call_records = [r for r in all_meta if r.get("_is_meta")]
    total_pt    = sum(r["_pt"]   for r in call_records)
    total_ct    = sum(r["_ct"]   for r in call_records)
    total_cost  = sum(r["_cost"] for r in call_records)
    n_calls     = len(call_records)

    print(f"\n{'='*60}")
    print(f"{'DRY RUN' if args.dry_run else 'FULL RUN'} COST SUMMARY")
    print(f"{'='*60}")
    print(f"  Calls made         : {n_calls}")
    print(f"  Total prompt tok.  : {total_pt:,}")
    print(f"  Total completion t.: {total_ct:,}")
    print(f"  Total cost (actual): ${total_cost:.6f}")
    if n_calls > 0:
        print(f"  Avg cost / call    : ${total_cost / n_calls:.6f}")
    else:
        print(f"  (no calls made — API key missing or dry_run_mode)")

    if args.dry_run and n_calls > 0:
        avg_per_call  = total_cost / n_calls
        projected     = avg_per_call * (FULL_SINGLE_CALLS + FULL_MULTI2_CALLS
                                        + FULL_MULTI3_CALLS + FULL_IMPLIED_CALLS)
        current_spend = cost_tracker["total_usd"]
        headroom      = HALT_AT_USD - current_spend

        print(f"\n{'='*60}")
        print(f"EXTRAPOLATION TO FULL RUN")
        print(f"{'='*60}")
        print(f"  Dry-run avg cost/call : ${avg_per_call:.6f}")
        print(f"  Full-run total calls  : {FULL_TOTAL_CALLS}")
        print(f"  Projected total cost  : ${projected:.6f}")
        print(f"  Current running spend : ${current_spend:.6f}")
        print(f"  Budget remaining      : ${headroom:.6f}  (halt at ${HALT_AT_USD:.2f})")
        print()

        if projected + current_spend >= HALT_AT_USD:
            print(f"  *** WARNING: projected total ${projected + current_spend:.4f} "
                  f"would exceed ${HALT_AT_USD:.2f} halt limit ***")
            print(f"  Do NOT proceed without raising the budget or reducing scope.")
        elif projected + current_spend >= 2.00:
            print(f"  *** CAUTION: projected total ${projected + current_spend:.4f} "
                  f"is within $1 of the ${HALT_AT_USD:.2f} limit — confirm before proceeding ***")
        else:
            print(f"  VERDICT: projected ${projected + current_spend:.4f} total is well "
                  f"under ${HALT_AT_USD:.2f} limit.")
            print(f"  Safe to proceed — run without --dry-run to generate the full eval set.")

        print(f"\nNOTE: dry run did not write any files.")
        return

    # ── full run: build eval records and save ─────────────────────────────────
    prompt_records = [r for r in all_meta if not r.get("_is_meta")]
    prefix         = getattr(args, "prompt_id_prefix", "eval")
    eval_set = []
    for i, r in enumerate(prompt_records, 1):
        skills_used = r["_skills"]
        eval_set.append({
            "prompt_id":           f"{prefix}_{i:03d}",
            "prompt":              r["_prompt"],
            "ground_truth_ids":    [s["id"]   for s in skills_used],
            "ground_truth_names":  [s["name"] for s in skills_used],
            "set_size":            len(skills_used),
            "trigger_type":        r.get("_trigger_type", "explicit"),
        })

    out_path = Path(args.out) if getattr(args, "out", None) else EVAL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(eval_set)} eval prompts → {out_path}")
    set_dist = Counter(r["set_size"] for r in eval_set)
    for size in sorted(set_dist):
        print(f"  set_size={size}: {set_dist[size]} prompts")
    ttype_dist = Counter(r.get("trigger_type", "explicit") for r in eval_set)
    for ttype in sorted(ttype_dist):
        print(f"  trigger_type={ttype}: {ttype_dist[ttype]} prompts")


if __name__ == "__main__":
    main()
