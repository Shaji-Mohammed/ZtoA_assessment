#!/usr/bin/env python3
"""
skillrouter.py — Skill Router CLI

Two-stage retrieve-then-decide pipeline:
  Stage 1 (local, free): embed query with all-MiniLM-L6-v2, score all skills,
    optionally fuse with BM25 lexical scores (Change 1: hybrid retrieval),
    aggregate per-skill scores (MAX or conservative weighted, Change 3),
    apply relative-score gating (Change 4: alpha * top_score).
  Stage 2 (OpenRouter, cheap): send prompt + Stage-1 candidates to an LLM;
    classify each as REQUIRED / HELPFUL / IRRELEVANT (Change 5).
    Stage 2 is skipped when Stage 1 produces a single clearly-dominant match.

Usage:
    python scripts/skillrouter.py query "<prompt>" [options]
    python scripts/skillrouter.py cost

Options (query):
    --threshold FLOAT   Stage-1 absolute cosine cut-off  (default: 0.30)
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
LEX_PATH     = ROOT / "data" / "lexical_index.pkl"

load_dotenv(ROOT / ".env")

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_NAME      = "all-MiniLM-L6-v2"
STAGE2_MODEL    = "openai/gpt-4o-mini"
S2_INPUT_PRICE  = 0.15  / 1_000_000
S2_OUTPUT_PRICE = 0.60  / 1_000_000
HALT_AT_USD     = 3.00
UNAMBIGUOUS_GAP = 0.15   # top-1 vs top-2 gap → skip Stage 2


# ── module-level caches ────────────────────────────────────────────────────────
_index_cache   = {}   # {suffix: (vectors, meta)}
_model_cache   = {}   # {model_name: SentenceTransformer}
_lexical_cache: dict = {}  # {path_str: loaded BM25 data}


# ── index loaders ──────────────────────────────────────────────────────────────

def _load_index(suffix: str = ""):
    """Load semantic index for *suffix* (e.g. '' or '_multi')."""
    global _index_cache
    if suffix not in _index_cache:
        vec_path  = ROOT / "data" / f"skills_index{suffix}.npz"
        meta_path = ROOT / "data" / f"skills_index{suffix}_meta.json"
        if not vec_path.exists():
            sys.exit(
                f"Index not found at {vec_path}\n"
                "Run:  python scripts/build_index.py"
            )
        data     = np.load(vec_path)
        vectors  = data["vectors"]
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if len(vectors) != len(meta):
            sys.exit(f"Index corrupted ({vec_path}): vectors/meta length mismatch.")
        _index_cache[suffix] = (vectors, meta)
    return _index_cache[suffix]


def _get_model(model_name: str = MODEL_NAME):
    global _model_cache
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _load_lexical_index(path: str | None = None):
    """Load BM25 index once per path; return None if not built yet."""
    import pickle
    lex_path = Path(path) if path else LEX_PATH
    cache_key = str(lex_path)
    if cache_key not in _lexical_cache and lex_path.exists():
        with open(lex_path, "rb") as f:
            _lexical_cache[cache_key] = pickle.load(f)
    return _lexical_cache.get(cache_key)


def _tokenize(text: str) -> list[str]:
    """Match tokeniser used in build_lexical_index.py."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) >= 2]


# ── Stage 1: embed + score + (optional) BM25 fusion ──────────────────────────

def stage1(
    prompt: str,
    threshold: float,
    cap: int = 0,
    # Change 4: relative gating (already in place)
    alpha: float | None = None,
    alpha_floor: float = 0.20,
    # Lever B: embedding model / index selection
    embed_model: str | None = None,
    index_suffix: str = "",
    # Change 1: hybrid BM25 + semantic retrieval
    hybrid: bool = False,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
    lexical_path: str | None = None,  # override default lexical_index.pkl
    # Change 3: conservative weighted aggregation (multi-vector mode only)
    conservative_agg: bool = False,
    w_desc: float = 0.50,
    w_max_phrase: float = 0.30,
    w_top2: float = 0.20,
) -> tuple[list[dict], float, float]:
    """
    Stage 1: embed prompt, score all skills, return candidates above cutoff.

    Cutoff modes:
      alpha=None  → absolute threshold (keep score >= threshold)
      alpha=float → relative gating   (keep score >= max(alpha*top, alpha_floor))

    Multi-vector index (detected by 'skill_id' field in meta):
      Aggregation options:
        conservative_agg=False → MAX over all vectors (original Lever D)
        conservative_agg=True  → w_desc*desc + w_max_phrase*max_phrase + w_top2*mean_top2

    Hybrid mode (hybrid=True):
      Fuses semantic score with normalised BM25 score:
        hybrid_score = semantic_weight * sem + lexical_weight * norm_bm25
      Requires data/lexical_index.pkl (build with build_lexical_index.py).
      If the lexical index is absent, falls back to semantic-only silently.

    Returns:
        candidates   — skill dicts with 'score' key, sorted descending
        top_score    — best per-skill score after fusion/aggregation
        second_score — second-best per-skill score
    """
    model   = _get_model(embed_model or MODEL_NAME)
    vec     = model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)[0]
    vec     = vec.astype(np.float32)
    vectors, meta = _load_index(index_suffix)
    scores  = vectors @ vec   # cosine similarity (L2-normalised)

    # ── Multi-vector mode ─────────────────────────────────────────────────────
    if meta and "skill_id" in meta[0]:
        # Collect per-skill: description score + phrase scores
        skill_desc   = {}   # {sid: desc_score}
        skill_phrs   = {}   # {sid: [phrase_score, ...]}
        skill_entry  = {}   # {sid: meta entry from the description vector}

        for i, m in enumerate(meta):
            sid   = m["skill_id"]
            s     = float(scores[i])
            vtype = m.get("vector_type", "description")

            if vtype == "description":
                skill_desc[sid]  = s
                skill_entry[sid] = m   # description meta as canonical entry
            elif vtype.startswith("phrase"):
                skill_phrs.setdefault(sid, []).append(s)

        # Aggregate per-skill semantic score
        skill_sem = {}
        for sid in skill_entry:
            desc_score    = skill_desc.get(sid, 0.0)
            phrase_scores = sorted(skill_phrs.get(sid, []), reverse=True)
            max_phrase    = phrase_scores[0] if phrase_scores else 0.0
            top2_mean     = (sum(phrase_scores[:2]) / max(1, len(phrase_scores[:2]))
                             if phrase_scores else 0.0)

            if conservative_agg:
                skill_sem[sid] = (w_desc * desc_score
                                  + w_max_phrase * max_phrase
                                  + w_top2 * top2_mean)
            else:
                skill_sem[sid] = max(desc_score, max_phrase)   # original MAX

    # ── Single-vector mode ────────────────────────────────────────────────────
    else:
        skill_sem   = {m["id"]: float(scores[i]) for i, m in enumerate(meta)}
        skill_entry = {m["id"]: m for m in meta}

    # ── Change 1: BM25 fusion ─────────────────────────────────────────────────
    if hybrid:
        lex_data = _load_lexical_index(lexical_path)
        if lex_data is not None:
            query_tokens = _tokenize(prompt)
            bm25_raw     = np.array(lex_data["bm25"].get_scores(query_tokens),
                                    dtype=np.float32)
            max_bm25     = float(bm25_raw.max())
            if max_bm25 > 0:
                norm_lex = bm25_raw / max_bm25
            else:
                norm_lex = bm25_raw
            lex_ids      = lex_data["skill_ids"]
            lex_meta_map = {m["id"]: m for m in lex_data["skill_meta"]}

            for i, sid in enumerate(lex_ids):
                sem = skill_sem.get(sid, 0.0)
                lex = float(norm_lex[i])
                # Union: add skills from lexical that weren't in semantic index
                if sid not in skill_entry and sid in lex_meta_map:
                    skill_entry[sid] = lex_meta_map[sid]
                skill_sem[sid] = semantic_weight * sem + lexical_weight * lex
        # else: lexical index absent — fall through to semantic-only

    # ── Sort, gate, cap ───────────────────────────────────────────────────────
    sorted_items = sorted(skill_sem.items(), key=lambda kv: -kv[1])
    top_score    = sorted_items[0][1]    if sorted_items else 0.0
    second_score = sorted_items[1][1]    if len(sorted_items) > 1 else 0.0
    cutoff       = (max(alpha * top_score, alpha_floor)
                    if alpha is not None else threshold)

    candidates = []
    for sid, s in sorted_items:
        if s < cutoff:
            break
        entry = skill_entry[sid]
        candidates.append({"score": s, **entry})

    if cap and len(candidates) > cap:
        candidates = candidates[:cap]

    return candidates, top_score, second_score


def is_unambiguous(candidates: list[dict], top_score: float, second_score: float) -> bool:
    """
    True when Stage 1 has a single dominant match and Stage 2 adds no value.
    Condition: exactly one candidate, gap to second-best >= UNAMBIGUOUS_GAP.
    """
    return len(candidates) == 1 and (top_score - second_score) >= UNAMBIGUOUS_GAP


# ── Stage 2: LLM rerank (OpenRouter) ──────────────────────────────────────────

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
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\[[\s\S]*?\]', raw)
        if not m:
            return None
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    if isinstance(parsed, dict):
        nums = next((v for v in parsed.values() if isinstance(v, list)), [])
    elif isinstance(parsed, list):
        nums = parsed
    else:
        return None

    chosen = []
    for n in nums:
        try:
            idx = int(n) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates):
            chosen.append(candidates[idx])

    return chosen if chosen else None


def stage2(prompt: str, candidates: list[dict], dry_run: bool = False) -> list[dict]:
    """
    Stage-2 LLM rerank.  Returns chosen subset; falls back to candidates on error.
    """
    import requests as req

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not dry_run:
        print(
            "[WARN] OPENROUTER_API_KEY not set — skipping Stage 2",
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

    usage             = body.get("usage", {})
    prompt_tokens     = usage.get("prompt_tokens",     0)
    completion_tokens = usage.get("completion_tokens", 0)
    call_cost         = usage.get("cost") or _estimate_cost(prompt_tokens, completion_tokens)

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

    raw    = body["choices"][0]["message"]["content"]
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


# ── CLI commands ───────────────────────────────────────────────────────────────

def cmd_query(args):
    prompt    = args.prompt
    threshold = args.threshold
    no_stage2 = args.no_stage2
    dry_run   = args.dry_run
    show_top  = args.show_top
    cap       = args.cap

    print(f"\nQuery:     {prompt!r}")
    print(f"Threshold: {threshold}  cap: {cap if cap else 'none'}")

    candidates, top_score, second_score = stage1(prompt, threshold, cap=cap)

    print(f"\nStage 1 — {len(candidates)} candidate(s) above {threshold}:")
    if candidates:
        for c in candidates:
            print(f"  [{c['score']:.4f}]  {c['name']}")
            print(f"           {c['description'][:90]}")
    else:
        print(f"  (none — top score was {top_score:.4f})")

    if show_top and show_top > len(candidates):
        model = _get_model()
        v = model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        vectors, meta = _load_index()
        sc = vectors @ v
        order = np.argsort(sc)[::-1]
        n_extra = show_top - len(candidates)
        print(f"\n  Next {n_extra} below threshold:")
        shown = 0
        for idx in order:
            s = float(sc[idx])
            if s >= threshold:
                continue
            print(f"  [{s:.4f}]  {meta[idx]['name']}")
            shown += 1
            if shown >= n_extra:
                break

    if not candidates:
        return

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

    qp = sub.add_parser("query", help="Route a prompt to matching skills")
    qp.add_argument("prompt", help="Task description to route")
    qp.add_argument("--threshold", type=float, default=0.30)
    qp.add_argument("--no-stage2", action="store_true")
    qp.add_argument("--dry-run",   action="store_true")
    qp.add_argument("--show-top",  type=int, default=0, metavar="N")
    qp.add_argument("--cap",       type=int, default=40, metavar="N")

    cp = sub.add_parser("cost", help="Show current running cost")
    cp.add_argument("_unused", nargs="*")

    args = parser.parse_args()
    if args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "cost":
        cmd_cost(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
