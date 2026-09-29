#!/usr/bin/env python3
"""
rebuild_corpus.py — Curated corpus v2

Transforms the existing skills_corpus.json by applying four rules:

  1. Remove template-skill (placeholder, no real description).
  2. Collapse the 330 generic Composio boilerplate skills ("Automate X tasks
     via Rube MCP... Always search tools first...") into a single canonical
     'composio-automation' entry.  Keep the 53 non-generic Composio skills
     that have real, distinct descriptions.
  3. Deduplicate by skill name across repos.
     Priority order: anthropics_skills > alirezarezvani_skills > composio_awesome.
     When two repos have a skill with the same name, only the higher-priority
     entry is kept — this prevents name-based match masking in the evaluator.
  4. Rewrite descriptions for known Layer-3 offenders (skills whose original
     description describes WHEN to trigger rather than WHAT the skill does,
     making them invisible to the embedding search).
     Original description is preserved in a 'description_original' field.

Outputs:
  data/skills_corpus.json      — new corpus (overwrites existing)
  data/skills_corpus_v1.json   — backup of original

Reports full before/after composition.
"""

import json
import shutil
import hashlib
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
CORPUS_IN   = ROOT / "data" / "skills_corpus.json"
CORPUS_BACK = ROOT / "data" / "skills_corpus_v1.json"
CORPUS_OUT  = ROOT / "data" / "skills_corpus.json"

PRIORITY = ["anthropics_skills", "alirezarezvani_skills", "composio_awesome"]
PRIORITY_RANK = {repo: i for i, repo in enumerate(PRIORITY)}

# Repos excluded before any processing (failed selection criteria).
# Add repo names here to permanently exclude them without losing the cloned source.
EXCLUDE_REPOS = {"nvidia_skills"}


def make_id(source_repo: str, name: str) -> str:
    raw = f"{source_repo}:{name}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ── Rule 2: detect generic Composio boilerplate ───────────────────────────────

def is_generic_composio(skill: dict) -> bool:
    """True if description is the useless boilerplate template."""
    desc = skill.get("description", "")
    return "Rube MCP" in desc and "Always search tools first" in desc


COMPOSIO_CANONICAL = {
    "id":          make_id("composio_awesome", "composio-automation"),
    "name":        "composio-automation",
    "description": (
        "Automates workflows for third-party services via the Composio MCP integration. "
        "Use when the task requires creating, reading, updating, or deleting data in an "
        "external SaaS platform through automated API actions — for example triggering "
        "events, syncing records, managing users, or running scheduled operations in services "
        "such as Slack, GitHub, Jira, Salesforce, Notion, HubSpot, or any other "
        "Composio-supported app. Prefer a more specific service skill when one is available."
    ),
    "source_repo": "composio_awesome",
    "source_path": "canonical-representative",
    "license":     None,
}


# ── Rule 4: description rewrites for Layer-3 offenders ───────────────────────
# Keyed by skill NAME (after dedup, only one entry per name will exist).
# Any skill matching a key gets its description replaced; original is saved.

DESCRIPTION_REWRITES = {
    "discernment-nudge": (
        "Appends 2-3 targeted follow-up questions to a substantive response — "
        "advice, recommendations, plans, analysis, factual claims, or multi-step arguments — "
        "prompting the user to verify key facts, probe assumptions, and spot missing context. "
        "Activates at most once per conversation after a consequential answer; "
        "skips trivial lookups, format conversions, code tasks, creative writing, and casual chat."
    ),
    "karpathy-check": (
        "Reviews staged git changes or the last commit against Andrej Karpathy's four "
        "engineering principles: unnecessary complexity, diff noise (spurious or unrelated "
        "changes), hidden assumptions, and whether the change achieves its stated goal. "
        "Use before committing or submitting a PR for a principled sanity check on the diff."
    ),
    "internal-comms": (
        "Drafts internal-only communications for employees: all-hands announcements, "
        "re-org or leadership-transition notices, tool or policy rollout messages, "
        "layoff communications, acquisition updates, or any change-management message "
        "where the audience is staff rather than customers. "
        "Produces a sequenced touchpoint calendar, a primary announcement, "
        "audience-segmented FAQs, and manager talking points."
    ),
}


def main():
    # ── load & backup ─────────────────────────────────────────────────────────
    with open(CORPUS_IN, encoding="utf-8") as f:
        original = json.load(f)

    shutil.copy(CORPUS_IN, CORPUS_BACK)
    print(f"Backed up original corpus → {CORPUS_BACK}")
    print(f"\nOriginal corpus: {len(original)} skills")
    from collections import Counter
    orig_repos = Counter(s["source_repo"] for s in original)
    for repo in sorted(orig_repos):
        print(f"  {repo}: {orig_repos[repo]}")

    # ── Exclusions: drop repos that failed selection criteria ─────────────────
    if EXCLUDE_REPOS:
        before_excl = len(original)
        original = [s for s in original if s["source_repo"] not in EXCLUDE_REPOS]
        print(f"\nExclusions ({sorted(EXCLUDE_REPOS)}): {before_excl} → {len(original)}")
        excl_repos = Counter(s["source_repo"] for s in original)
        for repo in PRIORITY:
            if repo in excl_repos:
                print(f"  kept {repo}: {excl_repos[repo]}")

    # ── Rule 1: remove template-skill ─────────────────────────────────────────
    before_r1 = len(original)
    skills = [s for s in original if s["name"] != "template-skill"]
    print(f"\nRule 1 — remove template-skill: {before_r1} → {len(skills)}")

    # ── Rule 2: collapse generic Composio boilerplate ─────────────────────────
    generic  = [s for s in skills if s["source_repo"] == "composio_awesome"
                                   and is_generic_composio(s)]
    rest     = [s for s in skills if s not in generic]

    print(f"\nRule 2 — Composio collapse:")
    print(f"  Generic boilerplate skills removed : {len(generic)}")
    non_generic = [s for s in skills if s["source_repo"] == "composio_awesome"
                                      and not is_generic_composio(s)]
    print(f"  Non-generic Composio kept          : {len(non_generic)}")
    print(f"  Canonical entry added              : 1  (composio-automation)")

    skills = rest + [COMPOSIO_CANONICAL]

    # ── Rule 3: dedup by name, highest-priority repo wins ────────────────────
    # Sort by priority rank so the first occurrence of a name is the best one.
    def sort_key(s):
        return PRIORITY_RANK.get(s["source_repo"], 99)

    skills.sort(key=sort_key)

    seen_names = {}
    deduped    = []
    dedup_log  = []

    for s in skills:
        name = s["name"]
        if name in seen_names:
            winner = seen_names[name]
            dedup_log.append(
                f"  DROP  [{s['source_repo']}] {name!r}  "
                f"→ kept [{winner['source_repo']}] version"
            )
        else:
            seen_names[name] = s
            deduped.append(s)

    print(f"\nRule 3 — name deduplication:")
    for line in dedup_log:
        print(line)
    print(f"  Before: {len(skills)}  After: {len(deduped)}  "
          f"(dropped {len(skills) - len(deduped)} dupes)")

    # ── Rule 4: description rewrites ──────────────────────────────────────────
    rewritten = 0
    for s in deduped:
        if s["name"] in DESCRIPTION_REWRITES:
            s["description_original"] = s["description"]
            s["description"] = DESCRIPTION_REWRITES[s["name"]]
            rewritten += 1

    print(f"\nRule 4 — description rewrites applied: {rewritten}")
    for name in DESCRIPTION_REWRITES:
        if name in seen_names:
            print(f"  ✓ {name}")

    # ── final composition report ──────────────────────────────────────────────
    final_repos = Counter(s["source_repo"] for s in deduped)
    print(f"\n{'='*50}")
    print(f"NEW CORPUS: {len(deduped)} skills")
    for repo in PRIORITY:
        if repo in final_repos:
            print(f"  {repo}: {final_repos[repo]}")
    print(f"{'='*50}")

    # ── write ─────────────────────────────────────────────────────────────────
    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"\nWrote new corpus → {CORPUS_OUT}")


if __name__ == "__main__":
    main()
