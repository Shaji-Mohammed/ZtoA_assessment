#!/usr/bin/env python3
"""
build_corpus.py

Walks the cloned skill repos under repos/, finds every SKILL.md file,
parses its YAML frontmatter (name + description), and normalizes all
of them into one shared schema stored at data/skills_corpus.json.

Usage:
    python3 build_corpus.py
"""

import argparse
import json
import random
import re
import hashlib
from pathlib import Path

import yaml

REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "skills_corpus.json"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_skill_md(path: Path):
    """Extract (name, description) from a SKILL.md file's YAML frontmatter."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    match = FRONTMATTER_RE.match(text)
    if not match:
        return None

    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None

    if not isinstance(meta, dict):
        return None

    name = meta.get("name")
    description = meta.get("description")

    if not name or not description:
        return None

    return {
        "name": str(name).strip(),
        "description": str(description).strip(),
        "license": meta.get("license"),
    }


def make_id(source_repo: str, rel_path: str) -> str:
    raw = f"{source_repo}:{rel_path}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def normalize_text_for_dedup(name: str, description: str) -> str:
    combined = f"{name} {description}".lower()
    combined = re.sub(r"[^a-z0-9 ]", "", combined)
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


# Repos listed here are processed first, so when two repos contain a
# near-identical skill (e.g. a community repo copying an official one),
# the higher-priority source wins the dedup instead of losing to
# whichever folder happens to sort first alphabetically.
SOURCE_PRIORITY = ["anthropics_skills"]


def collect_skills():
    records = []
    if not REPOS_DIR.exists():
        raise SystemExit(f"repos dir not found: {REPOS_DIR}")

    all_dirs = sorted(d for d in REPOS_DIR.iterdir() if d.is_dir())
    ordered_dirs = (
        [d for name in SOURCE_PRIORITY for d in all_dirs if d.name == name]
        + [d for d in all_dirs if d.name not in SOURCE_PRIORITY]
    )

    for repo_dir in ordered_dirs:
        source_repo = repo_dir.name

        for skill_md in repo_dir.rglob("SKILL.md"):
            parsed = parse_skill_md(skill_md)
            if parsed is None:
                continue

            rel_path = str(skill_md.relative_to(repo_dir))
            record = {
                "id": make_id(source_repo, rel_path),
                "name": parsed["name"],
                "description": parsed["description"],
                "source_repo": source_repo,
                "source_path": rel_path,
                "license": parsed["license"],
            }
            records.append(record)

    return records


def dedupe(records):
    """Drop near-identical entries (same normalized name+description text).
    Keeps the first occurrence; logs how many were dropped and from where."""
    seen = {}
    deduped = []
    dropped = 0

    for r in records:
        key = normalize_text_for_dedup(r["name"], r["description"])
        if key in seen:
            dropped += 1
            continue
        seen[key] = r["id"]
        deduped.append(r)

    return deduped, dropped


def stratified_sample(records, limit, seed=42, force_full_repos=None):
    """Sample down to `limit` records.

    Repos named in `force_full_repos` (e.g. the small official Anthropic
    repo) are always kept 100% intact, regardless of proportional math,
    since they're the highest-confidence/highest-quality source. The
    remaining budget is then split proportionally by size across whatever
    repos are left, so no single large repo dominates the sample.
    """
    rng = random.Random(seed)
    force_full_repos = set(force_full_repos or [])

    by_repo = {}
    for r in records:
        by_repo.setdefault(r["source_repo"], []).append(r)

    total = len(records)
    if limit >= total:
        return records

    kept = []
    remaining_limit = limit
    proportional_repos = {}

    for repo, items in by_repo.items():
        if repo in force_full_repos:
            kept.extend(items)
            remaining_limit -= len(items)
        else:
            proportional_repos[repo] = items

    if remaining_limit < 0:
        raise ValueError(
            f"force_full_repos alone ({limit - remaining_limit} skills) "
            f"already exceed --limit {limit}; raise --limit or drop one "
            f"from force_full_repos."
        )

    remaining_total = sum(len(v) for v in proportional_repos.values())
    allocated = 0
    shares = {}
    for repo, items in proportional_repos.items():
        share = round(remaining_limit * len(items) / remaining_total) if remaining_total else 0
        share = min(share, len(items))
        shares[repo] = share
        allocated += share

    # Rounding can leave us a few short of the target; hand any leftover
    # slots to the largest remaining repo (it has the most to spare).
    leftover = remaining_limit - allocated
    if leftover > 0 and proportional_repos:
        biggest_repo = max(proportional_repos, key=lambda r: len(proportional_repos[r]))
        shares[biggest_repo] = min(
            shares[biggest_repo] + leftover, len(proportional_repos[biggest_repo])
        )

    for repo, items in proportional_repos.items():
        items_copy = list(items)
        rng.shuffle(items_copy)
        kept.extend(items_copy[: shares[repo]])

    rng.shuffle(kept)
    return kept[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the final corpus to N skills, sampled proportionally "
             "across source repos (small repos kept in full first).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_records = collect_skills()
    print(f"Found {len(raw_records)} SKILL.md files with valid frontmatter.")

    per_repo = {}
    for r in raw_records:
        per_repo[r["source_repo"]] = per_repo.get(r["source_repo"], 0) + 1
    for repo, count in sorted(per_repo.items()):
        print(f"  {repo}: {count}")

    deduped, dropped = dedupe(raw_records)
    print(f"Deduped: dropped {dropped} near-identical entries.")
    print(f"After dedup: {len(deduped)}")

    final = deduped
    if args.limit is not None and args.limit < len(deduped):
        final = stratified_sample(
            deduped, args.limit, seed=args.seed, force_full_repos=SOURCE_PRIORITY
        )
        final_per_repo = {}
        for r in final:
            final_per_repo[r["source_repo"]] = final_per_repo.get(r["source_repo"], 0) + 1
        print(f"Sampled down to {len(final)} (--limit {args.limit}):")
        for repo, count in sorted(final_per_repo.items()):
            print(f"  {repo}: {count}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"Final corpus size: {len(final)}")
    print(f"Wrote corpus to {OUT_PATH}")


if __name__ == "__main__":
    main()
