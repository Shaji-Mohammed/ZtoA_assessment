#!/usr/bin/env python3
"""
build_lexical_index.py  — Change 1: BM25 lexical index over skill text fields

Builds a BM25Okapi index from the expanded corpus.  Each skill is tokenised
from three sources, weighted by repetition:
  - name tokens  × 3  (exact-name match gets a strong boost)
  - description tokens × 1
  - trigger_phrase tokens × 1 each

Saves a single pickle to  data/lexical_index.pkl.

Usage:
    python scripts/build_lexical_index.py
    python scripts/build_lexical_index.py --corpus data/skills_corpus.json
"""

import argparse
import json
import pickle
import re
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
EXPANDED_PATH = ROOT / "data" / "skills_corpus_expanded.json"
CORPUS_PATH   = ROOT / "data" / "skills_corpus.json"
INDEX_PATH    = ROOT / "data" / "lexical_index.pkl"


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace+punctuation tokenizer, splits on hyphens too."""
    # split on non-alphanumeric, keep tokens of length >= 2
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) >= 2]


def build_doc_tokens(skill: dict, phrase_field: str = "trigger_phrases") -> list[str]:
    """
    Combine name (3×), description (1×), and trigger phrases (1× each) into
    one token list.  Name repetition boosts exact skill-name matches without
    adding a separate field-aware BM25 (which would require BM25L/BM25Plus).
    phrase_field: which phrase field to include (default: trigger_phrases).
    """
    tokens  = tokenize(skill["name"]) * 3
    tokens += tokenize(skill["description"])
    for phrase in skill.get(phrase_field, []):
        tokens += tokenize(phrase)
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", default=str(EXPANDED_PATH),
        help="Corpus JSON (default: skills_corpus_expanded.json)",
    )
    parser.add_argument(
        "--phrase-field", default="trigger_phrases",
        help="Which phrase field to include in BM25 tokens (default: trigger_phrases)",
    )
    parser.add_argument(
        "--out", default=str(INDEX_PATH),
        help=f"Output pickle path (default: {INDEX_PATH})",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        corpus_path = CORPUS_PATH
        print(f"Expanded corpus not found, falling back to: {corpus_path}")

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} skills from {corpus_path.name}")

    from rank_bm25 import BM25Okapi

    doc_tokens = [build_doc_tokens(s, phrase_field=args.phrase_field) for s in corpus]
    skill_ids  = [s["id"] for s in corpus]
    skill_meta = [
        {
            "id":          s["id"],
            "name":        s["name"],
            "description": s["description"],
            "source_repo": s["source_repo"],
        }
        for s in corpus
    ]

    print("Fitting BM25Okapi …")
    bm25 = BM25Okapi(doc_tokens)

    index_data = {
        "bm25":       bm25,
        "skill_ids":  skill_ids,
        "skill_meta": skill_meta,
    }
    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        pickle.dump(index_data, f, protocol=5)

    total_tokens = sum(len(t) for t in doc_tokens)
    print(f"Saved → {out_path}")
    print(f"  {len(skill_ids)} skills   {total_tokens:,} total tokens   "
          f"avg {total_tokens/len(skill_ids):.0f} tokens/skill")


if __name__ == "__main__":
    main()
