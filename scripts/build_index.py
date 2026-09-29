#!/usr/bin/env python3
"""
build_index.py

Embeds every skill in data/skills_corpus.json and saves a fast-lookup index.

Default (single-vector, MiniLM):
    python scripts/build_index.py

Swap embedding model (Lever B):
    python scripts/build_index.py --model BAAI/bge-small-en-v1.5
    python scripts/build_index.py --model thenlper/gte-small

Multi-vector mode (Lever D) — uses trigger_phrases field from expanded corpus:
    python scripts/build_index.py --multi-vec [--corpus data/skills_corpus_expanded.json]

Outputs (suffix derived from model; empty = MiniLM default):
    data/skills_index{suffix}.npz
    data/skills_index{suffix}_meta.json

In multi-vector mode each skill emits (1 + N_phrases) rows, all sharing the
same skill_id. stage1() detects the 'skill_id' field and aggregates by MAX.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT        = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "skills_corpus.json"
BATCH_SIZE  = 64

# Maps model names → short suffix used in filenames
MODEL_SUFFIX = {
    "all-MiniLM-L6-v2":       "",       # default; no suffix → backward compat
    "BAAI/bge-small-en-v1.5": "_bge",
    "thenlper/gte-small":      "_gte",
}


def skill_text(skill: dict) -> str:
    return f"{skill['name']}: {skill['description']}"


def build_rows_single(corpus: list[dict]) -> tuple[list[str], list[dict]]:
    """One vector per skill (description only)."""
    texts = [skill_text(s) for s in corpus]
    meta  = [
        {
            "id":          s["id"],
            "name":        s["name"],
            "description": s["description"],
            "source_repo": s["source_repo"],
        }
        for s in corpus
    ]
    return texts, meta


def build_rows_multi(corpus: list[dict], phrase_field: str = "trigger_phrases") -> tuple[list[str], list[dict]]:
    """
    One vector per (skill, text-variant): description + each trigger phrase.
    Meta entries carry 'skill_id' so stage1() knows to aggregate by MAX.
    phrase_field: which field to read phrases from (default: trigger_phrases).
    """
    texts = []
    meta  = []
    missing_phrases = 0

    for s in corpus:
        base_meta = {
            "skill_id":    s["id"],
            "id":          s["id"],
            "name":        s["name"],
            "description": s["description"],
            "source_repo": s["source_repo"],
        }
        # Description vector
        texts.append(skill_text(s))
        meta.append({**base_meta, "vector_type": "description"})

        # Trigger-phrase vectors
        phrases = s.get(phrase_field, [])
        if not phrases:
            missing_phrases += 1
        for pi, phrase in enumerate(phrases):
            texts.append(phrase)
            meta.append({**base_meta, "vector_type": f"phrase_{pi}"})

    if missing_phrases:
        print(f"  [WARN] {missing_phrases}/{len(corpus)} skills have no {phrase_field!r}")
    return texts, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--multi-vec", action="store_true",
        help="Emit one vector per description + trigger phrase (Lever D)",
    )
    parser.add_argument(
        "--corpus", default=str(CORPUS_PATH),
        help=f"Corpus JSON file (default: {CORPUS_PATH})",
    )
    parser.add_argument(
        "--phrase-field", default="trigger_phrases",
        help="Which field to read trigger phrases from (default: trigger_phrases). "
             "Use trigger_phrases_v2 for the 8-phrase implicit-heavy set.",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"Corpus not found: {corpus_path}")

    print(f"Loading corpus from {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  {len(corpus)} skills")

    # Derive output suffix
    suffix = MODEL_SUFFIX.get(args.model, "_" + args.model.split("/")[-1].lower())
    if args.multi_vec:
        suffix += "_multi"
        pf = args.phrase_field
        if pf != "trigger_phrases":
            suffix += f"_{pf.replace('trigger_phrases_', '')}"

    vec_path  = ROOT / "data" / f"skills_index{suffix}.npz"
    meta_path = ROOT / "data" / f"skills_index{suffix}_meta.json"

    # Build text rows and meta
    if args.multi_vec:
        pf = args.phrase_field
        print(f"Multi-vector mode: description + {pf!r} per skill")
        texts, meta = build_rows_multi(corpus, phrase_field=pf)
    else:
        texts, meta = build_rows_single(corpus)
    print(f"  {len(texts)} text rows to embed")

    print(f"Loading model '{args.model}' …")
    model = SentenceTransformer(args.model)

    print(f"Embedding (batch_size={BATCH_SIZE}) …")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1)
    print(f"  shape={vectors.shape}  norm=[{norms.min():.4f},{norms.max():.4f}]")

    ROOT.joinpath("data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vec_path, vectors=vectors)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved → {vec_path}")
    print(f"Saved → {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
