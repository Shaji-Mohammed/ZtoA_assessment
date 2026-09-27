#!/usr/bin/env python3
"""
build_index.py

Embeds every skill in data/skills_corpus.json using all-MiniLM-L6-v2,
L2-normalises the vectors so cosine similarity reduces to a dot product at
query time, and caches results to:

  data/skills_index.npz        — float32 matrix, shape (N, 384)
  data/skills_index_meta.json  — list of {id, name, description, source_repo}
                                  row-order matches the matrix

Run once after building the corpus, or re-run whenever the corpus changes.

Usage:
    python scripts/build_index.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT         = Path(__file__).resolve().parent.parent
CORPUS_PATH  = ROOT / "data" / "skills_corpus.json"
VECTORS_PATH = ROOT / "data" / "skills_index.npz"
META_PATH    = ROOT / "data" / "skills_index_meta.json"

MODEL_NAME   = "all-MiniLM-L6-v2"
BATCH_SIZE   = 64


def skill_text(skill: dict) -> str:
    """Single string fed to the encoder for each skill."""
    return f"{skill['name']}: {skill['description']}"


def main():
    if not CORPUS_PATH.exists():
        sys.exit(f"Corpus not found: {CORPUS_PATH}\nRun scripts/build_corpus.py first.")

    print(f"Loading corpus from {CORPUS_PATH}")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  {len(corpus)} skills")

    print(f"Loading model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    texts = [skill_text(s) for s in corpus]

    print(f"Embedding {len(texts)} skills (batch_size={BATCH_SIZE}) …")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise: cosine sim == dot product
    )
    vectors = vectors.astype(np.float32)
    print(f"  vectors shape: {vectors.shape}  dtype: {vectors.dtype}")

    # Sanity-check: norms should all be ~1.0
    norms = np.linalg.norm(vectors, axis=1)
    print(f"  norm range: [{norms.min():.4f}, {norms.max():.4f}]  (should be ~1.0)")

    ROOT_data = CORPUS_PATH.parent
    ROOT_data.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(VECTORS_PATH, vectors=vectors)
    print(f"Saved vectors → {VECTORS_PATH}")

    meta = [
        {
            "id":          s["id"],
            "name":        s["name"],
            "description": s["description"],
            "source_repo": s["source_repo"],
        }
        for s in corpus
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata → {META_PATH}")

    print("\nDone. Index is ready.")


if __name__ == "__main__":
    main()
