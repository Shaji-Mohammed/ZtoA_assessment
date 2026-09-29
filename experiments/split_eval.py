#!/usr/bin/env python3
"""
split_eval.py — One-time stratified 80/20 split of eval_set.json

Outputs:
  data/eval_tune.json     (80% — used for all lever tuning)
  data/eval_holdout.json  (20% — untouched until final validation)

Stratified by trigger_type so both sets preserve the explicit/implied ratio.
Deterministic: seed=42.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
EVAL_PATH    = ROOT / "data" / "eval_set.json"
TUNE_PATH    = ROOT / "data" / "eval_tune.json"
HOLDOUT_PATH = ROOT / "data" / "eval_holdout.json"


def main():
    with open(EVAL_PATH) as f:
        records = json.load(f)

    rng = random.Random(42)

    by_type = defaultdict(list)
    for r in records:
        by_type[r.get("trigger_type", "explicit")].append(r)

    tune, holdout = [], []
    for ttype, group in sorted(by_type.items()):
        shuffled  = rng.sample(group, len(group))
        n_holdout = max(1, round(len(shuffled) * 0.20))
        holdout.extend(shuffled[:n_holdout])
        tune.extend(shuffled[n_holdout:])

    # Shuffle each set so ordering isn't correlated with type
    rng.shuffle(tune)
    rng.shuffle(holdout)

    def ttype_counts(lst):
        c = defaultdict(int)
        for r in lst:
            c[r.get("trigger_type", "explicit")] += 1
        return dict(c)

    print(f"Total    : {len(records)}")
    print(f"Tune     : {len(tune)}     {ttype_counts(tune)}")
    print(f"Holdout  : {len(holdout)}  {ttype_counts(holdout)}")

    with open(TUNE_PATH, "w") as f:
        json.dump(tune, f, indent=2)
    with open(HOLDOUT_PATH, "w") as f:
        json.dump(holdout, f, indent=2)
    print(f"\nWrote {TUNE_PATH}")
    print(f"Wrote {HOLDOUT_PATH}")


if __name__ == "__main__":
    main()
