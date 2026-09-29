# Skill Router CLI

Routes a natural-language task prompt to the skill(s) needed to complete it, from a corpus of 534 skills gathered from public repositories.

Built as a two-stage pipeline: local embedding retrieval narrows the corpus, then a cheap LLM call selects the final set. The design target was low latency, minimal cost, and 95%+ accuracy. **We did not reach 95%** — most of this README explains what we measured instead, and why the gap is structural rather than a tuning problem.

---

## Quick start

```bash
git clone <repo-url> && cd skillrouter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add your OPENROUTER_API_KEY

python scripts/build_corpus.py    # gather + normalize skills
python scripts/build_index.py               # embed corpus, cache vectors
python scripts/skillrouter.py query "convert this spreadsheet into a slide deck"
```

---

## Architecture

```
prompt
  │
  ├─ Stage 1  embed query (all-MiniLM-L6-v2, local)
  │           hybrid score: 0.7 × cosine + 0.3 × normalized BM25
  │           relative gate: cutoff = max(0.85 × top_score, 0.20)
  │           return all skills above cutoff, capped at 40
  │           single-vector index: one embedding per skill (description)
  │             → free, no network call, ~17ms p50 warm
  │
  └─ Stage 2  send prompt + candidates to gpt-4o-mini via OpenRouter
              model returns the minimal required subset (5-shot prompted)
              SKIPPED when Stage 1 returns one clearly dominant match
                → ~$0.0002/call, invoked on ~59% of queries
```

**Why two stages.** Sending all 534 skill descriptions to an LLM per query would be slow, expensive, and _less_ accurate — long-context ranking degrades as the candidate list grows. Embeddings are near-free and handle the easy majority; the LLM only ever sees a short list.

**Why the cap.** Some corpus regions are dense enough that a threshold alone returns hundreds of candidates. One prompt pulled 316. A hard top-40 cap bounds worst-case cost and keeps the Stage 2 context small. Median candidate count is 5, so normal queries are unaffected.

**Why Stage 2 can be skipped.** When exactly one candidate clears the threshold and the gap to second place is ≥ 0.15, there is nothing to decide. This keeps the cheap path cheap.

**Why hybrid retrieval.** BM25 catches exact tool-name matches that cosine similarity undersells (e.g., "PhantomBuster" → `phantombuster-automation`). The 0.7/0.3 split was the best-performing ratio tested and costs nothing at runtime beyond a prebuilt index.

---

## Data sources

| Repo                               | Skills   | Notes                                    |
| ---------------------------------- | -------- | ---------------------------------------- |
| `anthropics/skills`                | 19       | Official; highest description quality    |
| `alirezarezvani/claude-skills`     | 432      | Broad cross-tool collection              |
| `ComposioHQ/awesome-claude-skills` | 83       | After dedup; boilerplate detection pass  |
| **Total**                          | **534**  | Distinct after dedup and collapse        |

Normalized to one schema: `{id, name, description, source_repo, source_path, license}`.

### Corpus decisions

Two large clusters were removed or collapsed, both for the same reason: **skills that are indistinguishable by description cannot be routed between, by any method.**

- **382 NVIDIA skills dropped entirely.** Added mid-project, then removed after measurement (see [Finding 4](#finding-4-aggregate-similarity-hides-dense-clusters)). Pre-exclusion backup preserved in `data/skills_corpus_v1.json`; sub-cluster density analysis in `data/corpus_analysis_nvidia.json`.
- **Composio near-boilerplate skills flagged.** Skills with placeholder descriptions ("toolkit not currently available in Composio") were detected but were already informative enough to retain. A `composio-automation` canonical entry covers the remaining generic routing case. Full analysis in `data/corpus_analysis_composio.json`.
- **Cross-repo duplicates deduplicated by name**, prioritizing `anthropics` > `alirezarezvani` > `composio`. Before this, a duplicate scoring above threshold could mask a Stage 1 miss and inflate accuracy.
- **Three descriptions rewritten** where they described _when_ a skill triggers rather than _what it does_ (`discernment-nudge`, `karpathy-check`, `internal-comms`). Originals preserved in a `description_original` field for auditability.

---

## How accuracy is defined

The brief specified "95%+ accuracy" without defining it, and clarified that prompts may require more than one skill and that routing must hit all correct ones. We measure:

**Exact-set match** — the returned set equals the ground-truth set. Strict: one missing or one extra skill is a failure. This is the headline number.

Reported alongside it, as diagnostics:

| Layer                         | What it isolates                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| End-to-end exact match        | Real performance, all prompts                                                      |
| Stage-1-reachable exact match | Stage 2's decision quality, excluding prompts where the right skill never surfaced |
| Embedding limitation rate     | % of prompts where Stage 1 structurally cannot surface a ground-truth skill        |

The three-layer split matters because a single number conflates two unrelated failures: _the router chose badly_ and _the router was never shown the right option_. Only the first is tunable at the routing layer.

### Ground truth: required vs helpful

Ground-truth labels apply an explicit rule:

> A skill is **REQUIRED** if the task explicitly mentions or clearly implies the deliverable, format, platform, or action that skill provides, **and** the task cannot be completed to its stated specification without it.
> A skill is **HELPFUL** (excluded) if it is adjacent or would improve quality, but the task does not invoke its specific capability.
> For N skills, the prompt must contain N distinct triggers. If a skill's role must be inferred, it is HELPFUL.

**This rule has a known bias, stated plainly:** it makes ground truth largely keyword-triggered. A prompt saying "PPTX" makes `pptx` required; one saying "something for the board meeting" does not. This was a deliberate trade — the previous, looser labels produced failures that were really human disagreements about "required," which is unmeasurable. To keep that bias visible rather than hidden, the eval set separates **explicit-trigger** prompts from **implied-need** prompts and reports them separately. See [Finding 3](#finding-3-the-router-is-largely-keyword-triggered).

---

## Results

Current configuration: `hybrid`, `alpha=0.85`, `cap=40`, `gpt-4o-mini`, `fewshot` (5 examples), corpus 534 skills, single-vector index (534 vectors).

**Headline: 75.0% (90/120) exact-set match on clean_test — 95% CI [66.6%, 81.9%].**

Three-split summary:

| Split | Config | N | Exact | 95% CI | Notes |
| ----- | ------ | --- | ----- | ------ | ----- |
| **clean_test** (seed=99) | baseline | 120 | **75.0%** | [66.6%, 81.9%] | **Headline** — independent sample, no overlap with dev or final_test |
| final_test | v3 | 124 | 85.5% | [78.2%, 90.6%] | **CONTAMINATED** — 13 skills selected from this split's MISS_STAGE1 failures; the gain was fitting, not generalization |
| dev (K=40) | v3 | 220 | 63.6% | [57.1%, 69.7%] | Phrases not fitted to dev failures; the unchanged dev score is what exposed the contamination |

See [Detecting and correcting test-set contamination](#detecting-and-correcting-test-set-contamination) for the full analysis.

By trigger type (clean_test, baseline config):

| Type | n | Exact match |
| ---- | --- | ----------- |
| Explicit (prompt names the format/tool) | 90 | **83.3%** (75/90) |
| Implied (prompt describes the goal only) | 30 | **50.0%** (15/30) |

Failure taxonomy on clean_test (non-exclusive; one prompt can have multiple labels):

| Label | Count | Meaning |
| ----- | ----- | ------- |
| MISS_STAGE1 | 21 | Ground-truth skill never retrieved by Stage 1 |
| OVERROUTING | 30 | Returned correct skills plus at least one extra |
| WRONG_STAGE2 | 5 | GT was in Stage 1 candidates; Stage 2 dropped it |
| ABSTAIN | 1 | Stage 2 returned an empty set |

**Latency** (baseline config, measured on clean_test run).

| Path | p50 | p95 |
| ---- | --- | --- |
| Stage 1 only | 17 ms | 48 ms |
| Stage 1 + Stage 2 | 993 ms | 1721 ms |

Stage 2 invoked on 59.2% (71/120) of queries; skipped on 40.8%.

**Cost.** Measured, not estimated — every call's cost comes from the OpenRouter response and is logged to a tracker with a hard halt at $3.00. Total spend across the entire project, including all experiments: **~$2.40**. Per-query cost at the current config (~59% Stage 2 rate, gpt-4o-mini): **~$0.099/1k queries**.

**Legacy holdout** (held out before any tuning; burned once, at the end): **54.2%** (13/24), 95% CI [35%, 72%]. Explicit: 72.2% (13/18), Implied: 0.0% (0/6). The CIs overlap with the clean_test result ([66.6%, 81.9%]) — the gap is consistent with small-sample noise rather than a contradiction.

---

## Detecting and correcting test-set contamination

During the v3 experiment, the 13 skills to receive new trigger phrases were selected by inspecting MISS_STAGE1 failures on `final_test`. Six intent-oriented phrases were generated per skill, the index was rebuilt, and the system was immediately re-evaluated on the same `final_test`. The result was 85.5% — a 15.3pp improvement over baseline.

**The divergence that exposed the fitting.** The development split, whose failures were never used to select which skills to tune, moved from 61.4% to 63.6% — a 2.2pp change. Genuine retrieval improvement should lift both splits proportionally. A 22pp gap, with the unselected split unmoved, is a contamination signature.

**Confirming with a matched sample.** A fresh independent test set (seed=99, n=120) was generated using the same `build_eval.py` generator. Both the v3 config and the original single-vector baseline were evaluated on it:

| Config | N | Exact | 95% CI |
| ------ | --- | ----- | ------ |
| Baseline (single-vector, v1 phrases) | 120 | 75.0% (90/120) | [66.6%, 81.9%] |
| v3 (multi-vector, intent-oriented phrases) | 120 | 75.8% (91/120) | [67.4%, 82.6%] |

**Result: +0.8pp (one prompt). Confidence intervals fully overlap. The entire 15.3pp apparent gain was test-set fitting.** The new phrases were tuned to the exact prompts being measured, not to the underlying distribution.

**Consequence.** The v3 phrases provided no measured benefit on unseen data and were selected from a contaminated split. The shipped config reverts to single-vector baseline (`skills_index.npz`, 534 vectors, description-only embeddings). The null result covers the v3 phrase set; whether multi-vector indexing itself would help with v1 phrases on unseen data was not separately tested.

**Methodological lesson.** When selecting what to tune based on failures in a test split, the resulting improvement must be measured on a different split. The correct protocol: observe failures on dev → design intervention → measure on held-out clean set. The contamination was detected here because the dev signal was available and unmoved — without a second split, the inflated number would have been invisible.

---

## Example queries

```
$ python scripts/skillrouter.py query "extract the tables from this PDF and put them in a spreadsheet"

Query:     'extract the tables from this PDF and put them in a spreadsheet'
Stage 1 — 2 candidate(s) above threshold

Final skill set (2):
  [0.4399] pdf
  [0.3333] Excel Automation

JSON:
[
  {"id": "aac3216b9b78", "name": "pdf"},
  {"id": "4fb254d75c5c", "name": "Excel Automation"}
]
```

```
$ python scripts/skillrouter.py query "write an announcement for the team about our new pricing"

Query:     'write an announcement for the team about our new pricing'
Stage 1 — 18 candidate(s) above threshold

Final skill set (2):
  [0.4793] pricing-strategy
  [0.3713] team-communications

JSON:
[
  {"id": "2f3fc623a1b1", "name": "pricing-strategy"},
  {"id": "da8123523541", "name": "team-communications"}
]
```

```
$ python scripts/skillrouter.py query "review my staged changes before I commit"

Query:     'review my staged changes before I commit'
Stage 1 — 10 candidate(s) above threshold

Final skill set (3):
  [0.5793] karpathy-check
  [0.5237] cs-karpathy-reviewer
  [0.3683] karpathy-coder

JSON:
[
  {"id": "bfb07fe55192", "name": "karpathy-check"},
  {"id": "419bf881944a", "name": "cs-karpathy-reviewer"},
  {"id": "72a331fc4eb0", "name": "karpathy-coder"}
]
```

The third example illustrates overrouting: the Karpathy-check corpus region has three closely-described skills. Stage 2 correctly identifies all three as relevant, but a strict exact-match evaluator penalizes this if only one was labeled.

---

## What we tried, and what it told us

Each lever was tested in isolation against a fixed baseline.

### Finding 1: False-positive over-inclusion is the dominant failure, and it is invariant

Across every experiment, the same failure dominated: Stage 2 includes semantically _adjacent_ skills alongside the correct ones — `+pptx` with `brand-guidelines`, `+docx` with `pdf`, `+claude-coach` with `academy-guide`. Recall was often perfect; precision was not.

| Lever tested                                                      | Effect on FP-only failures                                                                               |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Tightened Stage 2 prompt ("minimal set", "when in doubt exclude") | 14 → 11, but created 12 new false-negative failures. Net zero on exact match, worse on recall.           |
| Stronger model (`gpt-4o`, ~13× cost)                              | 14 → 14. Identical. Slightly worse end-to-end (−3.7pp).                                                  |
| Lower threshold (0.30 → 0.25)                                     | 14 → 20. Halved the embedding ceiling but every recovered prompt became an FP failure. Exact match fell. |
| Larger corpus (264 → 916)                                         | FP-only failures roughly doubled.                                                                        |

**What this rules out.** Two model tiers, two prompt framings, and two thresholds produce the same over-inclusion. This is not a model-capability problem and not a prompt-engineering problem. The ambiguity lives in the skill descriptions themselves — when two skills genuinely overlap in what they describe, no amount of reasoning reliably separates "required" from "would help."

### Finding 2: Corpus scale degrades accuracy

Explicit-prompt accuracy fell from 57.4% to 29.2% when the corpus grew 3.5× (264 → 916). More skills means more plausible-looking candidates in the Stage 2 window, and a higher chance one of them gets included. Any claim about router accuracy is meaningless without stating corpus size.

### Finding 3: The router is largely keyword-triggered

Splitting the eval set by trigger type produced the most useful number in the project:

| Type                                     | Exact match (clean_test, N=120) |
| ---------------------------------------- | ------------------------------- |
| Explicit (prompt names the format/tool)  | 83.3% (75/90)                   |
| Implied (prompt describes the goal only) | 50.0% (15/30)                   |

When a user says _what they want_ rather than _what tool they need_, routing largely fails. This is partly an artifact of the labeling rule above, and partly real: embedding similarity between "streamline AI interactions with external services" and a skill described as `mcp-builder` is genuinely low. **This is the most important open problem in the system**, and we measured it deliberately rather than letting it hide inside an aggregate.

### Finding 4: Aggregate similarity hides dense clusters

When the NVIDIA repo was added, corpus-wide mean pairwise similarity looked benign — 0.208, essentially identical to the official Anthropic repo's 0.213. Sub-cluster analysis told the opposite story. The numbers below are computed from the 382 ingested skills in `data/skills_corpus_v1.json` using the same `all-MiniLM-L6-v2` model used at runtime (full descriptions, up to 20 skills sampled per cluster for pairwise averaging):

| Sub-cluster  | n   | Mean pairwise cosine |
| ------------ | --- | -------------------- |
| dicom        | 3   | 0.677                |
| earth2studio | 7   | 0.652                |
| digital      | 4   | 0.636                |
| dynamo       | 4   | 0.619                |
| holoscan     | 6   | 0.612                |
| holohub      | 3   | 0.603                |
| nv           | 9   | 0.598                |
| rtvi         | 3   | 0.566                |
| kermt        | 8   | 0.542                |
| tilegym      | 7   | 0.529                |
| rag          | 3   | 0.529                |
| deepstream   | 6   | 0.518                |
| warp         | 3   | 0.489                |
| nemotron     | 6   | 0.480                |
| doca         | 60  | 0.476                |
| amc          | 4   | 0.468                |
| vss          | 15  | 0.445                |
| hsb          | 7   | 0.435                |
| i4h          | 14  | 0.428                |
| nvflare      | 9   | 0.427                |
| paidf        | 5   | 0.422                |

21 of 28 clusters exceeded 0.40 — every one denser than any cluster in the non-NVIDIA portion of the corpus. The aggregate averaged this away because DOCA skills are dissimilar to NeMo skills. Full table: `data/corpus_analysis_nvidia.json`.

**Generalizable takeaway: corpus-wide similarity is the wrong gate for corpus selection. Intra-cluster similarity is the right one.** This check is now part of the selection criteria below.

### Finding 5: Eval sets contain their own defects

10 of the original 64 generated eval prompts were degenerate — the generating model wrote prompts that mentioned a skill's topic without describing its function, making the ground truth unreachable at any threshold. Removing them raised the measured Stage 1 recall ceiling from 81.2% to 92.6% with **no change to the router at all.** A meaningful share of apparent model failure was measurement failure. Removal used a stated criterion (ground-truth similarity below a floor), applied before seeing results — not "the router got this one wrong."

### Finding 6: Relative gating is a more stable gate than fixed threshold

Alpha sweep (α=0.80/0.85/0.90/0.95, everything else frozen):

| α    | Exact match | MISS_STAGE1 | OVERROUTING |
| ---- | ----------- | ----------- | ----------- |
| 0.80 | 69.4% (86/124) | 25 | 34 |
| 0.85 | **70.2% (87/124)** | 28 | 33 |
| 0.90 | 67.7% (84/124) | 36 | 36 |
| 0.95 | 66.1% (82/124) | 41 | 38 |

Raising alpha does not convert over-routing failures into exact matches — it converts them into MISS_STAGE1 failures instead. Alpha is not a precision lever; it is a recall lever with a precision illusion. α=0.85 is the best-measured setting and the frozen default.

---

## Why 95% was not reached

The gap decomposes into two independent ceilings that compound:

1. **Stage 1 recall ceiling.** 16.7% of prompts (20/120 on clean_test) have a ground-truth skill that never surfaces at any usable threshold. Lowering the threshold to catch them returns most of the corpus for other queries, which destroys Stage 2 precision. Driven by descriptions written in internal jargon rather than user language.
2. **Stage 2 precision floor.** Over-inclusion of adjacent skills, shown above to be invariant to model, prompt, and threshold. Drives 30/120 OVERROUTING failures on clean_test. Driven by genuine semantic overlap between skill descriptions.

Neither is a routing-logic problem. Both are **corpus quality problems** — and the corpus is what it is, because it is real public data rather than a curated benchmark.

**The two sharpest remaining bottlenecks** are visible in the clean_test breakdown: **implied-need routing is at 50.0%** (vs 83.3% explicit), and **overrouting accounts for 30 failures with at least one false positive**. These are the next levers.

**What would actually close the gap**, in rough order of expected impact:

- **Implied-need routing.** The 33pp gap between explicit (83.3%) and implied (50.0%) is the single largest open problem. Embedding similarity between a user's goal description and a jargon-named skill is structurally low. Fine-tuning the embedding model on synthetic query→skill pairs, or generating one natural-language "user would say" sentence per skill, directly targets this.
- **Rewrite descriptions at scale.** Most skills describe themselves to a developer reading a repo, not to a router matching a user's phrasing. A normalized functional description plus example trigger phrases per skill targets both ceilings at once — but only if the phrases are validated on held-out data, not the set used to select them.
- **Sharpen or collapse the remaining dense clusters.** After the NVIDIA removal, `alirezarezvani` accounts for ~70% of remaining false-positive slots via overlapping business-strategy skills (`marketing-strategy`, `launch-strategy`, `pricing-strategy`, `cmo-advisor`).
- **Hierarchical routing.** Route to a domain first, then within it. Dense clusters become a local problem instead of a global one.
- **A stronger embedding model** (`bge-small-en-v1.5` or similar). Cheapest untested lever; still local, still free.

---

## Corpus selection criteria

Derived from Finding 4, applied before ingesting any new repo:

1. **Semantic distinctness** — no name-prefix or topic sub-cluster with 5+ members above 0.40 mean pairwise cosine similarity. Corpus-wide mean is not sufficient.
2. **Functional description** — sample 20 skills; reject if >30% require product-specific knowledge to understand what they do.
3. **Domain spread** — no single domain past 35% of target corpus size.
4. **Size discipline** — target 400–600 distinct skills after all transformations.

---

## Tradeoffs

| Decision                               | Chose                                     | Gave up                                                              |
| -------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| Local embeddings for Stage 1           | Zero marginal cost, ~13ms p50, no rate limits | Weaker semantics than a hosted embedding API                     |
| `gpt-4o-mini` for Stage 2              | ~$0.09/1k queries                         | Nothing measurable — `gpt-4o` at 13× cost performed _worse_         |
| Hybrid retrieval (0.7 cosine + 0.3 BM25) | +0.8pp over cosine-only, catches exact tool names | Slightly longer index build                                  |
| Multi-vector index (tried, reverted)   | Principled extension of single-vector     | No measured benefit on clean data (+0.8pp, not significant); reverted to single-vector |
| Relative alpha gate over fixed threshold | More stable across corpus density changes | One more hyperparameter to explain                                  |
| Strict exact-set match                 | Honest, unambiguous metric                | A friendlier number; partial credit would read much better           |
| Collapsing duplicate clusters          | A corpus that can actually be routed      | Raw skill count, and per-service granularity                         |
| Keyword-leaning ground-truth rule      | Consistent, reproducible labels           | Some validity — measured and reported via the explicit/implied split |
| Stopping tuning after six experiments  | A documented, well-understood ceiling     | Possible further gains from corpus rewriting                         |

---

## Known limitations

- **Multi-skill coverage is thin.** The final eval set skews single-skill; generating realistic multi-skill prompts requires domain-aware pairing, since randomly paired cross-domain skills do not co-occur in any plausible task. Multi-skill accuracy is therefore under-measured.
- **Eval set is LLM-generated**, so it inherits that model's idea of a realistic task. Human-written prompts would be a stronger test.
- **`description_original` is preserved but rewritten descriptions are ours**, not upstream — reported numbers reflect a lightly modified corpus, documented above.
- **Numbers are corpus-specific.** Finding 2 shows accuracy is a function of corpus size and density. These results do not transfer to a different skill set.
- **Near-duplicate label ambiguity.** A small number of failures across splits involve near-identical skills (`pm-skills` cluster, `financial-health`/`financial-analyst`, `discernment-nudge`/`challenge`) where any reasonable prediction of either would satisfy the user's intent. These are labeled failures under exact-set match but not meaningful errors in practice.

---

## Repo layout

```
scripts/
  build_corpus.py       gather, normalize, dedup, collapse
  build_index.py        embed corpus → data/skills_index.npz (single-vector, shipped default)
  build_lexical_index.py  build BM25 index → data/lexical_index.pkl
  skillrouter.py        CLI: query (Stage 1 + Stage 2), cost
  build_eval.py         generate eval prompts + ground truth
  eval.py               score router output, taxonomy breakdown
  rebuild_corpus.py     corpus v2 transform (NVIDIA excl, dedup)
  expand_descriptions.py  generate trigger phrases for index
data/
  skills_corpus.json            normalized corpus (534 skills)
  skills_corpus_v1.json         pre-rebuild backup (916 skills, incl. NVIDIA)
  skills_index.npz              single-vector embeddings (534 vectors) — shipped default
  skills_index_meta.json        id/name/description lookup
  skills_corpus_v3.json         v3 corpus with intent-oriented trigger phrases (not in use)
  skills_index_multi.npz        multi-vector embeddings (2748 vectors, v3) — not in use
  lexical_index.pkl             BM25 index
  eval_final_test.json          final test set (N=124, frozen)
  eval_clean_test.json          clean test set (N=120, seed=99, independent)
  eval_holdout.json             legacy holdout (N=24, burned once)
  FINAL_final_test.json         baseline-config run results
  baseline_clean_test.json      baseline config on clean_test (75.0%)
  v3_clean_test.json            v3 config on clean_test (75.8%)
  FINAL_legacy_holdout.json     holdout run results
  corpus_analysis_nvidia.json   NVIDIA sub-cluster density table
  corpus_analysis_composio.json Composio boilerplate analysis
  .cost_tracker.json            cumulative spend (gitignored)
experiments/
  expand_phrases_v2.py           8-phrase generation experiment (−3.3pp)
  split_eval.py                  one-time eval-set splitting script
  v3_improvement_summary.json    full v3 experiment record incl. contamination finding
  generated_phrases_v3.json      the 6 intent-oriented phrases per 13 skills (contaminated)
```
