#!/usr/bin/env python3
"""
eval.py — Offline evaluator for the skill router

Subcommands:
  calibrate        Sweep Stage-1 thresholds; measure recall ceiling (free, local).
  stage2-dry-run   Run Stage 2 on N sampled prompts; report real costs + extrapolation.
  run              Full eval: Stage 1 + Stage 2 on every prompt.

Usage:
  python scripts/eval.py calibrate [--eval-path data/eval_tune.json] [--cap 40]
  python scripts/eval.py stage2-dry-run [--n 8] [--threshold 0.30]
  python scripts/eval.py run [--split development] [--alpha 0.80] [--hybrid]
                             [--conservative-agg] [--prompt-variant fewshot|rhi]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT              = Path(__file__).resolve().parent.parent
EVAL_PATH_DEFAULT = ROOT / "data" / "eval_set.json"
COST_PATH         = ROOT / "data" / ".cost_tracker.json"
EVAL_PATH = EVAL_PATH_DEFAULT   # backwards-compat alias

# ── dataset split paths ───────────────────────────────────────────────────────
SPLIT_PATHS = {
    "development":    ROOT / "data" / "eval_development.json",
    "legacy_holdout": ROOT / "data" / "eval_holdout.json",
    "final_test":     ROOT / "data" / "eval_final_test.json",
}

load_dotenv(ROOT / ".env")

STAGE2_MODEL = "openai/gpt-4o-mini"
HALT_AT_USD  = 3.00

MODEL_PRICING = {
    "openai/gpt-4o-mini":          (0.15  / 1_000_000,  0.60 / 1_000_000),
    "openai/gpt-4o":               (2.50  / 1_000_000, 10.00 / 1_000_000),
    "openai/gpt-4o-2024-11-20":    (2.50  / 1_000_000, 10.00 / 1_000_000),
    "anthropic/claude-sonnet-4.5": (3.00  / 1_000_000, 15.00 / 1_000_000),
    "anthropic/claude-sonnet-4.6": (3.00  / 1_000_000, 15.00 / 1_000_000),
    "anthropic/claude-haiku-4.5":  (0.80  / 1_000_000,  4.00 / 1_000_000),
}
_DEFAULT_PRICING = (1.00 / 1_000_000, 5.00 / 1_000_000)


# ── shared helpers ─────────────────────────────────────────────────────────────

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


def estimate_cost(pt: int, ct: int, model: str = STAGE2_MODEL) -> float:
    in_p, out_p = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return pt * in_p + ct * out_p


def resolve_eval_path(args) -> Path:
    """
    Resolve which eval file to use.
    --split overrides --eval-path if provided.
    Final-test split requires --final-test-approved flag.
    """
    split = getattr(args, "split", None)
    if split:
        if split == "final_test":
            approved = getattr(args, "final_test_approved", False)
            if not approved:
                sys.exit(
                    "\n[GUARD] You requested --split final_test.\n"
                    "This set is NEVER to be inspected until final validation.\n"
                    "Add --final-test-approved to confirm this is a final validation run."
                )
        path = SPLIT_PATHS.get(split)
        if path is None:
            sys.exit(f"Unknown split: {split!r}")
        if not path.exists():
            sys.exit(f"Split file not found: {path}\nGenerate it with scripts/build_eval.py")
        return path
    return Path(getattr(args, "eval_path", str(EVAL_PATH_DEFAULT)))


# ── Stage-2 prompt construction ───────────────────────────────────────────────

STAGE2_SYSTEM = (
    "You are a skill router. Your only job is to select which "
    "skills from a numbered list are needed for a given task. "
    "Always respond with a JSON array of integers and nothing else."
)

STAGE2_SYSTEM_RHI = (
    "You are a skill router. For each candidate skill, classify it as "
    "REQUIRED (the task explicitly needs it; cannot be done without it), "
    "HELPFUL (related but not strictly necessary), or "
    "IRRELEVANT (not needed). "
    "Respond ONLY with a JSON object mapping number-string to label. "
    'Example: {"1": "REQUIRED", "2": "HELPFUL", "3": "IRRELEVANT"}'
)

# ── Few-shot examples (Lever C — kept for fewshot and rhi variants) ───────────
# V1 = original 5 examples; _FEWSHOT_EXAMPLES = V1 + 6 targeted confusable-pair examples
_FEWSHOT_EXAMPLES_V1 = """\
Example 1
Task: Can you help me decide whether to start a LinkedIn newsletter and create a six-month content strategy for it?
Candidates:
1. linkedin-strategy: Use when someone needs a LinkedIn plan rather than a post — content pillars, positioning for a career change or consulting practice, newsletter strategy, or a personal brand roadmap.
2. content-strategy: When the user wants to plan a content strategy, decide what content to create, or figure out what topics to cover. Also use when the user wants to understand their target audience.
Answer: [1]
Reason: The task is specifically about LinkedIn; content-strategy is the generic version. Pick the specific skill.

Example 2
Task: I need a detailed plan for an A/B test to evaluate the effectiveness of two different landing page designs.
Candidates:
1. ab-test-setup: When the user wants to plan, design, or implement an A/B test or experiment. Covers hypothesis, variant design, sample size, and test setup.
2. experiment-designer: Use when planning product experiments, writing testable hypotheses, estimating sample size, or prioritizing tests across a product roadmap.
3. statistical-analyst: Run hypothesis tests, analyze A/B experiment results, calculate sample sizes, and interpret statistical significance.
Answer: [1]
Reason: ab-test-setup covers the full task. experiment-designer and statistical-analyst are supporting skills used after or alongside ab-test-setup, not instead of it.

Example 3
Task: Create a new job listing for a software engineer position and post it on our careers page (we use Ashby).
Candidates:
1. Ashby Automation: Automate recruiting and hiring workflows in Ashby — manage candidates, jobs, applications, interviews, and notes through the Ashby ATS API.
2. Lever Automation: Automate recruiting workflows in Lever ATS — manage opportunities, job postings, requisitions, and pipeline stages.
Answer: [1]
Reason: The task names Ashby explicitly. Lever Automation is a different ATS; it is not needed.

Example 4
Task: Draft a launch plan for our upcoming software release, including key milestones and promotional strategy.
Candidates:
1. marketing-strategy-pmm: Product marketing skill for positioning, GTM strategy, competitive intelligence, and product launches.
2. launch-strategy: When the user wants to plan a product launch, feature announcement, or release strategy. Also use when the user mentions go-to-market timing.
3. marketing-context: Create and maintain the marketing context document that all marketing skills read before starting.
Answer: [1]
Reason: marketing-strategy-pmm covers launch plans end-to-end. launch-strategy overlaps but is narrower; marketing-context is a setup skill, not a deliverable skill.

Example 5
Task: Build an interactive dashboard using Tailwind CSS and shadcn/ui components that visualizes real-time data.
Candidates:
1. web-artifacts-builder: Suite of tools for creating elaborate, multi-component claude.ai HTML artifacts using modern frontend web technologies (React, Tailwind, shadcn/ui).
2. senior-frontend: Frontend development skill for React, Next.js, TypeScript, and Tailwind CSS applications. Use when building React components or a full frontend codebase.
Answer: [1]
Reason: The task asks for a self-contained artifact (a dashboard), not a full application codebase. web-artifacts-builder matches the deliverable; senior-frontend is for full-app development.
"""

# V1 + 6 targeted confusable-pair examples = full fewshot block
_FEWSHOT_EXAMPLES = _FEWSHOT_EXAMPLES_V1 + """\

Example 6
Task: Pull engagement stats from our LinkedIn and Twitter accounts and compile them into a Google Sheet.
Candidates:
1. PhantomBuster Automation: Automate lead generation, web scraping, and social media data extraction workflows through PhantomBuster's cloud platform via Composio. NOTE: this extracts raw data FROM social platforms; for analyzing existing data use social-media-analyzer.
2. social-media-analyzer: Social media campaign analysis and performance tracking. Calculates engagement rates, ROI, and benchmarks. NOTE: this ANALYZES data you already have; for scraping/extracting data from accounts, use PhantomBuster Automation.
Answer: [1]
Reason: The task requires extracting data from social accounts (PhantomBuster's job). social-media-analyzer analyzes data already in hand — it cannot retrieve it from LinkedIn/Twitter.

Example 7
Task: Add a new client to our Toggl workspace and assign them to the 'Agency' project.
Candidates:
1. Toggl Automation: Automate time tracking workflows in Toggl Track — create time entries, manage projects, clients, tags, and workspaces. NOTE: 'workspace' here is Toggl Track's concept, not Google Workspace.
2. cs-workspace-admin: Google Workspace administration agent for Gmail, Google Drive, Sheets, and Calendar. NOTE: specifically for Google products.
Answer: [1]
Reason: The task is in Toggl Track (time tracking). cs-workspace-admin is for Google Workspace — unrelated to Toggl.

Example 8
Task: Organize last quarter's training materials into a written summary report for the management team.
Candidates:
1. pptx: Use any time a .pptx presentation file is involved — creating slide decks, pitch decks, or presentations. NOTE: for written internal communications use team-communications.
2. team-communications: Write internal company communications — status reports, newsletters, project updates, and internal written content. NOTE: for slide decks use pptx instead.
Answer: [2]
Reason: The output is a written summary report, not a slide deck. team-communications handles written internal documents; pptx is only needed if a .pptx file is the output.

Example 9
Task: Create a valuation model for the target company we're planning to acquire, including DCF and comparable transactions.
Candidates:
1. ma-playbook: M&A strategy for acquiring or being acquired — due diligence, valuation, integration, deal structure. Use when a 'target company' or acquisition context is present.
2. cs-financial-analyst: Financial Analyst agent for DCF valuation, financial modeling, budgeting. NOTE: for M&A-specific valuation and due diligence, use ma-playbook instead.
Answer: [1]
Reason: The task is explicitly M&A ('target company we're planning to acquire'). ma-playbook handles acquisition valuation end-to-end. cs-financial-analyst is for standalone financial modeling without M&A context.

Example 10
Task: Set up alert rules and notification channels in New Relic so the team gets paged when error rates exceed 1%.
Candidates:
1. New Relic Automation: CONFIGURE alert policies, notification channels, and alert conditions in New Relic via Composio MCP. NOTE: this configures the monitoring tool; for incident response during an outage use incident-commander.
2. incident-commander: Comprehensive incident RESPONSE framework — severity classification, coordination during outages, post-mortems. NOTE: for configuring monitoring tools use New Relic Automation.
Answer: [1]
Reason: The task is CONFIGURING New Relic (setting up alerts). incident-commander manages response once an incident is active — it does not configure monitoring tools.

Example 11
Task: I need to ensure our SaaS product meets SOC 2 and ISO 27001 requirements before our enterprise sales push.
Candidates:
1. compliance-readiness: Multi-framework compliance officer for any industry — SOC 2, ISO 27001, GDPR, and other programs. For HealthTech/MedTech FDA submissions, use regulatory-affairs-head.
2. regulatory-affairs-head: Senior Regulatory Affairs Manager SPECIFICALLY for HealthTech and MedTech companies — FDA 510(k), PMA, CE marking. For general multi-industry compliance use compliance-readiness.
Answer: [1]
Reason: SOC 2 and ISO 27001 are general software/cloud compliance frameworks. compliance-readiness handles any industry. regulatory-affairs-head is only for HealthTech/MedTech medical device submissions.
"""

# ── RHI few-shot examples (Change 5: per-candidate REQUIRED/HELPFUL/IRRELEVANT)
_RHI_FEWSHOT_EXAMPLES_V1 = """\
Rule: classify each candidate as REQUIRED (task explicitly needs it — cannot be done without it), HELPFUL (related but not strictly necessary), or IRRELEVANT (not needed for this task).

Example 1
Task: Can you help me decide whether to start a LinkedIn newsletter and create a six-month content strategy for it?
Candidates:
1. linkedin-strategy: Use when someone needs a LinkedIn plan rather than a post — content pillars, positioning for a career change or consulting practice, newsletter strategy, or a personal brand roadmap.
2. content-strategy: When the user wants to plan a content strategy, decide what content to create, or figure out what topics to cover.
Answer: {"1": "REQUIRED", "2": "HELPFUL"}
Reason: The task is specifically about LinkedIn; content-strategy is the generic version and merely helpful.

Example 2
Task: I need a detailed plan for an A/B test to evaluate the effectiveness of two different landing page designs.
Candidates:
1. ab-test-setup: When the user wants to plan, design, or implement an A/B test or experiment. Covers hypothesis, variant design, sample size, and test setup.
2. experiment-designer: Use when planning product experiments, writing testable hypotheses, estimating sample size.
3. statistical-analyst: Run hypothesis tests, analyze A/B experiment results, calculate sample sizes.
Answer: {"1": "REQUIRED", "2": "HELPFUL", "3": "HELPFUL"}
Reason: ab-test-setup covers the full task. The others are supporting skills used alongside it.

Example 3
Task: Create a new job listing for a software engineer position and post it on our careers page (we use Ashby).
Candidates:
1. Ashby Automation: Automate recruiting and hiring workflows in Ashby ATS.
2. Lever Automation: Automate recruiting workflows in Lever ATS.
Answer: {"1": "REQUIRED", "2": "IRRELEVANT"}
Reason: The task names Ashby explicitly. Lever Automation is a different ATS entirely.

Example 4
Task: Draft a launch plan for our upcoming software release, including key milestones and promotional strategy.
Candidates:
1. marketing-strategy-pmm: Product marketing skill for positioning, GTM strategy, competitive intelligence, and product launches.
2. launch-strategy: When the user wants to plan a product launch, feature announcement, or release strategy.
3. marketing-context: Create and maintain the marketing context document that all marketing skills read before starting.
Answer: {"1": "REQUIRED", "2": "HELPFUL", "3": "IRRELEVANT"}
Reason: marketing-strategy-pmm covers launch plans end-to-end. launch-strategy is helpful but redundant. marketing-context is a setup skill, not a deliverable skill.

Example 5
Task: Build an interactive dashboard using Tailwind CSS and shadcn/ui components that visualizes real-time data.
Candidates:
1. web-artifacts-builder: Suite of tools for creating elaborate, multi-component claude.ai HTML artifacts using modern frontend web technologies (React, Tailwind, shadcn/ui).
2. senior-frontend: Frontend development skill for React, Next.js, TypeScript, and Tailwind CSS applications.
Answer: {"1": "REQUIRED", "2": "HELPFUL"}
Reason: The task asks for a self-contained artifact (a dashboard). senior-frontend is for full-app development — helpful but not required here.
"""

_RHI_FEWSHOT_EXAMPLES = _RHI_FEWSHOT_EXAMPLES_V1 + """\

Example 6
Task: Pull engagement stats from our LinkedIn and Twitter accounts and compile them into a Google Sheet.
Candidates:
1. PhantomBuster Automation: Automate lead generation, web scraping, and social media data extraction workflows. NOTE: this extracts raw data FROM social platforms; for analyzing existing data use social-media-analyzer.
2. social-media-analyzer: Social media campaign analysis and performance tracking. NOTE: this ANALYZES data you already have; for scraping/extracting data from accounts, use PhantomBuster Automation.
Answer: {"1": "REQUIRED", "2": "IRRELEVANT"}
Reason: The task requires extracting data from social accounts. social-media-analyzer cannot retrieve data from LinkedIn/Twitter; it only analyzes data already in hand.

Example 7
Task: Add a new client to our Toggl workspace and assign them to the Agency project.
Candidates:
1. Toggl Automation: Automate time tracking in Toggl Track — manage projects, clients, tags, and workspaces. NOTE: 'workspace' is Toggl Track's concept, not Google Workspace.
2. cs-workspace-admin: Google Workspace administration for Gmail, Google Drive, Sheets, and Calendar. NOTE: specifically for Google products.
Answer: {"1": "REQUIRED", "2": "IRRELEVANT"}
Reason: The task is in Toggl Track (time tracking tool). cs-workspace-admin is for Google Workspace — a completely different product.

Example 8
Task: Organize last quarter's training materials into a written summary report for the management team.
Candidates:
1. pptx: Use any time a .pptx presentation file is involved. NOTE: for written internal communications use team-communications.
2. team-communications: Write internal company communications — status reports, newsletters, project updates. NOTE: for slide decks use pptx instead.
Answer: {"1": "IRRELEVANT", "2": "REQUIRED"}
Reason: The output is a written summary report, not a slide deck. team-communications handles written internal documents.

Example 9
Task: Create a valuation model for the target company we're planning to acquire, including DCF and comparable transactions.
Candidates:
1. ma-playbook: M&A strategy for acquiring or being acquired — due diligence, valuation, integration, deal structure. Use when a 'target company' or acquisition context is present.
2. cs-financial-analyst: Financial Analyst agent for DCF valuation, financial modeling. NOTE: for M&A-specific valuation, use ma-playbook instead.
Answer: {"1": "REQUIRED", "2": "HELPFUL"}
Reason: M&A context ('target company we're planning to acquire') makes ma-playbook the required skill. cs-financial-analyst is helpful but subordinate to ma-playbook here.

Example 10
Task: Set up alert rules and notification channels in New Relic so the team gets paged when error rates exceed 1%.
Candidates:
1. New Relic Automation: CONFIGURE alert policies, notification channels, and alert conditions in New Relic. NOTE: for incident response during an outage use incident-commander.
2. incident-commander: Incident RESPONSE framework — severity classification, outage coordination, post-mortems. NOTE: for configuring monitoring tools use New Relic Automation.
Answer: {"1": "REQUIRED", "2": "IRRELEVANT"}
Reason: The task is CONFIGURING New Relic (setting up alerts). incident-commander manages active incident response; it does not configure monitoring tools.

Example 11
Task: I need to ensure our SaaS product meets SOC 2 and ISO 27001 requirements before our enterprise sales push.
Candidates:
1. compliance-readiness: Multi-framework compliance officer for any industry — SOC 2, ISO 27001, GDPR. For HealthTech/MedTech FDA submissions, use regulatory-affairs-head.
2. regulatory-affairs-head: Senior Regulatory Affairs Manager SPECIFICALLY for HealthTech and MedTech — FDA 510(k), PMA, CE marking. For general compliance use compliance-readiness.
Answer: {"1": "REQUIRED", "2": "IRRELEVANT"}
Reason: SOC 2 and ISO 27001 are general software compliance frameworks. regulatory-affairs-head is only for HealthTech/MedTech medical device submissions.
"""


def build_stage2_user_msg(
    prompt: str,
    candidates: list[dict],
    prompt_variant: str = "baseline",
) -> str:
    """
    Build the Stage-2 user message.

    prompt_variant options:
      "baseline"  — plain selection prompt
      "tight"       — stricter wording
      "fewshot"     — all worked examples + integer array output
      "fewshot-v1"  — original 5 examples only (legacy baseline)
      "rhi"         — Change 5: per-candidate REQUIRED/HELPFUL/IRRELEVANT output
    """
    if prompt_variant in ("fewshot", "fewshot-v1"):
        # fewshot / fewshot-v1: original 5 examples (reverted from v2)
        lines = [
            _FEWSHOT_EXAMPLES_V1,
            "---",
            "Now apply the same rule to the following.",
            "",
            f"Task: {prompt}",
            "",
            "Candidates:",
        ]
        for i, c in enumerate(candidates, 1):
            lines.append(f"{i}. {c['name']}: {c['description']}")
        lines += [
            "",
            "Answer (JSON array only, e.g. [1]):",
        ]

    elif prompt_variant == "rhi":
        lines = [
            _RHI_FEWSHOT_EXAMPLES,
            "---",
            "Now apply the same rule to the following.",
            "",
            f"Task: {prompt}",
            "",
            "Candidates:",
        ]
        for i, c in enumerate(candidates, 1):
            lines.append(f"{i}. {c['name']}: {c['description']}")
        lines += [
            "",
            'Answer (JSON object only, e.g. {"1": "REQUIRED", "2": "HELPFUL"}):',
        ]

    elif prompt_variant == "tight":
        lines = [f"Task: {prompt}", "", "Candidate skills:"]
        for i, c in enumerate(candidates, 1):
            lines.append(f"{i}. {c['name']}: {c['description']}")
        lines.append("")
        lines.append(
            "Return a JSON array of the numbers of skills the user CANNOT complete "
            "this task without. Exclude any skill that would merely be helpful, "
            "that overlaps with another selected skill, or that addresses only a "
            "secondary aspect of the task. If no skill is strictly necessary, "
            "return []. Return ONLY the JSON array. Example: [1, 3]"
        )

    else:  # baseline
        lines = [f"Task: {prompt}", "", "Candidate skills:"]
        for i, c in enumerate(candidates, 1):
            lines.append(f"{i}. {c['name']}: {c['description']}")
        lines.append("")
        lines.append(
            "Return a JSON array of the numbers of the skills needed to complete "
            "this task. Include a skill only if it is genuinely required. "
            "Return ONLY the JSON array. Example: [1, 3]"
        )

    return "\n".join(lines)


def call_stage2(
    prompt: str,
    candidates: list[dict],
    model: str = STAGE2_MODEL,
    prompt_variant: str = "baseline",
) -> dict:
    """
    One Stage-2 API call.
    Returns raw result dict; does NOT touch the cost tracker file.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set. Add it to .env")

    system_msg = STAGE2_SYSTEM_RHI if prompt_variant == "rhi" else STAGE2_SYSTEM
    user_msg   = build_stage2_user_msg(prompt, candidates, prompt_variant)

    t0 = time.monotonic()
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/skillrouter",
            "X-Title":       "SkillRouter-Eval",
        },
        json={
            "model":       model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0,
            "max_tokens":  256 if prompt_variant == "rhi" else 128,
        },
        timeout=60,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    resp.raise_for_status()
    body     = resp.json()
    usage    = body.get("usage", {})
    pt       = usage.get("prompt_tokens",     0)
    ct       = usage.get("completion_tokens", 0)
    cost_api = usage.get("cost")
    cost_est = estimate_cost(pt, ct, model)
    cost     = cost_api if (cost_api is not None and cost_api > 0) else cost_est
    raw      = body["choices"][0]["message"]["content"]

    chosen = _parse_chosen(raw, candidates, prompt_variant)

    return {
        "prompt_tokens":     pt,
        "completion_tokens": ct,
        "cost_from_api":     cost_api,
        "cost_used":         cost,
        "n_candidates":      len(candidates),
        "raw_output":        raw,
        "chosen":            chosen,
        "latency_ms":        latency_ms,
    }


def _parse_chosen(
    raw: str,
    candidates: list[dict],
    prompt_variant: str = "baseline",
) -> list[dict] | None:
    """
    Parse Stage-2 output.
    - fewshot / baseline / tight: JSON array of 1-based integers → select those candidates
    - rhi: JSON object {"1": "REQUIRED", ...} → keep only REQUIRED candidates
    """
    raw = raw.strip()

    if prompt_variant == "rhi":
        # Try to extract JSON object
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*?\}', raw)
            if not m:
                return None
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                return None

        if not isinstance(parsed, dict):
            return None

        chosen = []
        for key, label in parsed.items():
            if str(label).strip().upper() == "REQUIRED":
                try:
                    idx = int(key) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(candidates):
                    chosen.append(candidates[idx])
        return chosen or None

    else:
        # Integer array format
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
        return chosen or None


# ── metrics helpers ────────────────────────────────────────────────────────────

def compute_metrics(predicted_names: set[str], truth_names: set[str]) -> dict:
    tp = len(predicted_names & truth_names)
    fp = len(predicted_names - truth_names)
    fn = len(truth_names - predicted_names)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    exact     = predicted_names == truth_names

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "exact":     exact,
        "tp": tp, "fp": fp, "fn": fn,
    }


def classify_failure_taxonomy(
    truth_names: set[str],
    predicted_names: set[str],
    stage1_candidate_names: set[str],
    stage2_ran: bool,
    stage2_abstained: bool,
) -> list[str]:
    """
    Change 7: Assign one or more failure taxonomy labels to a non-exact result.

    MISS_STAGE1    : at least one GT skill never reached Stage 2 candidates
    WRONG_STAGE2   : GT skill was in Stage 1 candidates but Stage 2 dropped it
    OVERROUTING    : false positives (extra skills returned)
    ABSTAIN        : Stage 2 was called but returned nothing (fell back to Stage 1)
    """
    if predicted_names == truth_names:
        return []

    labels = []
    missed_gt = truth_names - predicted_names

    if missed_gt:
        not_in_s1 = missed_gt - stage1_candidate_names
        in_s1_dropped = missed_gt & stage1_candidate_names
        if not_in_s1:
            labels.append("MISS_STAGE1")
        if in_s1_dropped and stage2_ran:
            labels.append("WRONG_STAGE2")

    if predicted_names - truth_names:
        labels.append("OVERROUTING")

    if stage2_abstained:
        labels.append("ABSTAIN")

    return labels or ["OTHER"]


# ── stage2-dry-run command ─────────────────────────────────────────────────────

def cmd_stage2_dry_run(args):
    eval_path = resolve_eval_path(args)
    if not eval_path.exists():
        sys.exit(f"Eval set not found: {eval_path}")

    with open(eval_path) as f:
        eval_set = json.load(f)

    total_eval     = len(eval_set)
    n_sample       = args.n
    threshold      = args.threshold
    stage2_model   = args.stage2_model
    prompt_variant = args.prompt_variant

    from collections import defaultdict
    import random
    by_size = defaultdict(list)
    for rec in eval_set:
        by_size[rec["set_size"]].append(rec)

    rng    = random.Random(42)
    sample = []
    for size in sorted(by_size):
        pool = by_size[size]
        frac = len(pool) / total_eval
        take = max(1, round(n_sample * frac))
        take = min(take, len(pool))
        sample.extend(rng.sample(pool, take))
    sample = sample[:n_sample]

    print(f"\n=== STAGE 2 DRY RUN ===")
    print(f"Eval prompts sampled : {len(sample)} / {total_eval}")
    print(f"Stage-1 threshold    : {threshold}")
    print(f"Stage-2 model        : {stage2_model}")
    print(f"Prompt variant       : {prompt_variant}")

    tracker = load_cost()
    check_budget(tracker)

    sys.path.insert(0, str(ROOT / "scripts"))
    from skillrouter import stage1, is_unambiguous
    print("\nLoading embedding model and index …")

    cap = args.cap
    call_results = []
    skipped      = 0

    for i, rec in enumerate(sample, 1):
        prompt = rec["prompt"]
        truth  = set(rec["ground_truth_names"])

        print(f"\n  [{i}/{len(sample)}] set_size={rec['set_size']}  "
              f"truth={sorted(truth)}")
        print(f"  Prompt: {prompt[:90]!r}")

        candidates, top_score, second_score = stage1(prompt, threshold, cap=cap)

        print(f"  Stage 1: {len(candidates)} candidate(s) above {threshold}  "
              f"(top={top_score:.4f}, 2nd={second_score:.4f})")

        if not candidates:
            print("  Stage 2: SKIP (no candidates)")
            skipped += 1
            continue

        if is_unambiguous(candidates, top_score, second_score):
            print(f"  Stage 2: SKIP (unambiguous — gap={top_score - second_score:.4f})")
            skipped += 1
            continue

        result = call_stage2(prompt, candidates, model=stage2_model,
                             prompt_variant=prompt_variant)
        pt   = result["prompt_tokens"]
        ct   = result["completion_tokens"]
        cost = result["cost_used"]
        src  = "API" if result["cost_from_api"] is not None else "estimate"

        print(f"  Stage 2: prompt_tokens={pt}  completion_tokens={ct}  "
              f"cost=${cost:.6f} ({src})  candidates_sent={result['n_candidates']}  "
              f"latency={result['latency_ms']:.0f}ms")
        if result["chosen"]:
            chosen_names = {c["name"] for c in result["chosen"]}
            print(f"  Chosen: {sorted(chosen_names)}")
        else:
            print(f"  Parse failed — raw: {result['raw_output'][:80]!r}")

        call_results.append(result)

    n_calls    = len(call_results)
    total_pt   = sum(r["prompt_tokens"]     for r in call_results)
    total_ct   = sum(r["completion_tokens"] for r in call_results)
    total_cost = sum(r["cost_used"]         for r in call_results)
    avg_cost   = total_cost / n_calls if n_calls > 0 else 0.0

    print(f"\n{'=' * 60}")
    print("STAGE 2 DRY RUN — COST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Prompts sampled      : {len(sample)}")
    print(f"  Stage 2 calls made   : {n_calls}  (skipped: {skipped})")
    print(f"  Total prompt tokens  : {total_pt:,}")
    print(f"  Total completion tok : {total_ct:,}")
    print(f"  Total cost (actual)  : ${total_cost:.6f}")
    print(f"  Avg cost / call      : ${avg_cost:.6f}")

    if n_calls > 0:
        skip_rate         = skipped / len(sample)
        full_stage2_calls = round(total_eval * (1 - skip_rate))
        projected         = avg_cost * full_stage2_calls
        current_spend     = tracker["total_usd"]
        headroom          = HALT_AT_USD - current_spend

        print(f"\n{'=' * 60}")
        print("EXTRAPOLATION TO FULL EVAL SET")
        print(f"{'=' * 60}")
        print(f"  Skip rate                  : {skip_rate * 100:.0f}%")
        print(f"  Projected Stage-2 calls    : {full_stage2_calls} / {total_eval}")
        print(f"  Avg cost/call              : ${avg_cost:.6f}")
        print(f"  Projected Stage-2 cost     : ${projected:.6f}")
        print(f"  Current running spend      : ${current_spend:.6f}")
        print(f"  Projected total            : ${projected + current_spend:.6f}")
        print(f"  Budget remaining           : ${headroom:.6f}")

        if projected + current_spend >= HALT_AT_USD:
            print("  *** WARNING: projected total would exceed halt limit ***")
        elif projected + current_spend >= 2.00:
            print(f"  *** CAUTION: projected ${projected + current_spend:.4f} "
                  f"is within $1.00 of the ${HALT_AT_USD:.2f} limit ***")
        else:
            print(f"  VERDICT: projected ${projected + current_spend:.4f} total is "
                  f"well under ${HALT_AT_USD:.2f} limit.")

    print("\nDry run complete. No cost tracker was updated.")


# ── calibrate command ──────────────────────────────────────────────────────────

def cmd_calibrate(args):
    """
    Sweep Stage-1 thresholds.  Supports hybrid and conservative-agg flags
    so you can measure recall ceiling under different configurations.
    """
    eval_path = resolve_eval_path(args)
    if not eval_path.exists():
        sys.exit(f"Eval set not found: {eval_path}")

    with open(eval_path) as f:
        eval_set = json.load(f)

    embed_model      = getattr(args, "embed_model",      "all-MiniLM-L6-v2")
    index_suffix     = getattr(args, "index_suffix",     "")
    hybrid           = getattr(args, "hybrid",           False)
    semantic_weight  = getattr(args, "semantic_weight",  0.7)
    lexical_weight   = getattr(args, "lexical_weight",   0.3)
    conservative_agg = getattr(args, "conservative_agg", False)
    w_desc           = getattr(args, "w_desc",           0.50)
    w_max_phrase     = getattr(args, "w_max_phrase",     0.30)
    w_top2           = getattr(args, "w_top2",           0.20)

    hybrid_tag = " [hybrid]" if hybrid else ""
    agg_tag    = " [consv_agg]" if conservative_agg else ""
    print(f"\nLoading '{embed_model}' index (suffix='{index_suffix}'){hybrid_tag}{agg_tag} …")

    sys.path.insert(0, str(ROOT / "scripts"))
    from skillrouter import _load_index, _get_model, _load_lexical_index, _tokenize
    model         = _get_model(embed_model)
    vectors, meta = _load_index(index_suffix)

    # ── detect multi-vector mode ──────────────────────────────────────────────
    multivec = meta and "skill_id" in meta[0]
    if multivec:
        from collections import defaultdict as _dd
        skill_rows   = _dd(list)
        skill_id_map = {}
        for i, m in enumerate(meta):
            sid = m["skill_id"]
            skill_rows[sid].append(i)
            if sid not in skill_id_map:
                skill_id_map[sid] = m

        # separate description vs phrase row indices per skill
        skill_desc_rows  = {}
        skill_phrase_rows = _dd(list)
        for i, m in enumerate(meta):
            sid = m["skill_id"]
            vt  = m.get("vector_type", "description")
            if vt == "description":
                skill_desc_rows[sid] = i
            elif vt.startswith("phrase"):
                skill_phrase_rows[sid].append(i)

        unique_skills = list(skill_id_map.keys())
        sid_to_col    = {sid: ki for ki, sid in enumerate(unique_skills)}
        meta_id_to_idx = sid_to_col
        print(f"  Multi-vector: {len(vectors)} vectors → {len(unique_skills)} unique skills")
    else:
        meta_id_to_idx = {m["id"]: i for i, m in enumerate(meta)}

    # Load lexical index if hybrid
    lex_data = None
    if hybrid:
        lex_data = _load_lexical_index()
        if lex_data is None:
            print("  [WARN] Lexical index not found — running semantic-only")
        else:
            print(f"  Lexical index loaded ({len(lex_data['skill_ids'])} skills)")

    missing_ids = []
    gt_indices  = []
    for rec in eval_set:
        idxs = []
        for gt_id in rec["ground_truth_ids"]:
            if gt_id in meta_id_to_idx:
                idxs.append(meta_id_to_idx[gt_id])
            else:
                missing_ids.append(gt_id)
        gt_indices.append(idxs)

    if missing_ids:
        print(f"  [WARN] {len(missing_ids)} GT skill IDs not found in index.")

    # ── batch-embed all prompts ───────────────────────────────────────────────
    prompts = [rec["prompt"] for rec in eval_set]
    print(f"Embedding {len(prompts)} eval prompts …")
    query_vecs = model.encode(
        prompts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print("Computing similarity matrix …")
    raw_scores = query_vecs @ vectors.T   # (n_prompts, n_vectors)

    if multivec:
        n_unique   = len(unique_skills)
        all_scores = np.full((len(eval_set), n_unique), -1.0, dtype=np.float32)

        if conservative_agg:
            # Weighted aggregation per skill
            for ki, sid in enumerate(unique_skills):
                d_row = skill_desc_rows.get(sid)
                p_rows = skill_phrase_rows.get(sid, [])
                desc_col = raw_scores[:, d_row] if d_row is not None else np.zeros(len(eval_set))
                if p_rows:
                    phrase_mat = raw_scores[:, p_rows]   # (n_prompts, n_phrases)
                    phrase_sorted = np.sort(phrase_mat, axis=1)[:, ::-1]
                    max_phrase = phrase_sorted[:, 0]
                    top2 = phrase_sorted[:, :2].mean(axis=1)
                else:
                    max_phrase = np.zeros(len(eval_set))
                    top2       = np.zeros(len(eval_set))
                all_scores[:, ki] = (w_desc * desc_col
                                     + w_max_phrase * max_phrase
                                     + w_top2 * top2)
        else:
            # Original MAX aggregation
            for ki, sid in enumerate(unique_skills):
                all_scores[:, ki] = raw_scores[:, skill_rows[sid]].max(axis=1)
    else:
        all_scores = raw_scores

    # Hybrid: fuse with BM25 scores
    if hybrid and lex_data is not None:
        lex_ids = lex_data["skill_ids"]
        print("Computing BM25 scores …")
        bm25 = lex_data["bm25"]
        # Build (n_prompts, n_unique) BM25 matrix
        lex_id_to_col = {}
        if multivec:
            lex_id_to_col = {sid: sid_to_col[sid] for sid in lex_ids if sid in sid_to_col}
        else:
            lex_id_to_col = {m["id"]: i for i, m in enumerate(meta) if m["id"] in lex_ids}

        bm25_scores = np.zeros_like(all_scores)
        for pi, rec in enumerate(eval_set):
            tokens    = _tokenize(rec["prompt"])
            raw_bm25  = np.array(bm25.get_scores(tokens), dtype=np.float32)
            max_bm25  = raw_bm25.max()
            norm_bm25 = (raw_bm25 / max_bm25) if max_bm25 > 0 else raw_bm25
            for i, sid in enumerate(lex_ids):
                col = lex_id_to_col.get(sid)
                if col is not None:
                    bm25_scores[pi, col] = float(norm_bm25[i])

        all_scores = semantic_weight * all_scores + lexical_weight * bm25_scores

    # ── GT score range ────────────────────────────────────────────────────────
    gt_min_scores = []
    for pi, idxs in enumerate(gt_indices):
        if idxs:
            gt_min_scores.append(float(min(all_scores[pi, idx] for idx in idxs)))
        else:
            gt_min_scores.append(1.0)

    print(f"\nGT skill score range across {len(eval_set)} prompts:")
    print(f"  Min  : {min(gt_min_scores):.4f}")
    print(f"  P10  : {float(np.percentile(gt_min_scores, 10)):.4f}")
    print(f"  P50  : {float(np.percentile(gt_min_scores, 50)):.4f}")
    print(f"  P90  : {float(np.percentile(gt_min_scores, 90)):.4f}")
    print(f"  Max  : {max(gt_min_scores):.4f}")
    print(f"  → 100% S1 recall requires threshold ≤ {min(gt_min_scores):.4f}")

    TOKENS_PER_CAND = 60
    BASE_TOKENS     = 150
    N_EVAL          = len(eval_set)
    BALLOON_LIMIT   = 30

    cap  = args.cap
    rows = []
    thresholds = np.round(np.arange(0.25, 0.505, 0.025), 3)

    for thresh in thresholds:
        thresh = float(thresh)
        cand_counts = []
        recalls     = []
        n_perfect   = 0

        for pi, rec in enumerate(eval_set):
            sc    = all_scores[pi]
            above = sc >= thresh
            n_c   = int(above.sum())
            cand_counts.append(n_c)

            if cap and n_c > cap:
                above_idx   = np.where(above)[0]
                above_idx   = above_idx[np.argsort(sc[above_idx])[::-1]]
                capped_set  = set(above_idx[:cap])
            else:
                capped_set = set(np.where(above)[0])

            gt_idxs = gt_indices[pi]
            if gt_idxs:
                found  = sum(1 for idx in gt_idxs if idx in capped_set)
                recall = found / len(gt_idxs)
                recalls.append(recall)
                if recall == 1.0:
                    n_perfect += 1
            else:
                recalls.append(1.0)
                n_perfect += 1

        cands_arr  = np.array(cand_counts, dtype=float)
        avg_capped = float(np.mean(cands_arr))
        s2_in_price, _ = MODEL_PRICING.get(STAGE2_MODEL, _DEFAULT_PRICING)
        est_s2_cost = N_EVAL * (TOKENS_PER_CAND * avg_capped + BASE_TOKENS) * s2_in_price

        rows.append({
            "threshold":      thresh,
            "perfect_s1r":    n_perfect / N_EVAL,
            "avg_recall":     float(np.mean(recalls)),
            "avg_cands":      avg_capped,
            "p95_capped":     float(np.percentile(cands_arr, 95)),
            "max_uncapped":   int(cands_arr.max()),
            "n_over_balloon": int((cands_arr > BALLOON_LIMIT).sum()),
            "n_zero_cands":   int((cands_arr == 0).sum()),
            "est_s2_cost":    est_s2_cost,
        })

    print(f"\n{'─'*100}")
    print(
        f"{'Thresh':>7}  {'PerfS1R':>8}  {'AvgRec':>7}  "
        f"{'AvgCand':>7}  {'P95':>5}  {'MaxRaw':>6}  "
        f"{'Over30':>6}  {'NoCand':>6}  {'EstS2$':>7}"
    )
    print(f"{'─'*100}")
    for r in rows:
        flag = " ◄" if r["perfect_s1r"] >= 0.95 and r["avg_cands"] <= 20 else ""
        print(
            f"  {r['threshold']:.3f}   "
            f"{r['perfect_s1r']*100:>7.1f}%  "
            f"{r['avg_recall']*100:>6.1f}%  "
            f"{r['avg_cands']:>7.1f}  "
            f"{r['p95_capped']:>5.0f}  "
            f"{r['max_uncapped']:>6}  "
            f"{r['n_over_balloon']:>6}  "
            f"{r['n_zero_cands']:>6}  "
            f"${r['est_s2_cost']:.4f}"
            f"{flag}"
        )
    print(f"{'─'*100}")
    print(f"  Cap={cap}. PerfS1R = % prompts where ALL GT skills survive after cap + threshold")


# ── run command ────────────────────────────────────────────────────────────────

def cmd_run(args):
    eval_path = resolve_eval_path(args)
    if not eval_path.exists():
        sys.exit(f"Eval set not found: {eval_path}")

    with open(eval_path) as f:
        eval_set = json.load(f)

    threshold        = args.threshold
    no_stage2        = args.no_stage2
    cap              = args.cap
    stage2_model     = args.stage2_model
    prompt_variant   = args.prompt_variant
    alpha            = getattr(args, "alpha",            None)
    alpha_floor      = getattr(args, "alpha_floor",      0.20)
    embed_model      = getattr(args, "embed_model",      "all-MiniLM-L6-v2")
    index_suffix     = getattr(args, "index_suffix",     "")
    hybrid           = getattr(args, "hybrid",           False)
    semantic_weight  = getattr(args, "semantic_weight",  0.7)
    lexical_weight   = getattr(args, "lexical_weight",   0.3)
    conservative_agg = getattr(args, "conservative_agg", False)
    w_desc           = getattr(args, "w_desc",           0.50)
    w_max_phrase     = getattr(args, "w_max_phrase",     0.30)
    w_top2           = getattr(args, "w_top2",           0.20)
    out_path         = getattr(args, "out",              None)

    sys.path.insert(0, str(ROOT / "scripts"))
    from skillrouter import stage1, is_unambiguous

    tracker = load_cost()
    check_budget(tracker)

    results        = []
    exact_matches  = 0
    skipped_stage2 = 0
    latencies_ms   = []
    total_cost_run = 0.0

    alpha_label = f"alpha={alpha}" if alpha is not None else f"threshold={threshold}"
    split_name  = getattr(args, "split", None) or Path(args.eval_path).stem
    print(f"\n{'=' * 60}")
    print(f"FULL EVAL  ({len(eval_set)} prompts, {alpha_label}, cap={cap})")
    print(f"Stage-2 model   : {stage2_model}")
    print(f"Prompt variant  : {prompt_variant}")
    print(f"Embed model     : {embed_model}  suffix='{index_suffix}'")
    print(f"Hybrid          : {hybrid}  (sem={semantic_weight}, lex={lexical_weight})")
    print(f"Conservative agg: {conservative_agg}")
    print(f"Split / path    : {split_name}")
    print(f"{'=' * 60}\n")

    for i, rec in enumerate(eval_set, 1):
        prompt       = rec["prompt"]
        truth_names  = set(rec["ground_truth_names"])
        set_size     = rec["set_size"]

        t0 = time.monotonic()
        candidates, top_score, second_score = stage1(
            prompt, threshold, cap=cap,
            alpha=alpha, alpha_floor=alpha_floor,
            embed_model=embed_model, index_suffix=index_suffix,
            hybrid=hybrid,
            semantic_weight=semantic_weight, lexical_weight=lexical_weight,
            lexical_path=getattr(args, "lexical_index", None),
            conservative_agg=conservative_agg,
            w_desc=w_desc, w_max_phrase=w_max_phrase, w_top2=w_top2,
        )
        s1_latency_ms = (time.monotonic() - t0) * 1000

        stage1_candidate_names = {c["name"] for c in candidates}
        stage2_ran         = False
        stage2_abstained   = False

        if not candidates:
            predicted_names = set()
        elif no_stage2 or is_unambiguous(candidates, top_score, second_score):
            predicted_names = {c["name"] for c in candidates}
            skipped_stage2 += 1
        else:
            stage2_ran = True
            result = call_stage2(prompt, candidates, model=stage2_model,
                                 prompt_variant=prompt_variant)
            total_cost_run += result["cost_used"]
            tracker["total_usd"] = round(tracker["total_usd"] + result["cost_used"], 6)
            tracker["calls"] += 1
            save_cost(tracker)
            check_budget(tracker)
            latencies_ms.append(result["latency_ms"])

            if result["chosen"]:
                predicted_names = {c["name"] for c in result["chosen"]}
            else:
                predicted_names   = {c["name"] for c in candidates}
                stage2_abstained  = True

        total_latency_ms = (time.monotonic() - t0) * 1000

        metrics  = compute_metrics(predicted_names, truth_names)
        exact_matches += int(metrics["exact"])

        taxonomy = classify_failure_taxonomy(
            truth_names, predicted_names,
            stage1_candidate_names, stage2_ran, stage2_abstained,
        )

        row = {
            "prompt_id":              rec["prompt_id"],
            "set_size":               set_size,
            "trigger_type":           rec.get("trigger_type", "explicit"),
            "truth":                  sorted(truth_names),
            "predicted":              sorted(predicted_names),
            "stage1_candidates":      sorted(stage1_candidate_names),
            "stage2_ran":             stage2_ran,
            "stage2_abstained":       stage2_abstained,
            "taxonomy":               taxonomy,
            "total_latency_ms":       round(total_latency_ms, 1),
            **metrics,
        }
        results.append(row)

        status = "✓" if metrics["exact"] else "✗"
        print(f"  [{i:02d}/{len(eval_set)}] {status}  P={metrics['precision']:.2f}  "
              f"R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  "
              f"size={set_size}  s2={'Y' if stage2_ran else 'N'}")
        if not metrics["exact"]:
            print(f"       truth={sorted(truth_names)}")
            print(f"       pred ={sorted(predicted_names)}")
            if taxonomy:
                print(f"       tags ={taxonomy}")

    # ── aggregate metrics ──────────────────────────────────────────────────────
    n  = len(results)
    em = exact_matches / n

    def avg(key):
        return sum(r[key] for r in results) / n

    print(f"\n{'=' * 60}")
    print("EVAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Split / path         : {split_name}")
    print(f"  Prompts evaluated    : {n}")
    print(f"  Exact-set match rate : {em:.3f}  ({exact_matches}/{n})")
    print(f"  Avg precision        : {avg('precision'):.3f}")
    print(f"  Avg recall           : {avg('recall'):.3f}")
    print(f"  Avg F1               : {avg('f1'):.3f}")
    print(f"  Stage 2 skipped      : {skipped_stage2}")
    print(f"  Running cost         : ${tracker['total_usd']:.5f}")
    print(f"  This run cost        : ${total_cost_run:.5f}")

    # Latency
    if latencies_ms:
        lat = sorted(latencies_ms)
        p50 = lat[int(len(lat) * 0.50)]
        p95 = lat[int(len(lat) * 0.95)]
        print(f"  Stage-2 latency p50  : {p50:.0f}ms   p95: {p95:.0f}ms")
    cost_per_1k = (total_cost_run / n) * 1000
    print(f"  Cost per 1k queries  : ${cost_per_1k:.4f}")

    # By trigger type
    from collections import defaultdict
    by_ttype = defaultdict(list)
    for r in results:
        by_ttype[r.get("trigger_type", "explicit")].append(r)

    print(f"\n  By trigger type:")
    for ttype in sorted(by_ttype):
        rows_t = by_ttype[ttype]
        em_t   = sum(r["exact"] for r in rows_t) / len(rows_t)
        f1_t   = sum(r["f1"]    for r in rows_t) / len(rows_t)
        print(f"    {ttype:<12}  n={len(rows_t):>3}  exact={em_t:.3f}  avg_f1={f1_t:.3f}")

    # Failure modes
    fp_only = sum(1 for r in results if r["fp"] > 0 and r["fn"] == 0)
    fn_only = sum(1 for r in results if r["fn"] > 0 and r["fp"] == 0)
    both    = sum(1 for r in results if r["fp"] > 0 and r["fn"] > 0)
    print(f"\n  Failure modes (non-exact):")
    print(f"    Extra only  (FP)  : {fp_only}")
    print(f"    Missing only (FN) : {fn_only}")
    print(f"    Both FP+FN        : {both}")

    # Change 7: Failure taxonomy
    from collections import Counter
    tax_counts = Counter()
    for r in results:
        for cat in r.get("taxonomy", []):
            tax_counts[cat] += 1

    print(f"\n  Failure taxonomy (Change 7):")
    for cat in ["MISS_STAGE1", "WRONG_STAGE2", "OVERROUTING", "ABSTAIN", "OTHER"]:
        if tax_counts[cat] > 0:
            print(f"    {cat:<15}: {tax_counts[cat]}")

    # Save results
    default_out = ROOT / "data" / "eval_results.json"
    save_to     = Path(out_path) if out_path else default_out
    with open(save_to, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results → {save_to}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Offline evaluator for the skill router",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    _eval_path_default = str(EVAL_PATH_DEFAULT)

    def _add_shared_stage1_args(p):
        """Add Stage-1 config args shared by calibrate, dry-run, and run."""
        p.add_argument("--eval-path", default=_eval_path_default, metavar="PATH",
                       help="Eval JSON file (overridden by --split)")
        p.add_argument("--split", default=None,
                       choices=["development", "legacy_holdout", "final_test"],
                       help="Dataset split to use (overrides --eval-path)")
        p.add_argument("--final-test-approved", action="store_true",
                       help="Required guard to access final_test split")
        p.add_argument("--embed-model", default="all-MiniLM-L6-v2", metavar="MODEL")
        p.add_argument("--index-suffix", default="", metavar="SUFFIX",
                       help="Index suffix (e.g. '_multi')")
        p.add_argument("--hybrid", action="store_true",
                       help="Change 1: hybrid BM25+semantic retrieval")
        p.add_argument("--semantic-weight", type=float, default=0.7, metavar="W")
        p.add_argument("--lexical-weight",  type=float, default=0.3, metavar="W")
        p.add_argument("--lexical-index", default=None, metavar="PATH",
                       help="Override BM25 index path (default: data/lexical_index.pkl)")
        p.add_argument("--conservative-agg", action="store_true",
                       help="Change 3: weighted desc+phrase aggregation")
        p.add_argument("--w-desc",       type=float, default=0.50, metavar="W")
        p.add_argument("--w-max-phrase", type=float, default=0.30, metavar="W")
        p.add_argument("--w-top2",       type=float, default=0.20, metavar="W")

    # calibrate
    calp = sub.add_parser("calibrate",
                           help="Sweep Stage-1 thresholds (free, local)")
    calp.add_argument("--cap", type=int, default=40)
    _add_shared_stage1_args(calp)

    # stage2-dry-run
    dr = sub.add_parser("stage2-dry-run",
                         help="Run Stage 2 on N prompts; report real costs")
    dr.add_argument("--n", type=int, default=8)
    dr.add_argument("--threshold", type=float, default=0.35)
    dr.add_argument("--cap", type=int, default=40)
    dr.add_argument("--stage2-model", default=STAGE2_MODEL, metavar="MODEL")
    dr.add_argument("--prompt-variant", default="baseline",
                    choices=["baseline", "tight", "fewshot", "fewshot-v1", "rhi"])
    _add_shared_stage1_args(dr)

    # run
    rp = sub.add_parser("run", help="Full eval against eval prompts")
    rp.add_argument("--threshold", type=float, default=0.30)
    rp.add_argument("--cap", type=int, default=40)
    rp.add_argument("--no-stage2", action="store_true")
    rp.add_argument("--stage2-model", default=STAGE2_MODEL, metavar="MODEL")
    rp.add_argument("--prompt-variant", default="baseline",
                    choices=["baseline", "tight", "fewshot", "fewshot-v1", "rhi"])
    rp.add_argument("--alpha", type=float, default=None, metavar="ALPHA")
    rp.add_argument("--alpha-floor", type=float, default=0.20, metavar="FLOOR")
    rp.add_argument("--out", default=None, metavar="PATH",
                    help="Output path for detailed results JSON "
                         "(default: data/eval_results.json)")
    _add_shared_stage1_args(rp)

    args = parser.parse_args()
    if args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "stage2-dry-run":
        cmd_stage2_dry_run(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
