"""
run_eval.py

Main evaluation runner for the Email Generation Assistant.

USAGE:

  # Run from the project root (email-gen-assistant/)

  # Run Model A (OpenAI gpt-4o-mini)
  python -m evaluation.run_eval --model openai --label model_a

  # Run Model B (Ollama Qwen3-8B)
  # Make sure `ollama serve` is running and `ollama pull qwen3:8b` is done
  python -m evaluation.run_eval --model ollama --label model_b

  # Run both back-to-back and produce the comparison report in one shot
  python -m evaluation.run_eval --compare

  # Fast smoke test — no LLM judge API calls for fact recall
  python -m evaluation.run_eval --model openai --keyword-fact-recall
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from statistics import mean

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI

from backend.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    JUDGE_MODEL,
    JUDGE_BASE_URL,
)
from backend.graph.graph import email_graph
from evaluation.metrics.fact_recall import (
    score_fact_recall_embedding,
    score_fact_recall_keyword,
    FactRecallResult,
)
from evaluation.metrics.llm_judge import score_tone_adherence, ToneAdherenceResult
from evaluation.metrics.structural import score_structural

SCENARIOS_PATH = PROJECT_ROOT / "evaluation" / "scenarios.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


# Helpers

def load_scenarios() -> list[dict]:
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _judge_client() -> AsyncOpenAI:
    """
    Returns an OpenAI-backed client used for:
      - Tone Adherence judge (Metric 2) — chat completions
      - Fact Recall embeddings (Metric 1) — embeddings endpoint
    Always OpenAI regardless of which generator model is being evaluated.
    """
    return AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=JUDGE_BASE_URL)


# Single-scenario runner

async def run_single_scenario(
    scenario: dict,
    model_variant: str,
    judge_client: AsyncOpenAI,
    use_embedding_fact_recall: bool = True,
) -> dict:
    """
    Runs one scenario through the production LangGraph pipeline for the given
    model variant, then scores the result with all three custom metrics.

    Returns a flat dict suitable for both CSV rows and JSON serialization.
    Private keys prefixed with "_" carry per-fact detail for the JSON
    side-channel (too verbose for CSV).
    """
    initial_state = {
        "intent":        scenario["intent"],
        "key_facts":     scenario["key_facts"],
        "tone":          scenario["tone"],
        "model":         model_variant,
        "system_prompt": "",
        "user_prompt":   "",
        "cot_reasoning": "",
        "email_output":  "",
        "model_used":    "",
        "error":         None,
    }

    final_state = await email_graph.ainvoke(initial_state)

    row = {
        "scenario_id":          scenario["id"],
        "intent":               scenario["intent"],
        "tone":                 scenario["tone"],
        "model_variant":        model_variant,
        "model_used":           final_state.get("model_used", ""),
        "error":                final_state.get("error"),
        "email_output":         final_state.get("email_output", ""),
        "cot_captured":         bool(final_state.get("cot_reasoning")),
        "fact_recall_score":    None,
        "tone_adherence_score": None,
        "tone_adherence_raw":   None,
        "structural_score":     None,
        "composite_score":      None,
        "metric_notes":         "",
    }

    # Early exit on generation failure 
    if final_state.get("error"):
        row["metric_notes"] = f"Generation failed: {final_state['error']}"
        row.update({
            "fact_recall_score":    0.0,
            "tone_adherence_score": 0.0,
            "tone_adherence_raw":   1,
            "structural_score":     0.0,
            "composite_score":      0.0,
        })
        return row

    email_output = final_state["email_output"]

    # Metric 1: Fact Recall (embedding cosine similarity)
    if use_embedding_fact_recall:
        fact_result: FactRecallResult = await score_fact_recall_embedding(
            email_output=email_output,
            key_facts=scenario["key_facts"],
            embed_client=judge_client,   # same OpenAI client, embeddings endpoint
        )
    else:
        fact_result = score_fact_recall_keyword(email_output, scenario["key_facts"])

    # Metric 2: Tone Adherence
    tone_result: ToneAdherenceResult = await score_tone_adherence(
        email_output=email_output,
        target_tone=scenario["tone"],
        judge_client=judge_client,
        judge_model=JUDGE_MODEL,
    )

    # Metric 3: Structural Completeness
    structural_result = score_structural(email_output)

    composite = mean([
        fact_result.score,
        tone_result.normalized_score,
        structural_result["score"],
    ])

    row.update({
        "fact_recall_score":    fact_result.score,
        "tone_adherence_score": tone_result.normalized_score,
        "tone_adherence_raw":   tone_result.raw_score,
        "structural_score":     structural_result["score"],
        "composite_score":      round(composite, 3),
        "metric_notes":         "; ".join(structural_result["notes"]) if structural_result["notes"] else "",
    })

    # Per-fact detail and tone justification — stashed for JSON, skipped in CSV
    row["_fact_detail"]        = [
        {**v.model_dump(), "similarity_score": v.similarity_score}
        for v in fact_result.per_fact
    ]
    row["_tone_justification"] = tone_result.justification
    row["_structural_detail"]  = {
        "has_subject_line":    structural_result["has_subject_line"],
        "has_greeting":        structural_result["has_greeting"],
        "has_body":            structural_result["has_body"],
        "has_sign_off":        structural_result["has_sign_off"],
        "no_leaked_artifacts": structural_result["no_leaked_artifacts"],
    }

    return row


# Full-model runner

async def run_model(
    model_variant: str,
    label: str,
    use_embedding_fact_recall: bool = True,
) -> list[dict]:
    """Runs all 10 scenarios for one model variant; writes CSV + detail JSON."""
    scenarios = load_scenarios()
    judge = _judge_client()

    print(f"\n{'═' * 60}")
    print(f"  [{label}]  model={model_variant}  |  {len(scenarios)} scenarios")
    print(f"{'═' * 60}")

    rows = []
    for scenario in scenarios:
        print(f"  ▶ Scenario {scenario['id']:>2}:  {scenario['tone']:<12}  "
              f"{scenario['intent'][:52]}...")
        row = await run_single_scenario(scenario, model_variant, judge, use_embedding_fact_recall)
        status = "✓" if not row.get("error") else "✗"
        print(f"    {status}  fact={row['fact_recall_score']:.2f}  "
              f"tone={row['tone_adherence_score']:.2f}  "
              f"struct={row['structural_score']:.2f}  "
              f"composite={row['composite_score']:.3f}")
        rows.append(row)

    _write_csv(rows, label)
    _write_json_detail(rows, label)
    _print_summary(rows, label)
    return rows


# Output writers

_CSV_FIELDS = [
    "scenario_id", "intent", "tone", "model_variant", "model_used",
    "cot_captured",
    "fact_recall_score", "tone_adherence_score", "tone_adherence_raw",
    "structural_score", "composite_score",
    "error", "metric_notes", "email_output",
]


def _write_csv(rows: list[dict], label: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{label}_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n  Wrote → {out_path}")


def _write_json_detail(rows: list[dict], label: str) -> None:
    """Full detail with per-fact verdicts and tone justifications."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{label}_detail.json"

    # Strip the "_" prefix for cleaner JSON keys
    clean_rows = []
    for row in rows:
        clean_row = {k.lstrip("_"): v for k, v in row.items()}
        clean_rows.append(clean_row)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_rows, f, indent=2, ensure_ascii=False)
    print(f"  Wrote → {out_path}")


def _print_summary(rows: list[dict], label: str) -> None:
    valid = [r for r in rows if not r.get("error")]
    n_failed = len(rows) - len(valid)

    print(f"\n  ┌─ [{label}] Summary ({len(valid)}/{len(rows)} succeeded) {'─' * 25}")

    if not valid:
        print("  │  All scenarios failed — check your API keys and model availability.")
        print("  └" + "─" * 50)
        return

    avg_fact       = mean(r["fact_recall_score"]    for r in valid)
    avg_tone       = mean(r["tone_adherence_score"]  for r in valid)
    avg_structural = mean(r["structural_score"]      for r in valid)
    avg_composite  = mean(r["composite_score"]       for r in valid)

    print(f"  │  Fact Recall (avg):              {avg_fact:.3f}")
    print(f"  │  Tone Adherence (avg, 0-1):      {avg_tone:.3f}")
    print(f"  │  Structural Completeness (avg):  {avg_structural:.3f}")
    print(f"  │  Composite Score (avg):          {avg_composite:.3f}")
    if n_failed:
        print(f"  │  ⚠  {n_failed} scenario(s) failed generation.")
    print(f"  └{'─' * 50}")


# Comparison report builder

def _aggregate(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("error")]
    if not valid:
        return {
            "fact_recall_avg": 0.0, "tone_adherence_avg": 0.0,
            "structural_avg": 0.0, "composite_avg": 0.0,
            "success_rate": 0.0, "n_scenarios": len(rows),
        }
    return {
        "fact_recall_avg":    round(mean(r["fact_recall_score"]    for r in valid), 3),
        "tone_adherence_avg": round(mean(r["tone_adherence_score"] for r in valid), 3),
        "structural_avg":     round(mean(r["structural_score"]     for r in valid), 3),
        "composite_avg":      round(mean(r["composite_score"]      for r in valid), 3),
        "success_rate":       round(len(valid) / len(rows), 3),
        "n_scenarios":        len(rows),
    }


def build_comparison_report(
    model_a_rows: list[dict],
    model_b_rows: list[dict],
    label_a: str,
    label_b: str,
) -> dict:
    report = {
        "metric_definitions": {
            "fact_recall": (
                "Fraction of user-supplied key_facts whose substance appears "
                "(verbatim or paraphrased) in the generated email, judged "
                "fact-by-fact by an LLM (gpt-4o-mini). Score in [0, 1]."
            ),
            "tone_adherence": (
                "1-5 rubric score (normalized to 0-1) for how well the email's "
                "register, word choice, and sentence structure match the requested "
                "tone, judged by an LLM (gpt-4o-mini)."
            ),
            "structural_completeness": (
                "Deterministic 0-1 score checking for: Subject line, greeting, "
                "non-trivial body (≥25 words), sign-off, and absence of leaked "
                "CoT tags/template placeholders. No LLM call."
            ),
            "composite_score": (
                "Unweighted mean of the three metrics above, in [0, 1]."
            ),
        },
        label_a: _aggregate(model_a_rows),
        label_b: _aggregate(model_b_rows),
        "per_scenario_comparison": [],
    }

    a_by_id = {r["scenario_id"]: r for r in model_a_rows}
    b_by_id = {r["scenario_id"]: r for r in model_b_rows}

    for sid in sorted(set(a_by_id) | set(b_by_id)):
        a = a_by_id.get(sid, {})
        b = b_by_id.get(sid, {})
        report["per_scenario_comparison"].append({
            "scenario_id": sid,
            label_a: {
                "composite_score":      a.get("composite_score"),
                "fact_recall_score":    a.get("fact_recall_score"),
                "tone_adherence_score": a.get("tone_adherence_score"),
                "structural_score":     a.get("structural_score"),
                "error":                a.get("error"),
            },
            label_b: {
                "composite_score":      b.get("composite_score"),
                "fact_recall_score":    b.get("fact_recall_score"),
                "tone_adherence_score": b.get("tone_adherence_score"),
                "structural_score":     b.get("structural_score"),
                "error":                b.get("error"),
            },
        })

    return report


# Entry point

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Email Generation Assistant — Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.run_eval --model openai --label model_a
  python -m evaluation.run_eval --model ollama --label model_b
  python -m evaluation.run_eval --compare
  python -m evaluation.run_eval --model openai --keyword-fact-recall
        """,
    )
    parser.add_argument(
        "--model", choices=["openai", "ollama"],
        help="Which model variant to run (single-model mode).",
    )
    parser.add_argument(
        "--label", default=None,
        help="Output label used for {label}_results.csv. Defaults to --model value.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run OpenAI (model_a) then Ollama (model_b) and write comparison_report.json.",
    )
    parser.add_argument(
        "--keyword-fact-recall", action="store_true",
        help=(
            "Use free deterministic keyword-overlap fact recall instead of "
            "embedding cosine similarity. No API calls for Metric 1. "
            "Useful for smoke-testing the pipeline without spending credits."
        ),
    )
    args = parser.parse_args()
    use_embedding = not args.keyword_fact_recall

    if args.compare:
        model_a_rows = await run_model("openai", "model_a", use_embedding)
        model_b_rows = await run_model("ollama", "model_b", use_embedding)

        report = build_comparison_report(model_a_rows, model_b_rows, "model_a", "model_b")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RESULTS_DIR / "comparison_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Wrote comparison report → {report_path}")

    elif args.model:
        label = args.label or args.model
        await run_model(args.model, label, use_embedding)

    else:
        parser.error("Provide --model {openai,ollama} or --compare.")


if __name__ == "__main__":
    asyncio.run(main())