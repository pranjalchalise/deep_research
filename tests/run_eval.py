#!/usr/bin/env python3
"""
Main evaluation runner for the research pipeline.

Runs the pipeline against a set of test cases, then evaluates the outputs
using structural checks, LLM judges, and behavioral tests. Prints a
summary table and optionally saves results to JSON.

Usage:
    # Quick run (mock state, structural only — no API calls)
    python -m tests.run_eval --offline

    # Full run against live pipeline (costs real API calls)
    python -m tests.run_eval --live

    # Run with specific cases
    python -m tests.run_eval --live --cases factual_simple broad_comparison

    # Save results to file
    python -m tests.run_eval --live --output results.json

    # Use a specific judge model
    python -m tests.run_eval --live --judge-model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from tests.eval_cases import ALL_CASES, QUICK_CASES, EvalCase
from tests.conftest import run_pipeline, make_mock_state, save_snapshot
from tests.evaluators.structural import StructuralEvaluator, EvalResult
from tests.evaluators.llm_judge import LLMJudgeEvaluator
from tests.evaluators.behavioral import BehavioralEvaluator


def print_header(text: str):
    width = 70
    print(f"\n{'=' * width}")
    print(f" {text}")
    print(f"{'=' * width}")


def print_table(rows: List[Dict[str, Any]], columns: List[str]):
    """Print a simple ASCII table."""
    col_widths = {}
    for col in columns:
        max_w = len(col)
        for row in rows:
            val = str(row.get(col, ""))
            max_w = max(max_w, len(val))
        col_widths[col] = min(max_w + 2, 40)

    header = "".join(col.ljust(col_widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = ""
        for col in columns:
            val = str(row.get(col, ""))
            if len(val) > col_widths[col] - 2:
                val = val[:col_widths[col] - 5] + "..."
            line += val.ljust(col_widths[col])
        print(line)


def run_structural_eval(state: Dict[str, Any], case_name: str) -> Dict[str, Any]:
    """Run structural evaluators and return summary."""
    evaluator = StructuralEvaluator()
    results = evaluator.run_all(state)
    summary = evaluator.summary(results)
    summary["case"] = case_name
    summary["type"] = "structural"
    return summary


def run_llm_judge_eval(
    state: Dict[str, Any], case_name: str, judge_model: str
) -> Dict[str, Any]:
    """Run LLM judge evaluators and return summary."""
    evaluator = LLMJudgeEvaluator(judge_model=judge_model)
    results = evaluator.run_all(state)
    summary = evaluator.summary(results)
    summary["case"] = case_name
    summary["type"] = "llm_judge"
    return summary


def run_behavioral_eval_offline(state: Dict[str, Any], case_name: str) -> Dict[str, Any]:
    """Run the offline behavioral checks (no API calls)."""
    evaluator = BehavioralEvaluator()
    results = [
        evaluator.eval_state_consistency(state),
        evaluator.eval_verified_claims_ratio(state),
    ]

    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / len(results)

    return {
        "case": case_name,
        "type": "behavioral",
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 3),
        "avg_score": round(avg_score, 3),
        "failures": [
            {"name": r.name, "details": r.details}
            for r in results if not r.passed
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Research pipeline evaluation runner")
    parser.add_argument(
        "--live", action="store_true",
        help="Run the actual pipeline (costs API calls). Without this, uses mock states.",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Specific case names to run (default: all cases if --live, quick cases if offline)",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4o",
        help="Model to use as the LLM judge (default: gpt-4o)",
    )
    parser.add_argument(
        "--skip-llm-judge", action="store_true",
        help="Skip LLM judge evaluators (useful for quick structural-only runs)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to a JSON file",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=2,
        help="Max iterations for pipeline runs (default: 2)",
    )
    parser.add_argument(
        "--from-snapshots", action="store_true",
        help="Re-score from saved snapshots instead of running pipeline or using mocks.",
    )
    args = parser.parse_args()

    # figure out which cases to run
    if args.cases:
        case_map = {c.name: c for c in ALL_CASES}
        cases = []
        for name in args.cases:
            if name not in case_map:
                print(f"Unknown case: {name}. Available: {list(case_map.keys())}")
                sys.exit(1)
            cases.append(case_map[name])
    else:
        cases = ALL_CASES if args.live else QUICK_CASES

    mode_label = "LIVE (real API calls)" if args.live else "SNAPSHOTS (saved results)" if args.from_snapshots else "OFFLINE (mock states)"
    print_header("Research Pipeline Evaluation")
    print(f"Mode: {mode_label}")
    print(f"Cases: {[c.name for c in cases]}")
    print(f"Judge model: {args.judge_model}")
    if args.skip_llm_judge:
        print("LLM judge: SKIPPED")

    all_results = []

    for case in cases:
        print_header(f"Case: {case.name} ({case.mode} mode)")
        print(f"Query: {case.query}")

        # get the state — run pipeline, load snapshot, or build mock
        if args.live:
            print("Running pipeline...")
            t0 = time.time()
            try:
                state = run_pipeline(
                    query=case.query,
                    mode=case.mode,
                    max_iterations=args.max_iterations,
                )
                elapsed = time.time() - t0
                print(f"Pipeline finished in {elapsed:.1f}s")

                # save snapshot for future offline runs
                save_snapshot(state, case.name)
                print(f"Snapshot saved to tests/snapshots/{case.name}.json")

            except Exception as e:
                print(f"Pipeline FAILED: {e}")
                all_results.append({
                    "case": case.name, "type": "pipeline_error",
                    "error": str(e),
                })
                continue
        elif args.from_snapshots:
            try:
                from tests.conftest import load_snapshot
                state = load_snapshot(case.name)
                print("Loaded from snapshot")
            except FileNotFoundError:
                print(f"No snapshot found for {case.name} — run with --live first")
                continue
        else:
            state = make_mock_state(query=case.query, mode=case.mode)

        # --- structural eval ---
        print("\nRunning structural evaluators...")
        structural = run_structural_eval(state, case.name)
        all_results.append(structural)
        print(f"  Structural: {structural['passed']}/{structural['total']} passed "
              f"(avg score: {structural['avg_score']:.3f})")
        if structural["failures"]:
            for f in structural["failures"]:
                print(f"    FAIL: {f['name']} — {f['details']}")

        # --- llm judge eval ---
        if not args.skip_llm_judge:
            print("\nRunning LLM judge evaluators...")
            try:
                llm_judge = run_llm_judge_eval(state, case.name, args.judge_model)
                all_results.append(llm_judge)
                print(f"  LLM Judge avg score: {llm_judge['avg_score']:.3f}")
                for name, score in llm_judge.get("scores", {}).items():
                    print(f"    {name}: {score:.3f}")
            except Exception as e:
                print(f"  LLM Judge FAILED: {e}")
                all_results.append({
                    "case": case.name, "type": "llm_judge_error",
                    "error": str(e),
                })

        # --- behavioral eval (offline parts) ---
        print("\nRunning behavioral evaluators...")
        behavioral = run_behavioral_eval_offline(state, case.name)
        all_results.append(behavioral)
        print(f"  Behavioral: {behavioral['passed']}/{behavioral['total']} passed "
              f"(avg score: {behavioral['avg_score']:.3f})")
        if behavioral["failures"]:
            for f in behavioral["failures"]:
                print(f"    FAIL: {f['name']} — {f['details']}")

    # --- summary table ---
    print_header("Summary")

    summary_rows = []
    for r in all_results:
        if "error" in r:
            summary_rows.append({
                "case": r["case"], "type": r["type"],
                "score": "ERROR", "pass_rate": "—",
            })
        else:
            summary_rows.append({
                "case": r["case"],
                "type": r["type"],
                "score": f"{r.get('avg_score', 0):.3f}",
                "pass_rate": f"{r.get('pass_rate', 0):.0%}" if "pass_rate" in r else "—",
            })

    print_table(summary_rows, ["case", "type", "score", "pass_rate"])

    # overall
    numeric_scores = [
        r["avg_score"] for r in all_results
        if isinstance(r.get("avg_score"), (int, float))
    ]
    if numeric_scores:
        overall = sum(numeric_scores) / len(numeric_scores)
        print(f"\nOverall average score: {overall:.3f}")

    # save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    # exit code: fail if any structural check has <80% pass rate
    structural_results = [r for r in all_results if r.get("type") == "structural"]
    if any(r.get("pass_rate", 1) < 0.8 for r in structural_results):
        print("\nFAILED: structural pass rate below 80%")
        sys.exit(1)
    else:
        print("\nPASSED")


if __name__ == "__main__":
    main()
