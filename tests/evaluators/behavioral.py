"""
Behavioral evaluators — test properties of the pipeline's *behavior*,
not just the final output.

These go beyond what standard deep-research evals check. They verify that
architectural features actually work: config knobs take effect, gap detection
improves coverage across iterations, multi-agent parallelism produces
comparable quality to single-agent, ambiguity detection fires correctly, etc.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tests.conftest import run_pipeline


@dataclass
class BehavioralResult:
    """Result from a behavioral evaluation."""
    name: str
    passed: bool
    score: float
    details: str = ""


class BehavioralEvaluator:
    """Behavioral tests that probe architectural correctness.

    Most of these need to actually run the pipeline (they test behavior,
    not just output shape), so they're slower and cost real API calls.
    Use the quick= variants or pre-computed states when possible.
    """

    def eval_config_changes_output(
        self,
        query: str = "What is quantum computing?",
        mode: str = "single",
    ) -> BehavioralResult:
        """Verify that changing report_structure config actually changes output format.

        Runs the same query with 'detailed' vs 'bullet_points' and checks that
        the outputs are structurally different. If config has no effect, something
        is broken.
        """
        state_detailed = run_pipeline(
            query, mode=mode,
            config_overrides={"report_structure": "detailed"},
            max_iterations=1,
        )
        state_bullets = run_pipeline(
            query, mode=mode,
            config_overrides={"report_structure": "bullet_points"},
            max_iterations=1,
        )

        report_d = state_detailed.get("report", "")
        report_b = state_bullets.get("report", "")

        # bullet_points format should have more list items
        bullets_d = len(re.findall(r"^\s*[-*•]\s", report_d, re.MULTILINE))
        bullets_b = len(re.findall(r"^\s*[-*•]\s", report_b, re.MULTILINE))

        # they shouldn't be identical
        identical = report_d.strip() == report_b.strip()

        # bullet_points should have noticeably more bullets OR fewer paragraphs
        format_differs = bullets_b > bullets_d or not identical

        passed = format_differs and not identical
        score = 1.0 if passed else 0.0

        return BehavioralResult(
            name="config_changes_output",
            passed=passed,
            score=score,
            details=(
                f"detailed: {bullets_d} bullets, {len(report_d)} chars | "
                f"bullet_points: {bullets_b} bullets, {len(report_b)} chars | "
                f"identical={identical}"
            ),
        )

    def eval_gap_loop_adds_evidence(
        self,
        query: str = "What are the environmental impacts of cryptocurrency mining?",
    ) -> BehavioralResult:
        """Check that running more iterations actually collects more evidence.

        Compares a 1-iteration run vs a 3-iteration run. The 3-iteration run
        should have more evidence (or at least the same — the gap loop shouldn't
        lose information).
        """
        state_1 = run_pipeline(query, mode="single", max_iterations=1)
        state_3 = run_pipeline(query, mode="single", max_iterations=3)

        ev_1 = len(state_1.get("evidence", []))
        ev_3 = len(state_3.get("evidence", []))
        src_1 = len(state_1.get("sources", {}))
        src_3 = len(state_3.get("sources", {}))

        # more iterations should mean equal or more evidence
        more_evidence = ev_3 >= ev_1
        more_sources = src_3 >= src_1

        passed = more_evidence and more_sources
        # score based on how much more evidence we got
        improvement = (ev_3 - ev_1) / max(ev_1, 1)
        score = min(improvement + 0.5, 1.0) if passed else 0.0

        return BehavioralResult(
            name="gap_loop_adds_evidence",
            passed=passed,
            score=round(score, 3),
            details=(
                f"1 iter: {ev_1} evidence, {src_1} sources | "
                f"3 iter: {ev_3} evidence, {src_3} sources | "
                f"improvement: {improvement:.0%}"
            ),
        )

    def eval_mode_comparison(
        self,
        query: str = "What are the pros and cons of remote work?",
    ) -> BehavioralResult:
        """Both single and multi-agent modes should produce reports that
        pass basic quality checks.

        This isn't about which is better — it's about verifying both paths
        work and produce reasonable output. We check minimum evidence, source
        count, and report length for each.
        """
        state_single = run_pipeline(query, mode="single", max_iterations=2)
        state_multi = run_pipeline(query, mode="multi", max_iterations=2)

        checks = []
        for label, state in [("single", state_single), ("multi", state_multi)]:
            ev = len(state.get("evidence", []))
            src = len(state.get("sources", {}))
            report_len = len(state.get("report", "").split())
            has_report = report_len >= 100

            checks.append({
                "mode": label,
                "evidence": ev,
                "sources": src,
                "words": report_len,
                "ok": ev >= 2 and src >= 1 and has_report,
            })

        both_ok = all(c["ok"] for c in checks)
        score = sum(1 for c in checks if c["ok"]) / 2

        return BehavioralResult(
            name="mode_comparison",
            passed=both_ok,
            score=score,
            details=" | ".join(
                f"{c['mode']}: {c['evidence']}ev, {c['sources']}src, {c['words']}w, ok={c['ok']}"
                for c in checks
            ),
        )

    def eval_ambiguity_detection(
        self,
        ambiguous_queries: Optional[List[str]] = None,
        clear_queries: Optional[List[str]] = None,
    ) -> BehavioralResult:
        """Ambiguous queries should trigger is_ambiguous=True, clear ones shouldn't.

        Rather than running the full pipeline, we just run the understand node
        in isolation to keep this cheap and fast.
        """
        from langchain_core.runnables import RunnableConfig
        from src.pipeline.nodes import understand_node

        if ambiguous_queries is None:
            ambiguous_queries = [
                "Tell me about Mercury.",
                "Explain the impact of AI.",
                "What happened with the crash?",
            ]
        if clear_queries is None:
            clear_queries = [
                "What is CRISPR-Cas9 and how does it work?",
                "List the planets in our solar system in order.",
                "Who won the 2024 Nobel Prize in Physics?",
            ]

        config = RunnableConfig(configurable={"model": "gpt-4o", "fast_model": "gpt-4o-mini"})
        correct = 0
        total = 0
        details_parts = []

        for query in ambiguous_queries:
            state = {"query": query}
            result = understand_node(state, config)
            is_amb = result.get("is_ambiguous", False)
            correct += int(is_amb)
            total += 1
            details_parts.append(f"'{query[:30]}...' -> ambiguous={is_amb} (expected True)")

        for query in clear_queries:
            state = {"query": query}
            result = understand_node(state, config)
            is_amb = result.get("is_ambiguous", False)
            correct += int(not is_amb)
            total += 1
            details_parts.append(f"'{query[:30]}...' -> ambiguous={is_amb} (expected False)")

        accuracy = correct / total if total else 0
        # we allow some slack since ambiguity is inherently subjective
        passed = accuracy >= 0.65

        return BehavioralResult(
            name="ambiguity_detection",
            passed=passed,
            score=round(accuracy, 3),
            details=f"{correct}/{total} correct. " + "; ".join(details_parts),
        )

    def eval_iteration_convergence(
        self,
        query: str = "What are the health benefits and risks of intermittent fasting?",
    ) -> BehavioralResult:
        """Coverage should increase (or plateau) over iterations — never decrease.

        Runs the pipeline with enough iterations to loop, then checks that
        coverage was monotonically non-decreasing. If coverage drops between
        iterations, gap detection or the search loop has a bug.

        Since we can't easily inspect intermediate states from outside the graph,
        we compare 1-iter vs 2-iter vs 3-iter coverage as a proxy.
        """
        coverages = []
        for n_iter in [1, 2, 3]:
            state = run_pipeline(query, mode="single", max_iterations=n_iter)
            cov = state.get("coverage", 0)
            coverages.append(cov)

        # check monotonically non-decreasing
        monotonic = all(coverages[i] <= coverages[i+1] + 0.01 for i in range(len(coverages)-1))
        # small tolerance (0.01) because coverage is LLM-estimated, not deterministic

        # also check that we got SOME improvement
        improved = coverages[-1] > coverages[0]

        passed = monotonic
        score = 1.0 if monotonic and improved else 0.5 if monotonic else 0.0

        return BehavioralResult(
            name="iteration_convergence",
            passed=passed,
            score=score,
            details=f"coverage by max_iterations: {[round(c, 3) for c in coverages]}, monotonic={monotonic}",
        )

    # ------------------------------------------------------------------
    # Quick offline checks (don't call APIs)
    # ------------------------------------------------------------------

    def eval_state_consistency(self, state: Dict[str, Any]) -> BehavioralResult:
        """Cross-check that metadata numbers match actual state contents.

        metadata.sources_count should match len(sources), etc. If these
        diverge, there's a bookkeeping bug in the pipeline.
        """
        metadata = state.get("metadata", {})
        issues = []

        actual_sources = len(state.get("sources", {}))
        reported_sources = metadata.get("sources_count", -1)
        if reported_sources != actual_sources:
            issues.append(f"sources_count: metadata={reported_sources}, actual={actual_sources}")

        actual_evidence = len(state.get("evidence", []))
        reported_evidence = metadata.get("evidence_count", -1)
        if reported_evidence != actual_evidence:
            issues.append(f"evidence_count: metadata={reported_evidence}, actual={actual_evidence}")

        reported_mode = metadata.get("mode", "?")
        actual_mode = state.get("mode", "?")
        if reported_mode != actual_mode:
            issues.append(f"mode: metadata={reported_mode}, state={actual_mode}")

        passed = len(issues) == 0
        score = 1.0 - (len(issues) * 0.33)

        return BehavioralResult(
            name="state_consistency",
            passed=passed,
            score=max(0, round(score, 3)),
            details="; ".join(issues) if issues else "all metadata matches state",
        )

    def eval_verified_claims_ratio(self, state: Dict[str, Any]) -> BehavioralResult:
        """The confidence score should match the actual verified claims ratio.

        If 6/10 claims are supported, confidence should be ~0.6. We allow a
        tolerance since rounding and edge cases exist.
        """
        verified = state.get("verified_claims", [])
        reported_confidence = state.get("confidence", -1)

        if not verified:
            return BehavioralResult(
                name="verified_claims_ratio",
                passed=True, score=1.0,
                details="no verified claims to check",
            )

        supported = sum(1 for v in verified if v.get("supported", False))
        expected_conf = supported / len(verified)
        diff = abs(expected_conf - reported_confidence)

        passed = diff < 0.05  # 5% tolerance
        score = max(0, 1.0 - diff * 5)

        return BehavioralResult(
            name="verified_claims_ratio",
            passed=passed,
            score=round(score, 3),
            details=(
                f"supported={supported}/{len(verified)} -> expected {expected_conf:.3f}, "
                f"reported {reported_confidence:.3f}, diff={diff:.3f}"
            ),
        )
