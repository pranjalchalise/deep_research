"""
Pytest tests for behavioral evaluators.

Split into two groups:
  - Offline: tests that use pre-built state (fast, no API calls)
  - Live: tests that actually run the pipeline (slow, costs money)

Run offline only:   pytest tests/test_behavioral.py -v -m "not live"
Run everything:     pytest tests/test_behavioral.py -v
"""
import pytest

from tests.evaluators.behavioral import BehavioralEvaluator


evaluator = BehavioralEvaluator()

# ---------------------------------------------------------------------------
# Offline tests — use mock state, no API calls
# ---------------------------------------------------------------------------

class TestStateConsistency:
    def test_consistent_state(self, mock_state):
        result = evaluator.eval_state_consistency(mock_state)
        assert result.passed, f"Inconsistency: {result.details}"

    def test_mismatched_sources_count(self, mock_state):
        mock_state["metadata"]["sources_count"] = 999
        result = evaluator.eval_state_consistency(mock_state)
        assert not result.passed

    def test_mismatched_evidence_count(self, mock_state):
        mock_state["metadata"]["evidence_count"] = 0
        result = evaluator.eval_state_consistency(mock_state)
        assert not result.passed


class TestVerifiedClaimsRatio:
    def test_ratio_matches(self, mock_state):
        result = evaluator.eval_verified_claims_ratio(mock_state)
        assert result.passed, f"Ratio mismatch: {result.details}"

    def test_empty_claims(self, empty_evidence_state):
        result = evaluator.eval_verified_claims_ratio(empty_evidence_state)
        assert result.passed  # should handle gracefully

    def test_wrong_confidence(self, mock_state):
        mock_state["confidence"] = 0.1  # doesn't match the actual supported ratio
        result = evaluator.eval_verified_claims_ratio(mock_state)
        assert not result.passed


# ---------------------------------------------------------------------------
# Live tests — actually run the pipeline, require API keys
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(
    not __import__("os").environ.get("OPENAI_API_KEY")
    or not __import__("os").environ.get("TAVILY_API_KEY"),
    reason="API keys not set — skipping live behavioral tests",
)


@pytest.mark.live
@live
class TestAmbiguityDetection:
    def test_ambiguity_accuracy(self):
        """LLM should correctly flag ambiguous vs clear queries."""
        result = evaluator.eval_ambiguity_detection()
        assert result.passed, f"Low accuracy: {result.details}"
        assert result.score >= 0.5


@pytest.mark.live
@live
class TestConfigEffectiveness:
    def test_config_changes_format(self):
        """Different report_structure values should produce different output."""
        result = evaluator.eval_config_changes_output()
        assert result.passed, f"Config had no effect: {result.details}"


@pytest.mark.live
@live
class TestGapLoop:
    def test_more_iterations_more_evidence(self):
        """More iterations should yield equal or more evidence."""
        result = evaluator.eval_gap_loop_adds_evidence()
        assert result.passed, f"Gap loop regression: {result.details}"


@pytest.mark.live
@live
class TestModeComparison:
    def test_both_modes_produce_reports(self):
        """Single and multi-agent modes should both produce valid reports."""
        result = evaluator.eval_mode_comparison()
        assert result.passed, f"Mode failure: {result.details}"


@pytest.mark.live
@live
class TestIterationConvergence:
    def test_coverage_increases(self):
        """Coverage should be monotonically non-decreasing across iterations."""
        result = evaluator.eval_iteration_convergence()
        assert result.passed, f"Coverage regression: {result.details}"
