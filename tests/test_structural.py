"""
Pytest tests for structural evaluators.

These run against mock states so they're fast (no API calls).
Use `pytest tests/test_structural.py -v` to run.
"""
from tests.evaluators.structural import StructuralEvaluator


evaluator = StructuralEvaluator()


class TestReportMarkdown:
    def test_valid_report_passes(self, mock_state):
        result = evaluator.check_report_has_markdown_structure(mock_state)
        assert result.passed, f"Failed: {result.details}"
        assert result.score >= 0.7

    def test_missing_title_fails(self, mock_state):
        mock_state["report"] = "No title here.\n## Section\nSome text.\n## Sources\n[1] x"
        result = evaluator.check_report_has_markdown_structure(mock_state)
        assert not result.passed or result.score < 1.0

    def test_missing_sources_section(self, mock_state):
        mock_state["report"] = "# Title\n## Section 1\ntext\n## Section 2\ntext"
        result = evaluator.check_report_has_markdown_structure(mock_state)
        assert result.score < 1.0  # should lose points for missing Sources


class TestCitations:
    def test_citations_found(self, mock_state):
        result = evaluator.check_citations_present(mock_state)
        assert result.passed, f"Failed: {result.details}"

    def test_no_citations_fails(self, mock_state):
        mock_state["report"] = "# Report\n\nSome text with no citations whatsoever."
        result = evaluator.check_citations_present(mock_state)
        assert not result.passed

    def test_citations_map_to_sources(self, mock_state):
        result = evaluator.check_citations_map_to_sources(mock_state)
        assert result.passed, f"Orphan citations: {result.details}"


class TestMetadata:
    def test_complete_metadata(self, mock_state):
        result = evaluator.check_metadata_complete(mock_state)
        assert result.passed, f"Failed: {result.details}"

    def test_missing_metadata(self, mock_state):
        mock_state["metadata"] = {"mode": "single"}  # most keys missing
        result = evaluator.check_metadata_complete(mock_state)
        assert not result.passed


class TestEvidence:
    def test_evidence_populated(self, mock_state):
        result = evaluator.check_evidence_populated(mock_state)
        assert result.passed

    def test_empty_evidence(self, empty_evidence_state):
        result = evaluator.check_evidence_populated(empty_evidence_state)
        assert not result.passed

    def test_evidence_fields_valid(self, mock_state):
        result = evaluator.check_evidence_fields(mock_state)
        assert result.passed, f"Failed: {result.details}"


class TestScoresAndBounds:
    def test_scores_in_range(self, mock_state):
        result = evaluator.check_scores_in_bounds(mock_state)
        assert result.passed, f"Failed: {result.details}"

    def test_out_of_range_coverage(self, mock_state):
        mock_state["coverage"] = 1.5  # impossible
        result = evaluator.check_scores_in_bounds(mock_state)
        assert not result.passed

    def test_report_minimum_length(self, mock_state):
        result = evaluator.check_report_minimum_length(mock_state)
        assert result.passed, f"Report too short: {result.details}"


class TestSourcesIntegrity:
    def test_sources_have_urls(self, mock_state):
        result = evaluator.check_sources_have_urls(mock_state)
        assert result.passed

    def test_no_empty_report(self, mock_state):
        result = evaluator.check_no_empty_report(mock_state)
        assert result.passed


class TestFullSuite:
    def test_all_structural_checks_pass(self, mock_state):
        results = evaluator.run_all(mock_state)
        summary = evaluator.summary(results)
        assert summary["pass_rate"] >= 0.8, (
            f"Too many failures: {summary['failures']}"
        )

    def test_multi_agent_state(self, mock_state_multi):
        results = evaluator.run_all(mock_state_multi)
        summary = evaluator.summary(results)
        assert summary["pass_rate"] >= 0.8, (
            f"Multi-agent failures: {summary['failures']}"
        )
