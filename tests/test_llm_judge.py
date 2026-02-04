"""
Pytest tests for LLM-as-judge evaluators.

These call the OpenAI API (through the judge model) so they cost money
and take a few seconds each. Mark them with @pytest.mark.llm so they
can be skipped in CI with `pytest -m "not llm"`.

For offline testing, run against saved snapshots:
    pytest tests/test_llm_judge.py -v --snapshot=factual_simple
"""
import pytest

from tests.evaluators.llm_judge import LLMJudgeEvaluator


# skip all tests in this file if no OPENAI_API_KEY
pytestmark = pytest.mark.skipif(
    not __import__("os").environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping LLM judge tests",
)


@pytest.fixture(scope="module")
def judge():
    return LLMJudgeEvaluator(judge_model="gpt-4o")


class TestRelevance:
    def test_relevant_report_scores_high(self, judge, mock_state):
        result = judge.eval_relevance(mock_state)
        assert result.score >= 0.6, f"Low relevance: {result.reasoning}"

    def test_off_topic_report_scores_low(self, judge, mock_state):
        mock_state["query"] = "What is quantum computing?"
        mock_state["report"] = (
            "# Chocolate Cake Recipe\n\n"
            "## Ingredients\nFlour, sugar, cocoa.\n"
            "## Steps\nMix and bake.\n## Sources\n[1] recipe.com"
        )
        result = judge.eval_relevance(mock_state)
        assert result.score <= 0.4, f"Off-topic report scored too high: {result.score}"


class TestGroundedness:
    def test_grounded_report(self, judge, mock_state):
        result = judge.eval_groundedness(mock_state)
        # mock state has evidence that matches the report, should be mostly grounded
        assert result.score >= 0.5, f"Low groundedness: {result.reasoning}"

    def test_fabricated_report(self, judge, mock_state):
        mock_state["evidence"] = [
            {"fact": "The sky is blue.", "source_url": "x", "source_title": "x"}
        ]
        mock_state["report"] = (
            "# Report\n\n## Findings\n"
            "Quantum computers can travel through time [1]. "
            "They are powered by dark matter [1]. "
            "NASA uses them to communicate with aliens [1].\n"
            "## Sources\n[1] x"
        )
        result = judge.eval_groundedness(mock_state)
        assert result.score <= 0.4, f"Fabricated claims scored too high: {result.score}"


class TestCompleteness:
    def test_complete_report(self, judge, mock_state):
        result = judge.eval_completeness(mock_state)
        assert result.score >= 0.5, f"Low completeness: {result.reasoning}"

    def test_partial_report(self, judge, mock_state):
        # keep research questions but strip report to just one topic
        mock_state["research_questions"] = [
            "What is quantum computing?",
            "How do qubits work?",
            "What are current applications?",
            "What are the main challenges?",
            "Who are the key players?",
        ]
        mock_state["report"] = (
            "# Report\n\n## Overview\nQuantum computing exists. [1]\n"
            "## Sources\n[1] example.com"
        )
        result = judge.eval_completeness(mock_state)
        assert result.score <= 0.6, f"Incomplete report scored too high: {result.score}"


class TestQuality:
    def test_quality_sub_scores(self, judge, mock_state):
        result = judge.eval_quality(mock_state)
        assert result.sub_scores is not None, "Should return sub-scores"
        assert all(0 <= v <= 1 for v in result.sub_scores.values()), (
            f"Sub-scores out of range: {result.sub_scores}"
        )

    def test_stub_report_low_quality(self, judge, mock_state):
        mock_state["report"] = "# Report\n\nNot much here.\n## Sources\nNone."
        result = judge.eval_quality(mock_state)
        assert result.score <= 0.5, f"Stub report quality too high: {result.score}"


class TestCitationFaithfulness:
    def test_faithful_citations(self, judge, mock_state):
        result = judge.eval_citation_faithfulness(mock_state)
        assert result.score >= 0.4, f"Low citation faithfulness: {result.reasoning}"


class TestFullJudgeSuite:
    def test_all_judges_return_scores(self, judge, mock_state):
        results = judge.run_all(mock_state)
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        for r in results:
            assert 0 <= r.score <= 1, f"{r.name} score out of range: {r.score}"
            assert r.reasoning, f"{r.name} has no reasoning"

    def test_summary_format(self, judge, mock_state):
        results = judge.run_all(mock_state)
        summary = judge.summary(results)
        assert "scores" in summary
        assert "avg_score" in summary
        assert 0 <= summary["avg_score"] <= 1
