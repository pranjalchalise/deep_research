"""
LLM-as-judge evaluators — use a strong model to assess report quality.

Inspired by standard deep-research evaluation criteria (relevance, groundedness,
completeness, quality) but written from scratch to fit our pipeline's state schema.

Each evaluator uses Pydantic structured output so we get reliable, parseable
scores instead of trying to regex free-text responses.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class JudgeResult:
    """Result from one LLM judge evaluation."""
    name: str
    score: float  # normalized 0-1
    reasoning: str
    sub_scores: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Pydantic models for structured output
# ---------------------------------------------------------------------------

class RelevanceAssessment(BaseModel):
    """Judge whether the report addresses the user's query."""
    section_analysis: str = Field(
        description="For each major section in the report, note whether it's relevant to the query."
    )
    overall_reasoning: str = Field(
        description="Overall assessment of how well the report answers the user's question."
    )
    score: int = Field(
        description="1-5 score. 1=completely off-topic, 3=partially relevant, 5=directly and fully addresses the query."
    )


class GroundedClaim(BaseModel):
    """A single factual claim extracted from the report."""
    claim: str = Field(description="The factual claim as stated in the report.")
    grounded: bool = Field(description="True if this claim is supported by the provided evidence.")


class GroundednessAssessment(BaseModel):
    """Extract claims from the report and check each against the evidence."""
    claims: List[GroundedClaim] = Field(
        description="Every factual claim from the report, with a grounded flag for each."
    )


class CompletenessAssessment(BaseModel):
    """Judge whether the report covers all planned research questions."""
    covered_questions: List[str] = Field(
        description="Research questions that the report adequately addresses."
    )
    missed_questions: List[str] = Field(
        description="Research questions that the report fails to address or barely mentions."
    )
    reasoning: str = Field(description="Explanation of coverage gaps, if any.")
    score: int = Field(
        description="1-5 score. 1=major gaps, 3=covers most points, 5=comprehensive coverage."
    )


class QualityAssessment(BaseModel):
    """Multi-dimensional quality scoring."""
    research_depth: int = Field(description="1-5: thoroughness of analysis and coverage.")
    source_diversity: int = Field(description="1-5: variety and authority of sources used.")
    analytical_rigor: int = Field(description="1-5: critical evaluation, nuance, limitations acknowledged.")
    clarity: int = Field(description="1-5: writing quality, readability, logical flow.")
    citation_integration: int = Field(description="1-5: how well citations are woven into the text.")
    reasoning: str = Field(description="Brief justification for the scores.")


class CitationFaithfulness(BaseModel):
    """Judge whether inline citations accurately represent their sources."""
    reasoning: str = Field(
        description="Analysis of whether cited claims match what the sources actually say."
    )
    score: int = Field(
        description="1-5 score. 1=citations are misleading, 3=mostly accurate, 5=all citations faithful."
    )


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------

class LLMJudgeEvaluator:
    """Run LLM-as-judge evaluations against a pipeline output.

    Uses gpt-4o by default as the judge model. Pass a different model
    name to the constructor if you want to use something else.
    """

    def __init__(self, judge_model: str = "gpt-4o"):
        from langchain_openai import ChatOpenAI
        # use ChatOpenAI directly (not the LLMWrapper) because we need
        # with_structured_output which requires the raw langchain model
        self.llm = ChatOpenAI(model=judge_model, temperature=0.0)

    def run_all(self, state: Dict[str, Any]) -> List[JudgeResult]:
        """Run all LLM judge evaluations. Skips gracefully if state is weird."""
        results = []
        for eval_fn in [
            self.eval_relevance,
            self.eval_groundedness,
            self.eval_completeness,
            self.eval_quality,
            self.eval_citation_faithfulness,
        ]:
            try:
                results.append(eval_fn(state))
            except Exception as e:
                results.append(JudgeResult(
                    name=eval_fn.__name__,
                    score=0.0,
                    reasoning=f"Evaluator failed: {e}",
                ))
        return results

    def summary(self, results: List[JudgeResult]) -> Dict[str, Any]:
        """Aggregate judge results into a summary."""
        scores = {r.name: r.score for r in results}
        avg = sum(scores.values()) / len(scores) if scores else 0
        return {
            "scores": scores,
            "avg_score": round(avg, 3),
            "details": {r.name: r.reasoning for r in results},
        }

    # ------------------------------------------------------------------
    # Evaluators
    # ------------------------------------------------------------------

    def eval_relevance(self, state: Dict[str, Any]) -> JudgeResult:
        """Does the report actually answer the user's question?

        Checks both overall relevance and per-section relevance.
        """
        query = state.get("query", "")
        report = state.get("report", "")

        result = self.llm.with_structured_output(RelevanceAssessment).invoke([
            {"role": "system", "content": (
                "You are evaluating a research report for relevance to the user's query. "
                "First, identify each ## section and assess whether it's on-topic. "
                "Then give an overall score. Be strict: every section should connect "
                "to the query. Off-topic padding should lower the score."
            )},
            {"role": "user", "content": (
                f"USER QUERY:\n{query}\n\n"
                f"REPORT:\n{report}\n\n"
                "Evaluate relevance."
            )},
        ])

        return JudgeResult(
            name="relevance",
            score=result.score / 5,
            reasoning=result.overall_reasoning,
        )

    def eval_groundedness(self, state: Dict[str, Any]) -> JudgeResult:
        """Extract every factual claim from the report and check if it's
        supported by the evidence the pipeline actually collected.

        This is the most granular evaluator — it gives a ratio of
        grounded claims to total claims, not just a vibes-based score.
        """
        report = state.get("report", "")
        evidence = state.get("evidence", [])

        # format the evidence so the judge can check against it
        evidence_text = "\n".join(
            f"- {e.get('fact', '')} [from: {e.get('source_title', '?')}]"
            for e in evidence
        )
        if not evidence_text:
            evidence_text = "(no evidence collected)"

        result = self.llm.with_structured_output(GroundednessAssessment).with_retry(
            stop_after_attempt=3
        ).invoke([
            {"role": "system", "content": (
                "You are checking whether a research report is grounded in its evidence. "
                "Extract every factual claim from the report (not opinions or hedged statements). "
                "For each claim, determine if it is directly supported by the provided evidence. "
                "A claim is grounded if the evidence contains information that supports it. "
                "Basic common knowledge (e.g. '2+2=4') counts as grounded."
            )},
            {"role": "user", "content": (
                f"EVIDENCE COLLECTED BY THE PIPELINE:\n{evidence_text}\n\n"
                f"REPORT:\n{report}\n\n"
                "Extract all factual claims and check each one."
            )},
        ])

        if not result.claims:
            return JudgeResult(name="groundedness", score=0.0, reasoning="no claims extracted")

        grounded = sum(1 for c in result.claims if c.grounded)
        total = len(result.claims)
        ratio = grounded / total

        ungrounded = [c.claim for c in result.claims if not c.grounded]
        reasoning = (
            f"{grounded}/{total} claims grounded ({ratio:.0%}). "
            f"Ungrounded: {ungrounded[:3]}" if ungrounded else f"All {total} claims grounded."
        )

        return JudgeResult(
            name="groundedness",
            score=round(ratio, 3),
            reasoning=reasoning,
        )

    def eval_completeness(self, state: Dict[str, Any]) -> JudgeResult:
        """Does the report cover all the research questions that were planned?

        This checks the pipeline's own plan against its output — if it planned
        to answer 5 questions but only addressed 3, that's incomplete.
        """
        query = state.get("query", "")
        report = state.get("report", "")
        research_questions = state.get("research_questions", [])
        aspects = state.get("aspects_to_cover", [])

        plan_text = "Research questions:\n"
        for q in research_questions:
            plan_text += f"- {q}\n"
        if aspects:
            plan_text += "\nAspects to cover:\n"
            for a in aspects:
                plan_text += f"- {a}\n"

        result = self.llm.with_structured_output(CompletenessAssessment).invoke([
            {"role": "system", "content": (
                "You are evaluating whether a research report fully addresses its planned "
                "research questions and aspects. Compare the report against the plan below. "
                "A question is 'covered' if the report contains a substantive answer to it, "
                "not just a passing mention."
            )},
            {"role": "user", "content": (
                f"ORIGINAL QUERY:\n{query}\n\n"
                f"RESEARCH PLAN:\n{plan_text}\n\n"
                f"REPORT:\n{report}\n\n"
                "Assess completeness."
            )},
        ])

        return JudgeResult(
            name="completeness",
            score=result.score / 5,
            reasoning=result.reasoning,
            sub_scores={
                "covered": len(result.covered_questions),
                "missed": len(result.missed_questions),
            },
        )

    def eval_quality(self, state: Dict[str, Any]) -> JudgeResult:
        """Multi-dimensional quality assessment across 5 axes.

        Returns the average of all sub-scores plus individual breakdowns.
        """
        query = state.get("query", "")
        report = state.get("report", "")

        result = self.llm.with_structured_output(QualityAssessment).invoke([
            {"role": "system", "content": (
                "You are an expert evaluator assessing a research report's quality. "
                "Score each dimension on a 1-5 scale:\n"
                "- Research depth: thoroughness, coverage of relevant aspects\n"
                "- Source diversity: range and authority of sources\n"
                "- Analytical rigor: critical thinking, nuance, limitations noted\n"
                "- Clarity: writing quality, flow, readability\n"
                "- Citation integration: how well citations support the narrative\n\n"
                "Be honest. A score of 3 means 'acceptable but not impressive.' "
                "Reserve 5 for genuinely excellent work."
            )},
            {"role": "user", "content": (
                f"USER QUERY:\n{query}\n\n"
                f"REPORT:\n{report}\n\n"
                "Evaluate quality across all dimensions."
            )},
        ])

        sub = {
            "research_depth": result.research_depth / 5,
            "source_diversity": result.source_diversity / 5,
            "analytical_rigor": result.analytical_rigor / 5,
            "clarity": result.clarity / 5,
            "citation_integration": result.citation_integration / 5,
        }
        avg = sum(sub.values()) / len(sub)

        return JudgeResult(
            name="quality",
            score=round(avg, 3),
            reasoning=result.reasoning,
            sub_scores=sub,
        )

    def eval_citation_faithfulness(self, state: Dict[str, Any]) -> JudgeResult:
        """Do the inline citations in the report accurately represent their sources?

        This is different from groundedness — groundedness checks if claims are
        supported by evidence. This checks if the specific citation markers
        point to the right sources and don't misrepresent them.
        """
        report = state.get("report", "")
        sources = state.get("sources", {})
        evidence = state.get("evidence", [])

        sources_text = "\n".join(
            f"[{s.get('source_id', '?')}] {s.get('title', '?')} — {s.get('url', '?')}"
            for s in sources.values()
        )
        evidence_text = "\n".join(
            f"From [{e.get('source_title', '?')}]: {e.get('fact', '')}"
            for e in evidence[:15]  # cap to keep context reasonable
        )

        result = self.llm.with_structured_output(CitationFaithfulness).invoke([
            {"role": "system", "content": (
                "You are checking whether a report's inline citations [N] are faithful "
                "to their sources. For each citation in the report, check:\n"
                "1. Does the cited claim actually come from that source?\n"
                "2. Is the claim accurately represented (not distorted or exaggerated)?\n"
                "3. Are citations placed at appropriate points (not random)?\n\n"
                "You have the source list and extracted evidence below."
            )},
            {"role": "user", "content": (
                f"SOURCES:\n{sources_text}\n\n"
                f"EVIDENCE EXTRACTED:\n{evidence_text}\n\n"
                f"REPORT:\n{report}\n\n"
                "Assess citation faithfulness."
            )},
        ])

        return JudgeResult(
            name="citation_faithfulness",
            score=result.score / 5,
            reasoning=result.reasoning,
        )
