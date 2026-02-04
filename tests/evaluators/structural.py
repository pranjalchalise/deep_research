"""
Structural evaluators — deterministic checks that don't need an LLM.

These verify that the pipeline output has the right shape: valid markdown,
citations that actually point to sources, metadata that makes sense, etc.
Fast to run, no API costs, good for catching regressions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class EvalResult:
    """Result from one evaluator."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: str = ""


class StructuralEvaluator:
    """Run all structural checks against a pipeline output state."""

    def run_all(self, state: Dict[str, Any]) -> List[EvalResult]:
        """Run every structural check and return the results."""
        checks = [
            self.check_report_has_markdown_structure,
            self.check_citations_present,
            self.check_citations_map_to_sources,
            self.check_metadata_complete,
            self.check_evidence_populated,
            self.check_scores_in_bounds,
            self.check_report_minimum_length,
            self.check_evidence_fields,
            self.check_sources_have_urls,
            self.check_no_empty_report,
        ]
        results = []
        for check_fn in checks:
            try:
                results.append(check_fn(state))
            except Exception as e:
                results.append(EvalResult(
                    name=check_fn.__name__,
                    passed=False,
                    score=0.0,
                    details=f"Evaluator crashed: {e}",
                ))
        return results

    def summary(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Aggregate results into a summary dict."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_score = sum(r.score for r in results) / total if total else 0
        return {
            "passed": passed,
            "total": total,
            "pass_rate": round(passed / total, 3) if total else 0,
            "avg_score": round(avg_score, 3),
            "failures": [
                {"name": r.name, "details": r.details}
                for r in results if not r.passed
            ],
        }

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_report_has_markdown_structure(self, state: Dict[str, Any]) -> EvalResult:
        """Report should have a title (# heading) and at least one section (## heading)."""
        report = state.get("report", "")
        has_title = bool(re.search(r"^#\s+\S", report, re.MULTILINE))
        sections = re.findall(r"^##\s+\S", report, re.MULTILINE)
        has_sources_section = bool(
            re.search(r"^##\s+(Sources|References)", report, re.MULTILINE)
        )

        score_parts = []
        if has_title:
            score_parts.append(0.3)
        if len(sections) >= 2:
            score_parts.append(0.4)
        if has_sources_section:
            score_parts.append(0.3)

        score = sum(score_parts)
        passed = has_title and len(sections) >= 2 and has_sources_section
        details = f"title={has_title}, sections={len(sections)}, sources_section={has_sources_section}"

        return EvalResult(
            name="report_has_markdown_structure",
            passed=passed,
            score=score,
            details=details,
        )

    def check_citations_present(self, state: Dict[str, Any]) -> EvalResult:
        """Report should contain inline citation markers like [1], [2], etc."""
        report = state.get("report", "")
        # match [1], [2], [12] etc but not things like [source_id]
        citations = re.findall(r"\[(\d+)\]", report)
        unique_citations = set(citations)
        count = len(unique_citations)

        # at least 2 distinct citations for a real report
        passed = count >= 2
        score = min(count / 5, 1.0)  # 5+ distinct citations = perfect score

        return EvalResult(
            name="citations_present",
            passed=passed,
            score=score,
            details=f"{count} unique citation markers found: {sorted(unique_citations)[:10]}",
        )

    def check_citations_map_to_sources(self, state: Dict[str, Any]) -> EvalResult:
        """Every [N] citation in the report body should have a matching source."""
        report = state.get("report", "")
        sources = state.get("sources", {})

        # extract source IDs from the sources dict
        valid_ids = set()
        for src in sources.values():
            sid = src.get("source_id")
            if sid is not None:
                valid_ids.add(str(sid))

        # find all citation numbers used in the report (excluding the Sources section)
        # split at Sources/References header if present
        body = re.split(r"^##\s+(Sources|References)", report, flags=re.MULTILINE)[0]
        cited_ids = set(re.findall(r"\[(\d+)\]", body))

        if not cited_ids:
            return EvalResult(
                name="citations_map_to_sources",
                passed=False, score=0.0,
                details="No citations found in report body",
            )

        matched = cited_ids & valid_ids
        orphans = cited_ids - valid_ids
        ratio = len(matched) / len(cited_ids) if cited_ids else 0

        return EvalResult(
            name="citations_map_to_sources",
            passed=len(orphans) == 0,
            score=round(ratio, 3),
            details=f"matched={len(matched)}/{len(cited_ids)}, orphan_ids={sorted(orphans)[:5]}",
        )

    def check_metadata_complete(self, state: Dict[str, Any]) -> EvalResult:
        """Metadata dict should have the expected keys with valid types."""
        metadata = state.get("metadata")
        if not isinstance(metadata, dict):
            return EvalResult(
                name="metadata_complete",
                passed=False, score=0.0,
                details="metadata is missing or not a dict",
            )

        expected_keys = {
            "iterations": (int, float),
            "coverage": (int, float),
            "confidence": (int, float),
            "sources_count": (int, float),
            "evidence_count": (int, float),
            "mode": (str,),
        }

        present = 0
        issues = []
        for key, types in expected_keys.items():
            val = metadata.get(key)
            if val is None:
                issues.append(f"missing '{key}'")
            elif not isinstance(val, types):
                issues.append(f"'{key}' has wrong type: {type(val).__name__}")
            else:
                present += 1

        score = present / len(expected_keys)
        return EvalResult(
            name="metadata_complete",
            passed=len(issues) == 0,
            score=round(score, 3),
            details="; ".join(issues) if issues else "all keys present and valid",
        )

    def check_evidence_populated(self, state: Dict[str, Any]) -> EvalResult:
        """Evidence list should have at least some items (unless we expect failure)."""
        evidence = state.get("evidence", [])
        count = len(evidence)
        # we consider 3+ evidence items a healthy run
        passed = count >= 1
        score = min(count / 5, 1.0)

        return EvalResult(
            name="evidence_populated",
            passed=passed,
            score=score,
            details=f"{count} evidence items",
        )

    def check_scores_in_bounds(self, state: Dict[str, Any]) -> EvalResult:
        """Coverage and confidence should be between 0 and 1."""
        coverage = state.get("coverage", 0)
        confidence = state.get("confidence", 0)
        metadata = state.get("metadata", {})

        issues = []
        for name, val in [
            ("coverage", coverage),
            ("confidence", confidence),
            ("metadata.coverage", metadata.get("coverage", 0)),
            ("metadata.confidence", metadata.get("confidence", 0)),
        ]:
            if not isinstance(val, (int, float)):
                issues.append(f"{name} is not numeric: {type(val).__name__}")
            elif val < 0 or val > 1:
                issues.append(f"{name}={val} is out of [0,1] range")

        passed = len(issues) == 0
        score = 1.0 if passed else max(0, 1.0 - len(issues) * 0.25)

        return EvalResult(
            name="scores_in_bounds",
            passed=passed,
            score=score,
            details="; ".join(issues) if issues else "all scores in [0,1]",
        )

    def check_report_minimum_length(self, state: Dict[str, Any]) -> EvalResult:
        """Report should be at least 200 words for a detailed report.

        This catches cases where the pipeline bails early with a stub.
        """
        report = state.get("report", "")
        word_count = len(report.split())

        # 200 words minimum for something meaningful, 500+ is solid
        passed = word_count >= 200
        score = min(word_count / 500, 1.0)

        return EvalResult(
            name="report_minimum_length",
            passed=passed,
            score=round(score, 3),
            details=f"{word_count} words",
        )

    def check_evidence_fields(self, state: Dict[str, Any]) -> EvalResult:
        """Each evidence item should have the required fields: fact, source_url, source_title."""
        evidence = state.get("evidence", [])
        if not evidence:
            return EvalResult(
                name="evidence_fields",
                passed=True, score=1.0,
                details="no evidence to check (may be expected)",
            )

        required = {"fact", "source_url", "source_title"}
        valid = 0
        issues = []
        for i, e in enumerate(evidence):
            if not isinstance(e, dict):
                issues.append(f"evidence[{i}] is not a dict")
                continue
            missing = required - set(e.keys())
            if missing:
                issues.append(f"evidence[{i}] missing: {missing}")
            else:
                valid += 1

        ratio = valid / len(evidence)
        return EvalResult(
            name="evidence_fields",
            passed=ratio >= 0.9,
            score=round(ratio, 3),
            details=f"{valid}/{len(evidence)} valid" + (f"; issues: {issues[:3]}" if issues else ""),
        )

    def check_sources_have_urls(self, state: Dict[str, Any]) -> EvalResult:
        """Every source in the sources dict should have a url and title."""
        sources = state.get("sources", {})
        if not sources:
            return EvalResult(
                name="sources_have_urls",
                passed=False, score=0.0,
                details="no sources found",
            )

        valid = 0
        for url, src in sources.items():
            if isinstance(src, dict) and src.get("url") and src.get("title"):
                valid += 1

        ratio = valid / len(sources)
        return EvalResult(
            name="sources_have_urls",
            passed=ratio >= 0.9,
            score=round(ratio, 3),
            details=f"{valid}/{len(sources)} sources have url+title",
        )

    def check_no_empty_report(self, state: Dict[str, Any]) -> EvalResult:
        """Report field should exist and not be empty or just whitespace."""
        report = state.get("report", "")
        has_content = bool(report and report.strip())

        return EvalResult(
            name="no_empty_report",
            passed=has_content,
            score=1.0 if has_content else 0.0,
            details=f"report length: {len(report)} chars" if has_content else "report is empty",
        )
