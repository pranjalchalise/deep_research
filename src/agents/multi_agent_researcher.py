"""
Multi-agent research using an orchestrator-workers pattern for parallel execution.
Workers research sub-questions independently, then findings are synthesized into a report.
"""
from __future__ import annotations

import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import create_chat_model
from src.utils.json_utils import parse_json_object, parse_json_array
from src.tools.tavily import cached_search
from src.tools.http_fetch import fetch_and_chunk


ORCHESTRATE_PROMPT = """You are a research orchestrator. Break down this research query into independent sub-questions that can be researched in parallel.

QUERY: {query}
{clarification_section}

Rules for creating sub-questions:
1. Each sub-question should be INDEPENDENT (can be researched without the others)
2. Together they should FULLY COVER the original query
3. Create 3-5 sub-questions (not more)
4. Each should be specific and searchable

Respond with JSON:
{{
    "sub_questions": [
        {{
            "id": "Q1",
            "question": "Specific sub-question",
            "focus": "What aspect this covers",
            "search_queries": ["search 1", "search 2"]
        }}
    ],
    "reasoning": "Why this breakdown covers the query well"
}}
"""

WORKER_EXTRACT_PROMPT = """You are a research worker focused on ONE specific question.

YOUR ASSIGNED QUESTION:
{question}

CONTENT FROM {url}:
{content}

Extract facts ONLY relevant to your assigned question. Be focused.

Respond with JSON array:
[
    {{"fact": "specific fact", "quote": "supporting quote", "confidence": 0.9}}
]

Return [] if nothing relevant to YOUR question.
"""

WORKER_SUMMARY_PROMPT = """Summarize your research findings for this specific question.

YOUR QUESTION:
{question}

EVIDENCE YOU FOUND:
{evidence}

Respond with JSON:
{{
    "answer": "Direct answer to the question based on evidence",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "confidence": 0.0-1.0,
    "gaps": ["What's still missing or unclear"],
    "sources_used": {sources_count}
}}
"""

SYNTHESIZE_PROMPT = """You are a research synthesizer. Combine findings from multiple research workers.

ORIGINAL QUERY:
{query}
{clarification_section}

WORKER FINDINGS:
{worker_findings}

Your job:
1. Combine all findings into a coherent understanding
2. Resolve any conflicts between workers
3. Identify overall gaps
4. Assess overall confidence

Respond with JSON:
{{
    "combined_answer": "Synthesized answer to the original query",
    "key_themes": ["theme 1", "theme 2"],
    "conflicts": ["Any conflicting information found"],
    "overall_gaps": ["Gaps that remain after all workers"],
    "overall_confidence": 0.0-1.0,
    "needs_more_research": true/false,
    "additional_searches": ["search if needs_more_research is true"]
}}
"""

WRITE_MULTI_PROMPT = """Write a comprehensive research report synthesizing findings from multiple research workers.

QUERY: {query}
{clarification_section}

WORKER FINDINGS:
{worker_summaries}

ALL EVIDENCE:
{evidence}

SOURCES:
{sources}

Rules:
1. Organize by themes/topics, not by worker
2. Every claim needs a citation [1], [2], etc.
3. Note any conflicting information
4. Acknowledge gaps

Write a well-structured report.
"""


@dataclass
class WorkerTask:
    """A task assigned to a worker."""
    id: str
    question: str
    focus: str
    search_queries: List[str]


@dataclass
class WorkerResult:
    """Result from a worker's research."""
    worker_id: str
    question: str
    evidence: List[Dict]
    sources: Dict[str, Dict]
    answer: str
    key_findings: List[str]
    confidence: float
    gaps: List[str]
    time_taken: float


@dataclass
class Metrics:
    """Performance metrics for comparison."""
    total_time: float = 0.0
    orchestration_time: float = 0.0
    parallel_research_time: float = 0.0
    total_worker_time: float = 0.0
    synthesis_time: float = 0.0
    verification_time: float = 0.0
    writing_time: float = 0.0

    num_workers: int = 0
    total_searches: int = 0
    total_sources: int = 0
    total_evidence: int = 0

    coverage: float = 0.0
    confidence: float = 0.0

    sequential_estimate: float = 0.0
    speedup_factor: float = 1.0


class ResearchWorker:
    """Independent worker that researches a single sub-question."""

    def __init__(
        self,
        worker_id: str,
        task: WorkerTask,
        llm,
        fast_llm,
        max_sources: int = 5,
        use_cache: bool = True,
        cache_dir: str = ".cache_v10",
        verbose: bool = False,
    ):
        self.worker_id = worker_id
        self.task = task
        self.llm = llm
        self.fast_llm = fast_llm
        self.max_sources = max_sources
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.verbose = verbose

        self.evidence = []
        self.sources = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [{self.worker_id}] {msg}")

    def research(self) -> WorkerResult:
        """Execute research for this worker's assigned question."""
        start_time = time.time()

        self._log(f"Starting: {self.task.question[:50]}...")

        for query in self.task.search_queries[:3]:
            self._log(f"Searching: {query[:40]}...")

            results = cached_search(
                query=query,
                max_results=4,
                use_cache=self.use_cache,
                cache_dir=f"{self.cache_dir}/search"
            )

            if not results:
                continue

            for result in results[:2]:
                url = result["url"]
                title = result["title"]

                if url in self.sources or len(self.sources) >= self.max_sources:
                    continue

                self.sources[url] = {
                    "url": url,
                    "title": title,
                    "source_id": len(self.sources) + 1
                }

                chunks = fetch_and_chunk(
                    url=url,
                    chunk_chars=3000,
                    max_chunks=2,
                    timeout_s=8,
                    use_cache=self.use_cache,
                    cache_dir=f"{self.cache_dir}/pages"
                )

                if not chunks:
                    continue

                content = "\n".join(chunks[:2])

                response = self.fast_llm.invoke([
                    SystemMessage(content="Extract facts as JSON array."),
                    HumanMessage(content=WORKER_EXTRACT_PROMPT.format(
                        question=self.task.question,
                        url=url,
                        content=content[:5000]
                    ))
                ])

                facts = parse_json_array(response.content, default=[])

                for fact in facts[:4]:
                    if isinstance(fact, dict) and fact.get("fact"):
                        self.evidence.append({
                            "fact": fact["fact"],
                            "quote": fact.get("quote", ""),
                            "source_url": url,
                            "source_title": title,
                            "confidence": fact.get("confidence", 0.8),
                            "worker_id": self.worker_id
                        })

        evidence_text = "\n".join([
            f"- {e['fact']}" for e in self.evidence
        ]) if self.evidence else "No evidence found."

        response = self.llm.invoke([
            SystemMessage(content="Summarize research findings as JSON."),
            HumanMessage(content=WORKER_SUMMARY_PROMPT.format(
                question=self.task.question,
                evidence=evidence_text,
                sources_count=len(self.sources)
            ))
        ])

        summary = parse_json_object(response.content, default={
            "answer": "Insufficient evidence",
            "key_findings": [],
            "confidence": 0.0,
            "gaps": ["Unable to find relevant information"]
        })

        elapsed = time.time() - start_time
        self._log(f"Done in {elapsed:.1f}s - {len(self.evidence)} evidence, {len(self.sources)} sources")

        return WorkerResult(
            worker_id=self.worker_id,
            question=self.task.question,
            evidence=self.evidence,
            sources=self.sources,
            answer=summary.get("answer", ""),
            key_findings=summary.get("key_findings", []),
            confidence=summary.get("confidence", 0.0),
            gaps=summary.get("gaps", []),
            time_taken=elapsed
        )


class MultiAgentResearcher:
    """Orchestrator-workers research agent that parallelizes sub-questions for faster, deeper coverage."""

    def __init__(
        self,
        model: str = "gpt-4o",
        fast_model: str = "gpt-4o-mini",
        max_workers: int = 4,
        max_sources_per_worker: int = 5,
        max_iterations: int = 2,
        min_coverage: float = 0.7,
        verbose: bool = True,
        use_cache: bool = True,
        cache_dir: str = ".cache_v10",
        clarification_callback: Optional[Callable] = None,
    ):
        self.llm = create_chat_model(model=model, temperature=0.2)
        self.fast_llm = create_chat_model(model=fast_model, temperature=0.1)
        self.max_workers = max_workers
        self.max_sources_per_worker = max_sources_per_worker
        self.max_iterations = max_iterations
        self.min_coverage = min_coverage
        self.verbose = verbose
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.clarification_callback = clarification_callback or self._default_clarify

        self.metrics = Metrics()

    def _log(self, msg: str, prefix: str = ""):
        if self.verbose:
            if prefix:
                print(f"[{prefix}] {msg}")
            else:
                print(msg)

    def _default_clarify(self, question: str, options: List) -> str:
        print(f"\n{'='*60}")
        print("CLARIFICATION NEEDED")
        print('='*60)
        print(f"\n{question}\n")
        if options:
            print("Options:")
            for i, opt in enumerate(options, 1):
                if isinstance(opt, dict):
                    label = opt.get("label", str(opt))
                    desc = opt.get("description", "")
                    print(f"  {i}. {label}")
                    if desc:
                        print(f"     {desc}")
                else:
                    print(f"  {i}. {opt}")
            print(f"  {len(options)+1}. Other (type your own)")
        response = input("\nYour choice (number or text): ").strip()
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(options):
                opt = options[idx]
                if isinstance(opt, dict):
                    return opt.get("label", str(opt))
                return opt
        return response

    def _understand_and_clarify(self, query: str) -> tuple[str, str]:
        """Understand query and clarify if needed. Returns (query, user_clarification)."""
        from src.agents.deep_researcher import UNDERSTAND_PROMPT

        self._log("Analyzing query...", "UNDERSTAND")

        response = self.llm.invoke([
            SystemMessage(content="Analyze the query. Output valid JSON only."),
            HumanMessage(content=UNDERSTAND_PROMPT.format(query=query))
        ])

        result = parse_json_object(response.content, default={"is_clear": True})

        if not result.get("is_clear", True):
            ambiguity_type = result.get("ambiguity_type", "unknown")
            self._log(f"Query is ambiguous ({ambiguity_type})", "CLARIFY")
            user_response = self.clarification_callback(
                result.get("clarification_question", "Please clarify"),
                result.get("clarification_options", [])
            )
            return query, user_response

        return query, ""

    def _clarification_section(self, user_clarification: str) -> str:
        """Build clarification section for prompts."""
        if user_clarification:
            return f"\nUSER CLARIFICATION:\n{user_clarification}"
        return ""

    def _orchestrate(self, query: str, user_clarification: str = "") -> List[WorkerTask]:
        """Break query into sub-questions for parallel workers."""
        self._log("Breaking down query into sub-questions...", "ORCHESTRATE")

        start = time.time()

        response = self.llm.invoke([
            SystemMessage(content="Break down research query. Output JSON."),
            HumanMessage(content=ORCHESTRATE_PROMPT.format(
                query=query,
                clarification_section=self._clarification_section(user_clarification)
            ))
        ])

        result = parse_json_object(response.content, default={"sub_questions": []})

        tasks = []
        for sq in result.get("sub_questions", [])[:self.max_workers]:
            tasks.append(WorkerTask(
                id=sq.get("id", f"Q{len(tasks)+1}"),
                question=sq.get("question", query),
                focus=sq.get("focus", ""),
                search_queries=sq.get("search_queries", [query])
            ))

        self.metrics.orchestration_time = time.time() - start
        self.metrics.num_workers = len(tasks)

        self._log(f"Created {len(tasks)} sub-questions:", "ORCHESTRATE")
        for t in tasks:
            self._log(f"  {t.id}: {t.question[:60]}...", "")

        return tasks

    def _run_workers_parallel(self, tasks: List[WorkerTask]) -> List[WorkerResult]:
        """Run all workers in parallel."""
        self._log(f"Launching {len(tasks)} parallel workers...", "WORKERS")

        start = time.time()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {}
            for i, task in enumerate(tasks):
                worker = ResearchWorker(
                    worker_id=f"W{i+1}",
                    task=task,
                    llm=self.llm,
                    fast_llm=self.fast_llm,
                    max_sources=self.max_sources_per_worker,
                    use_cache=self.use_cache,
                    cache_dir=self.cache_dir,
                    verbose=self.verbose
                )
                future = executor.submit(worker.research)
                future_to_task[future] = task

            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self._log(f"Worker failed: {e}", "ERROR")

        self.metrics.parallel_research_time = time.time() - start
        self.metrics.total_worker_time = sum(r.time_taken for r in results)

        self._log(f"All workers complete in {self.metrics.parallel_research_time:.1f}s (wall clock)", "WORKERS")
        self._log(f"Total worker time: {self.metrics.total_worker_time:.1f}s (would be sequential)", "WORKERS")

        return results

    def _synthesize(self, query: str, user_clarification: str, worker_results: List[WorkerResult]) -> Dict[str, Any]:
        """Synthesize findings from all workers."""
        self._log("Synthesizing worker findings...", "SYNTHESIZE")

        start = time.time()

        findings_text = ""
        for r in worker_results:
            findings_text += f"\n### {r.worker_id}: {r.question}\n"
            findings_text += f"Answer: {r.answer}\n"
            findings_text += f"Key findings: {', '.join(r.key_findings)}\n"
            findings_text += f"Confidence: {r.confidence:.0%}\n"
            findings_text += f"Gaps: {', '.join(r.gaps)}\n"

        response = self.llm.invoke([
            SystemMessage(content="Synthesize research findings. Output JSON."),
            HumanMessage(content=SYNTHESIZE_PROMPT.format(
                query=query,
                clarification_section=self._clarification_section(user_clarification),
                worker_findings=findings_text
            ))
        ])

        result = parse_json_object(response.content, default={
            "combined_answer": "",
            "overall_confidence": 0.5,
            "needs_more_research": False
        })

        self.metrics.synthesis_time = time.time() - start
        self.metrics.coverage = result.get("overall_confidence", 0.5)

        self._log(f"Overall confidence: {result.get('overall_confidence', 0):.0%}", "SYNTHESIZE")

        return result

    def _write_report(
        self,
        query: str,
        user_clarification: str,
        worker_results: List[WorkerResult],
        synthesis: Dict[str, Any]
    ) -> str:
        """Write final report combining all worker findings."""
        self._log("Writing final report...", "WRITE")

        start = time.time()

        all_evidence = []
        all_sources = {}
        evidence_id = 1
        source_id = 1

        for r in worker_results:
            for e in r.evidence:
                e["evidence_id"] = evidence_id
                all_evidence.append(e)
                evidence_id += 1

            for url, s in r.sources.items():
                if url not in all_sources:
                    s["source_id"] = source_id
                    all_sources[url] = s
                    source_id += 1

        worker_summaries = ""
        for r in worker_results:
            worker_summaries += f"\n### {r.question}\n"
            worker_summaries += f"{r.answer}\n"
            worker_summaries += f"Findings: {', '.join(r.key_findings)}\n"

        evidence_text = "\n".join([
            f"[{e['evidence_id']}] {e['fact']} (from: {e['source_title']})"
            for e in all_evidence
        ])

        sources_text = "\n".join([
            f"[{s['source_id']}] {s['title']} - {s['url']}"
            for s in all_sources.values()
        ])

        response = self.llm.invoke([
            SystemMessage(content="Write research report with citations."),
            HumanMessage(content=WRITE_MULTI_PROMPT.format(
                query=query,
                clarification_section=self._clarification_section(user_clarification),
                worker_summaries=worker_summaries,
                evidence=evidence_text,
                sources=sources_text
            ))
        ])

        report = response.content

        if "## Sources" not in report:
            report += "\n\n---\n\n## Sources\n\n"
            for s in all_sources.values():
                report += f"[{s['source_id']}] {s['title']} - {s['url']}\n"

        self.metrics.writing_time = time.time() - start
        self.metrics.total_sources = len(all_sources)
        self.metrics.total_evidence = len(all_evidence)

        return report

    def research(self, query: str) -> Dict[str, Any]:
        """Run multi-agent research and return dict with report, metrics, and comparison data."""
        total_start = time.time()

        self._log(f"\n{'='*60}", "")
        self._log("MULTI-AGENT DEEP RESEARCH", "")
        self._log(f"{'='*60}", "")
        self._log(f"Query: {query}", "")
        self._log(f"Max workers: {self.max_workers}", "")
        self._log("-" * 60, "")

        original_query, user_clarification = self._understand_and_clarify(query)

        tasks = self._orchestrate(original_query, user_clarification)

        if not tasks:
            return {"report": "Failed to create research tasks", "metrics": self.metrics}

        worker_results = self._run_workers_parallel(tasks)

        synthesis = self._synthesize(original_query, user_clarification, worker_results)

        report = self._write_report(original_query, user_clarification, worker_results, synthesis)

        self.metrics.total_time = time.time() - total_start
        self.metrics.confidence = synthesis.get("overall_confidence", 0.5)
        self.metrics.total_searches = sum(
            len(t.search_queries) for t in tasks
        )

        self.metrics.sequential_estimate = (
            self.metrics.orchestration_time +
            self.metrics.total_worker_time +
            self.metrics.synthesis_time +
            self.metrics.writing_time
        )

        if self.metrics.total_time > 0:
            self.metrics.speedup_factor = self.metrics.sequential_estimate / self.metrics.total_time

        self._log(f"\n{'='*60}", "")
        self._log("RESEARCH COMPLETE", "")
        self._log(f"{'='*60}", "")

        return {
            "report": report,
            "query": original_query,
            "user_clarification": user_clarification,
            "clarified_query": f"{original_query} - Clarification: {user_clarification}" if user_clarification else original_query,
            "worker_results": [
                {
                    "worker_id": r.worker_id,
                    "question": r.question,
                    "answer": r.answer,
                    "confidence": r.confidence,
                    "evidence_count": len(r.evidence),
                    "sources_count": len(r.sources),
                    "time_taken": r.time_taken
                }
                for r in worker_results
            ],
            "synthesis": synthesis,
            "metrics": self._format_metrics()
        }

    def _format_metrics(self) -> Dict[str, Any]:
        """Format metrics for output."""
        return {
            "total_time": round(self.metrics.total_time, 2),
            "orchestration_time": round(self.metrics.orchestration_time, 2),
            "parallel_research_time": round(self.metrics.parallel_research_time, 2),
            "total_worker_time": round(self.metrics.total_worker_time, 2),
            "synthesis_time": round(self.metrics.synthesis_time, 2),
            "writing_time": round(self.metrics.writing_time, 2),
            "num_workers": self.metrics.num_workers,
            "total_searches": self.metrics.total_searches,
            "total_sources": self.metrics.total_sources,
            "total_evidence": self.metrics.total_evidence,
            "coverage": round(self.metrics.coverage, 2),
            "confidence": round(self.metrics.confidence, 2),
            "sequential_estimate": round(self.metrics.sequential_estimate, 2),
            "speedup_factor": round(self.metrics.speedup_factor, 2),
            "time_saved_seconds": round(self.metrics.sequential_estimate - self.metrics.total_time, 2),
            "time_saved_percent": round((1 - self.metrics.total_time / self.metrics.sequential_estimate) * 100, 1) if self.metrics.sequential_estimate > 0 else 0
        }


def multi_agent_research(query: str, **kwargs) -> Dict[str, Any]:
    """Run multi-agent research on a query."""
    agent = MultiAgentResearcher(**kwargs)
    return agent.research(query)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multi_agent_researcher.py 'query'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = multi_agent_research(query)

    print("\n" + "="*60)
    print("REPORT")
    print("="*60)
    print(result["report"])

    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    m = result["metrics"]
    print(f"  Total Time:         {m['total_time']}s")
    print(f"  Workers:            {m['num_workers']}")
    print(f"  Parallel Time:      {m['parallel_research_time']}s (wall clock)")
    print(f"  Sequential Estimate:{m['sequential_estimate']}s")
    print(f"  Speedup Factor:     {m['speedup_factor']}x")
    print(f"  Time Saved:         {m['time_saved_seconds']}s ({m['time_saved_percent']}%)")
    print(f"  Sources:            {m['total_sources']}")
    print(f"  Evidence:           {m['total_evidence']}")
    print(f"  Confidence:         {m['confidence']:.0%}")
