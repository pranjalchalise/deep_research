"""
Shared fixtures for the eval suite.

Provides helpers to run the pipeline and capture the full output state,
plus factories that create realistic mock states for fast offline testing.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.pipeline.graph import build_graph
from src.pipeline.state import ResearchState


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline runner — actually invokes the graph
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    mode: str = "single",
    config_overrides: Optional[Dict[str, Any]] = None,
    max_iterations: int = 2,
    interrupt_on_clarify: bool = False,
) -> Dict[str, Any]:
    """Run the research pipeline end-to-end and return the final state dict.

    This calls real APIs (Tavily, OpenAI) so use sparingly or with caching.
    Set interrupt_on_clarify=False for automated runs (no HITL).
    """
    graph = build_graph(checkpointer=None, interrupt_on_clarify=interrupt_on_clarify)

    initial_state = {
        "query": query,
        "mode": mode,
        "iteration": 0,
        "max_iterations": max_iterations,
        "min_coverage": 0.7,
        "evidence": [],
        "worker_results": [],
        "done_workers": 0,
        "searches_done": [],
        "sources": {},
    }

    configurable = {
        "model": "gpt-4o",
        "fast_model": "gpt-4o-mini",
        "max_search_results": 5,
        "report_structure": "detailed",
        "system_prompt": "",
    }
    if config_overrides:
        configurable.update(config_overrides)

    t0 = time.time()
    result = graph.invoke(initial_state, config={"configurable": configurable})
    elapsed = time.time() - t0

    # tack on timing so evals can check it
    if "metadata" in result and isinstance(result["metadata"], dict):
        result["metadata"]["wall_time_s"] = round(elapsed, 2)

    return result


def save_snapshot(state: Dict[str, Any], name: str):
    """Persist a pipeline output state to disk for offline eval."""
    path = SNAPSHOTS_DIR / f"{name}.json"
    # messages aren't JSON-serializable, strip them
    serializable = {k: v for k, v in state.items() if k != "messages"}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    return path


def load_snapshot(name: str) -> Dict[str, Any]:
    """Load a previously saved state snapshot."""
    path = SNAPSHOTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No snapshot at {path}. Run with --live first.")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Mock state factory — for fast offline structural testing
# ---------------------------------------------------------------------------

def make_mock_state(
    query: str = "What is quantum computing?",
    mode: str = "single",
    num_sources: int = 5,
    num_evidence: int = 10,
    coverage: float = 0.85,
    confidence: float = 0.78,
    report: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a realistic-looking state dict without calling any APIs.

    Useful for testing structural evaluators quickly.
    """
    sources = {}
    evidence = []
    for i in range(num_sources):
        url = f"https://example.com/source-{i+1}"
        sources[url] = {
            "url": url,
            "title": f"Source {i+1}: Research on Topic",
            "source_id": i + 1,
            "evidence_count": num_evidence // num_sources,
        }

    for i in range(num_evidence):
        src_idx = i % num_sources
        url = f"https://example.com/source-{src_idx+1}"
        evidence.append({
            "fact": f"Finding {i+1} about quantum computing applications.",
            "quote": f"According to the research, finding {i+1} is significant.",
            "source_url": url,
            "source_title": f"Source {src_idx+1}: Research on Topic",
            "confidence": 0.75 + (i % 3) * 0.05,
        })

    verified_claims = []
    for i, e in enumerate(evidence[:8]):
        verified_claims.append({
            "claim": e["fact"],
            "supported": i % 5 != 0,  # ~80% supported
            "supporting_evidence": [e["source_url"]],
            "confidence": e["confidence"],
            "notes": "",
        })

    if report is None:
        # build a minimal but structurally valid report
        source_refs = "\n".join(
            f"[{s['source_id']}] {s['title']} - {s['url']}"
            for s in sources.values()
        )
        report = f"""# Research Report: Quantum Computing

## Overview

Quantum computing leverages quantum mechanics to process information in fundamentally
new ways [1]. Unlike classical computers that use bits representing 0 or 1, quantum
computers use qubits that can exist in superposition of both states simultaneously [2].
This represents a paradigm shift in computational capability with far-reaching
implications across science, industry, and national security.

## Key Concepts

Qubits can exist in superposition states, enabling a form of parallel computation
that classical machines cannot replicate [1][3]. Quantum entanglement creates
correlations between qubits that persist regardless of physical distance, allowing
measurements on one qubit to instantly affect another [2]. Quantum gates manipulate
qubits through unitary transformations, forming the basis of quantum circuits [3].
Error correction remains one of the biggest challenges because qubits are extremely
sensitive to environmental noise, a problem known as decoherence [4].

## Current Applications

Quantum computers are being actively explored for drug discovery, where they can
simulate molecular interactions at the quantum level [3]. Cryptography is another
major application area, since quantum algorithms like Shor's algorithm can
theoretically break widely used encryption schemes [5]. Optimization problems in
logistics, finance, and materials science also stand to benefit [1][4]. Several
companies including IBM, Google, and Quantinuum have demonstrated quantum advantage
on narrowly defined computational tasks [2][5].

## Limitations and Challenges

Current quantum computers are noisy intermediate-scale quantum (NISQ) devices with
limited qubit counts, typically ranging from a few dozen to a few thousand [3].
Decoherence—the loss of quantum information to the environment—remains a major
obstacle to building large-scale fault-tolerant systems [4][5]. The overhead
required for quantum error correction means that practical quantum advantage on
real-world problems is likely still years away [1][4].

## Sources

{source_refs}
"""

    return {
        "query": query,
        "mode": mode,
        "understanding": {"summary": query, "is_clear": True},
        "is_ambiguous": False,
        "research_questions": [
            "What is quantum computing?",
            "How do qubits work?",
            "What are current applications?",
        ],
        "aspects_to_cover": ["theory", "hardware", "applications", "limitations"],
        "iteration": 2,
        "coverage": coverage,
        "ready_to_write": True,
        "gaps": [],
        "evidence": evidence,
        "sources": sources,
        "searches_done": ["quantum computing", "quantum computing applications"],
        "verified_claims": verified_claims,
        "confidence": confidence,
        "report": report,
        "metadata": {
            "iterations": 2,
            "coverage": coverage,
            "confidence": confidence,
            "sources_count": num_sources,
            "evidence_count": num_evidence,
            "mode": mode,
        },
    }


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_state():
    """A realistic mock state for structural tests."""
    return make_mock_state()


@pytest.fixture
def mock_state_multi():
    """Mock state simulating multi-agent output."""
    state = make_mock_state(mode="multi", num_sources=8, num_evidence=20)
    state["worker_results"] = [
        {
            "question": "What are the theoretical foundations?",
            "answer": "Quantum computing is based on quantum mechanics...",
            "key_findings": ["superposition", "entanglement"],
            "confidence": 0.8,
            "gaps": [],
            "evidence_count": 7,
            "sources": {},
        },
        {
            "question": "What are practical applications?",
            "answer": "Applications include drug discovery and cryptography...",
            "key_findings": ["drug discovery", "cryptography", "optimization"],
            "confidence": 0.75,
            "gaps": ["limited real-world benchmarks"],
            "evidence_count": 6,
            "sources": {},
        },
    ]
    state["synthesis"] = {
        "combined_answer": "Quantum computing combines theoretical QM with practical applications...",
        "overall_confidence": 0.78,
        "needs_more_research": False,
    }
    return state


@pytest.fixture
def empty_evidence_state():
    """State where the pipeline found nothing — edge case."""
    return make_mock_state(
        num_sources=0,
        num_evidence=0,
        coverage=0.0,
        confidence=0.0,
        report="# Research Report\n\nInsufficient information found for: What is quantum computing?\n",
    )
