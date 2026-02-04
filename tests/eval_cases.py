"""
Test cases for evaluating the research pipeline.

Each case has a query, expected properties, and optional ground-truth
snippets for correctness checking. The idea is to cover a range of
query types so evals aren't just testing one happy path.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalCase:
    """One evaluation scenario."""
    name: str
    query: str
    mode: str  # "single" or "multi"
    # what we expect to see in the output
    expect_ambiguous: bool = False
    min_sources: int = 2
    min_evidence: int = 3
    # keywords that should appear somewhere in the report
    expected_keywords: List[str] = field(default_factory=list)
    # optional ground truth for correctness eval
    reference_answer: Optional[str] = None
    # expected report structure hints
    expect_sections: int = 3  # minimum ## headings


# -- Factual / straightforward ------------------------------------------------

FACTUAL_SIMPLE = EvalCase(
    name="factual_simple",
    query="What is CRISPR-Cas9 and how does it work?",
    mode="single",
    min_sources=3,
    min_evidence=5,
    expected_keywords=["CRISPR", "Cas9", "gene", "DNA", "edit"],
    expect_sections=3,
    reference_answer=(
        "CRISPR-Cas9 is a genome editing tool adapted from a bacterial immune "
        "system. It uses a guide RNA to direct the Cas9 enzyme to a specific "
        "DNA sequence, where it creates a double-strand break. The cell's "
        "repair machinery then fixes the break, allowing researchers to delete, "
        "insert, or modify genes."
    ),
)

FACTUAL_PERSON = EvalCase(
    name="factual_person",
    query="Who is Geoffrey Hinton and what are his major contributions to AI?",
    mode="single",
    min_sources=3,
    min_evidence=4,
    expected_keywords=["Hinton", "neural", "deep learning", "backpropagation"],
    expect_sections=3,
)

# -- Broad / multi-dimensional ------------------------------------------------

BROAD_COMPARISON = EvalCase(
    name="broad_comparison",
    query="Compare renewable energy sources: solar, wind, and nuclear power in terms of cost, scalability, and environmental impact.",
    mode="multi",
    min_sources=4,
    min_evidence=8,
    expected_keywords=["solar", "wind", "nuclear", "cost", "scalable"],
    expect_sections=4,
)

BROAD_TOPIC = EvalCase(
    name="broad_topic",
    query="What are the current applications and limitations of large language models in healthcare?",
    mode="multi",
    min_sources=4,
    min_evidence=6,
    expected_keywords=["LLM", "healthcare", "clinical", "limitation"],
    expect_sections=3,
)

# -- Ambiguous (should trigger clarification) ----------------------------------

AMBIGUOUS_POLYSEMY = EvalCase(
    name="ambiguous_polysemy",
    query="Tell me about Mercury.",
    mode="single",
    expect_ambiguous=True,
    min_sources=2,
    min_evidence=3,
    expected_keywords=["Mercury"],
)

AMBIGUOUS_SCOPE = EvalCase(
    name="ambiguous_scope",
    query="Explain the impact of AI.",
    mode="single",
    expect_ambiguous=True,
    min_sources=2,
    min_evidence=3,
    expected_keywords=["AI", "impact"],
)

# -- Time-sensitive ------------------------------------------------------------

TIME_SENSITIVE = EvalCase(
    name="time_sensitive",
    query="What are the latest developments in quantum computing as of 2024?",
    mode="multi",
    min_sources=3,
    min_evidence=5,
    expected_keywords=["quantum", "2024"],
    expect_sections=3,
)

# -- Technical / how-to -------------------------------------------------------

TECHNICAL_HOWTO = EvalCase(
    name="technical_howto",
    query="How do you implement a RAG pipeline using LangChain and a vector database?",
    mode="single",
    min_sources=3,
    min_evidence=4,
    expected_keywords=["RAG", "retrieval", "vector", "embedding", "LangChain"],
    expect_sections=3,
)


# All cases grouped for easy iteration
ALL_CASES = [
    FACTUAL_SIMPLE,
    FACTUAL_PERSON,
    BROAD_COMPARISON,
    BROAD_TOPIC,
    AMBIGUOUS_POLYSEMY,
    AMBIGUOUS_SCOPE,
    TIME_SENSITIVE,
    TECHNICAL_HOWTO,
]

# Subset for quick smoke tests (cheaper)
QUICK_CASES = [FACTUAL_SIMPLE, BROAD_COMPARISON]

# Only the ambiguous ones
AMBIGUOUS_CASES = [AMBIGUOUS_POLYSEMY, AMBIGUOUS_SCOPE]
