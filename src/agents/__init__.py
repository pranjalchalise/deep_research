"""
Research Agents

Available agents:
- researcher: Simple ReAct-based research agent
- deep_researcher: Full-featured research with HITL, gap detection, and citations
"""

from src.agents.researcher import research, ResearchAgent
from src.agents.deep_researcher import deep_research, DeepResearchAgent

__all__ = [
    "research",
    "ResearchAgent",
    "deep_research",
    "DeepResearchAgent",
]
