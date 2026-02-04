"""
Research Agents

Available agents:
- deep_researcher: Full-featured research with HITL, gap detection, and citations
- multi_agent_researcher: Anthropic-style orchestrator-workers (parallel)
"""

from src.agents.deep_researcher import deep_research, DeepResearchAgent
from src.agents.multi_agent_researcher import multi_agent_research, MultiAgentResearcher

__all__ = [
    "deep_research",
    "DeepResearchAgent",
    "multi_agent_research",
    "MultiAgentResearcher",
]
