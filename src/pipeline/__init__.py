"""
Research pipeline -- the main entry point.

Exposes the graph builder and the state schema so callers
only need one import line to get started.
"""
from src.pipeline.state import ResearchState, Configuration
from src.pipeline.graph import build_graph

__all__ = ["ResearchState", "Configuration", "build_graph"]
