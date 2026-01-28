from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.core.graph import build_graph


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    if len(sys.argv) < 2:
        print('Usage: python -m src.cli "your research question"')
        sys.exit(1)

    q = sys.argv[1]
    graph = build_graph()

    result = graph.invoke({
        "messages": [HumanMessage(content=q)],
        "depth": 8,
        "max_results": 5,
    })

    print("\n" + "=" * 90)
    print(result.get("report", ""))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
