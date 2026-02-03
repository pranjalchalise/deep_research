#!/usr/bin/env python3
"""
Test script for V9 Research Studio

Tests the new LLM-driven query analyzer to verify it correctly
classifies queries without using hardcoded patterns.

Run: python test_v9.py
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on env vars

from langchain_core.messages import HumanMessage


def test_query_analyzer():
    """Test the query analyzer with various query types."""
    from src.nodes.analyzer import query_analyzer_node

    test_cases = [
        # The bug case: should be current_events, NOT concept
        {
            "query": "Deep research about Trump's new immigrant related policies",
            "expected_class": "current_events",
            "expected_not": "concept",
            "description": "Policy research - should NOT be classified as concept"
        },
        # Person profile
        {
            "query": "Tell me about Elon Musk",
            "expected_class": "person_profile",
            "description": "Person biography request"
        },
        # Person + topic
        {
            "query": "What does Elon Musk think about AI safety?",
            "expected_class": "person_opinions",
            "description": "Person's opinions on a topic"
        },
        # Technical query
        {
            "query": "How does React useState work internally?",
            "expected_class": "technical_docs",
            "description": "Technical implementation question"
        },
        # Comparison
        {
            "query": "Compare PostgreSQL vs MongoDB for web applications",
            "expected_class": "comparative_analysis",
            "description": "Comparison query"
        },
        # Ambiguous person
        {
            "query": "John Smith professor",
            "expected_clarification": True,
            "description": "Ambiguous person - should need clarification"
        },
        # Current events
        {
            "query": "What is happening with the AI regulation in EU 2025?",
            "expected_class": "current_events",
            "description": "Recent events query"
        },
        # Concept explanation
        {
            "query": "What is quantum computing?",
            "expected_class": "concept_explanation",
            "description": "Concept explanation"
        },
    ]

    print("=" * 70)
    print("V9 QUERY ANALYZER TEST")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {tc['description']} ---")
        print(f"Query: {tc['query']}")

        # Create mock state
        state = {
            "messages": [HumanMessage(content=tc["query"])],
        }

        # Run analyzer
        try:
            result = query_analyzer_node(state)
            analysis = result.get("query_analysis", {})

            query_class = analysis.get("query_class", "unknown")
            needs_clarification = analysis.get("needs_clarification", False)
            intent = analysis.get("intent", "")
            confidence = analysis.get("analysis_confidence", 0)

            print(f"Result: class={query_class}, clarification={needs_clarification}, confidence={confidence:.2f}")
            print(f"Intent: {intent[:100]}...")

            # Check expectations
            test_passed = True

            if "expected_class" in tc:
                if query_class != tc["expected_class"]:
                    print(f"  ❌ FAIL: Expected class '{tc['expected_class']}', got '{query_class}'")
                    test_passed = False
                else:
                    print(f"  ✓ Class matches expected")

            if "expected_not" in tc:
                if query_class == tc["expected_not"]:
                    print(f"  ❌ FAIL: Should NOT be '{tc['expected_not']}'")
                    test_passed = False
                else:
                    print(f"  ✓ Correctly NOT classified as '{tc['expected_not']}'")

            if "expected_clarification" in tc:
                if needs_clarification != tc["expected_clarification"]:
                    print(f"  ❌ FAIL: Expected clarification={tc['expected_clarification']}, got {needs_clarification}")
                    test_passed = False
                else:
                    print(f"  ✓ Clarification need matches expected")

            if test_passed:
                passed += 1
                print("  ✓ PASSED")
            else:
                failed += 1
                print("  ❌ FAILED")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)

    return failed == 0


def test_planner():
    """Test the V9 planner with the Trump query."""
    from src.nodes.analyzer import query_analyzer_node
    from src.nodes.planner_v9 import planner_node

    print("\n" + "=" * 70)
    print("V9 PLANNER TEST - Trump Immigration Query")
    print("=" * 70)

    query = "Deep research about Trump's new immigrant related policies"

    # Step 1: Analyze
    state = {
        "messages": [HumanMessage(content=query)],
    }

    print("\n--- Step 1: Query Analysis ---")
    result = query_analyzer_node(state)
    state.update(result)

    analysis = state.get("query_analysis", {})
    print(f"Class: {analysis.get('query_class')}")
    print(f"Intent: {analysis.get('intent')}")
    print(f"Subject: {analysis.get('primary_subject')}")

    # Step 2: Plan
    print("\n--- Step 2: Research Planning ---")
    plan_result = planner_node(state)
    state.update(plan_result)

    plan = state.get("plan", {})
    print(f"Topic: {plan.get('topic')}")
    print(f"Outline: {plan.get('outline')}")

    print("\nGenerated Queries:")
    for q in plan.get("queries", [])[:5]:
        print(f"  - [{q.get('section')}] {q.get('query')}")

    # Verify queries are relevant (not generic "What is Trump?")
    queries_text = " ".join(q.get("query", "") for q in plan.get("queries", []))

    bad_patterns = ["what is trump", "definition of trump", "trump explained"]
    good_patterns = ["immigration", "policy", "border", "executive order", "deportation"]

    has_bad = any(p in queries_text.lower() for p in bad_patterns)
    has_good = any(p in queries_text.lower() for p in good_patterns)

    print("\n--- Verification ---")
    if has_bad:
        print("❌ FAIL: Found generic/wrong queries (e.g., 'What is Trump?')")
    else:
        print("✓ No generic/wrong queries found")

    if has_good:
        print("✓ Found relevant policy-related queries")
    else:
        print("❌ FAIL: Missing policy-related queries")

    return not has_bad and has_good


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RESEARCH STUDIO V9 - TEST SUITE")
    print("=" * 70)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set. Tests will fail.")
        print("Set it with: export OPENAI_API_KEY=your-key")
        return

    all_passed = True

    # Test 1: Query Analyzer
    if not test_query_analyzer():
        all_passed = False

    # Test 2: Planner
    if not test_planner():
        all_passed = False

    # Final result
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
