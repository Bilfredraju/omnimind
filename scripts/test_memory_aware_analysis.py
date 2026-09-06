import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from agents.analysis_agent import AnalysisAgent


def main():
    print("=" * 60)
    print("OMNIMIND MEMORY-AWARE ANALYSIS TEST")
    print("=" * 60)

    state = {
        "query": "What did I decide about my project 3 months ago?",

        "memory_results": [
            {
                "text": (
                    "I decided to use Qdrant as the vector "
                    "database for my OmniMind project."
                ),
                "score": 0.92,
                "metadata": {
                    "type": "decision",
                    "importance": 1.0,
                },
            }
        ],

        "memory_context": (
            "1. Decision: I decided to use Qdrant as the "
            "vector database for my OmniMind project."
        ),

        "temporal_memory_results": [
            {
                "topic": "OmniMind Vector Database",
                "summary": (
                    "The project initially selected Qdrant. "
                    "The current decision is PostgreSQL."
                ),
                "current_memory_id": "postgres-decision",
                "historical_memory_ids": [
                    "qdrant-decision"
                ],
                "timeline": [
                    {
                        "memory_id": "qdrant-decision",
                        "text": (
                            "Qdrant was selected as the "
                            "vector database."
                        ),
                        "status": "historical",
                    },
                    {
                        "memory_id": "postgres-decision",
                        "text": (
                            "PostgreSQL is the current "
                            "database decision."
                        ),
                        "status": "current",
                    },
                ],
                "score": 0.88,
            }
        ],

        "temporal_memory_context": (
            "Temporal Memory:\n"
            "Topic: OmniMind Vector Database\n"
            "Historical decision: Qdrant was selected "
            "as the vector database.\n"
            "Current decision: PostgreSQL."
        ),

        "temporal_intent": {
            "expression": "3 months ago",
            "has_time_filter": True,
            "is_current": False,
        },

        "rag_results": [
            {
                "source": "sample.pdf",
                "page": 2,
                "chunk": 1,
                "score": 0.82,
                "text": (
                    "The RAG architecture combines "
                    "retrieval with generation."
                ),
            }
        ],

        "research_results": [],

        "planning_memory_context": (
            "Relevant temporal memory and historical "
            "context should guide the response."
        ),

        "research_memory_context": "",
    }

    print("\nRunning Memory-Aware Analysis...")

    agent = AnalysisAgent()

    try:
        result = agent.run(state)
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()

    analysis = result.get("analysis", "")

    print("\n" + "=" * 60)
    print("ANALYSIS OUTPUT")
    print("=" * 60)
    print(analysis)

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = {
        "Analysis generated": bool(analysis.strip()),
        "Qdrant mentioned": "qdrant" in analysis.lower(),
        "Historical context considered": (
            "histor" in analysis.lower()
            or "3 months" in analysis.lower()
            or "past" in analysis.lower()
        ),
        "Temporal context considered": (
            "temporal" in analysis.lower()
            or "current" in analysis.lower()
        ),
    }

    passed = True

    for name, condition in checks.items():
        if condition:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")
            passed = False

    print("\n" + "=" * 60)

    if passed:
        print("MEMORY-AWARE ANALYSIS TEST PASSED")
    else:
        print("MEMORY-AWARE ANALYSIS TEST FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()