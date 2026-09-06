import os
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)


from agents.synthesis_agent import SynthesisAgent


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("OMNIMIND MEMORY-AWARE SYNTHESIS TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Test State
    # --------------------------------------------------------

    state = {
        "query": "What did I decide about my project 3 months ago?",

        "route": "rag",

        # ----------------------------------------------------
        # Analysis
        # ----------------------------------------------------

        "analysis": (
            "Three months ago, the historical decision was "
            "to use Qdrant as the vector database for the "
            "OmniMind project. A later current decision "
            "changed the database to PostgreSQL."
        ),

        # ----------------------------------------------------
        # Semantic Long-Term Memory
        # ----------------------------------------------------

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
                    "created_at": (
                        "2026-06-06T10:00:00"
                    ),
                },
            }
        ],

        "memory_context": (
            "1. Decision: I decided to use Qdrant as the "
            "vector database for my OmniMind project."
        ),

        # ----------------------------------------------------
        # Temporal / Consolidated Memory
        # ----------------------------------------------------

        "temporal_memory_results": [
            {
                "topic": "OmniMind Vector Database",

                "summary": (
                    "The project initially selected Qdrant. "
                    "The current decision is PostgreSQL."
                ),

                "current_memory_id": (
                    "postgres-decision"
                ),

                "historical_memory_ids": [
                    "qdrant-decision"
                ],

                "timeline": [
                    {
                        "memory_id": (
                            "qdrant-decision"
                        ),
                        "timestamp": (
                            "2026-06-06T10:00:00"
                        ),
                        "text": (
                            "Qdrant was selected as the "
                            "vector database."
                        ),
                        "status": "historical",
                    },
                    {
                        "memory_id": (
                            "postgres-decision"
                        ),
                        "timestamp": (
                            "2026-09-05T10:00:00"
                        ),
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

        # ----------------------------------------------------
        # RAG Evidence
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # No Web Evidence
        # ----------------------------------------------------

        "research_results": [],

        # ----------------------------------------------------
        # Planning Memory
        # ----------------------------------------------------

        "planning_memory_context": (
            "Relevant temporal memory and historical "
            "context should guide the response."
        ),

        "research_memory_context": "",
    }

    # --------------------------------------------------------
    # Run Synthesis
    # --------------------------------------------------------

    print("\nRunning Memory-Aware Synthesis...")

    agent = SynthesisAgent()

    try:
        result = agent.run(state)

    finally:
        close = getattr(
            agent,
            "close",
            None,
        )

        if callable(close):
            close()

    final_answer = result.get(
        "final_answer",
        "",
    )

    # --------------------------------------------------------
    # Display Final Answer
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(final_answer)

    # --------------------------------------------------------
    # Normalize LLM Output
    # --------------------------------------------------------

    normalized_answer = (
        final_answer
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2009", " ")
        .replace("\u2007", " ")
        .replace("\u2011", "-")
    )

    normalized_answer = " ".join(
        normalized_answer.split()
    )

    answer_lower = normalized_answer.lower()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = {

        # ----------------------------------------------------
        # Basic generation
        # ----------------------------------------------------

        "Final answer generated": bool(
            final_answer.strip()
        ),

        # ----------------------------------------------------
        # Historical decision
        # ----------------------------------------------------

        "Historical decision Qdrant preserved": (
            "qdrant" in answer_lower
        ),

        # ----------------------------------------------------
        # Historical time context
        #
        # Accept:
        #   3 months
        #   three months
        #   historical
        #   past
        # ----------------------------------------------------

        "Historical context recognized": (
            (
                "3 months" in answer_lower
                or "3 month" in answer_lower
                or "three months" in answer_lower
                or "three month" in answer_lower
            )
            or "histor" in answer_lower
            or "past" in answer_lower
        ),

        # ----------------------------------------------------
        # Critical temporal requirement
        #
        # If PostgreSQL appears, Qdrant must ALSO appear.
        #
        # PostgreSQL is not required because the question
        # asks specifically about the historical decision.
        # ----------------------------------------------------

        "Historical decision remains primary": (
            "qdrant" in answer_lower
            and (
                "postgresql" not in answer_lower
                or "qdrant" in answer_lower
            )
        ),

        # ----------------------------------------------------
        # No contradiction
        #
        # The answer must not say that PostgreSQL was the
        # decision 3 months ago.
        # ----------------------------------------------------

        "Historical decision not replaced by PostgreSQL": (
            not (
                "3 months" in answer_lower
                and "postgresql" in answer_lower
                and "qdrant" not in answer_lower
            )
        ),

        # ----------------------------------------------------
        # Internal implementation protection
        # ----------------------------------------------------

        "No internal orchestration mentioned": not any(
            term in answer_lower
            for term in [
                "planner agent",
                "rag agent",
                "research agent",
                "analysis agent",
                "synthesis agent",
                "langgraph",
                "mcp",
            ]
        ),
    }

    # --------------------------------------------------------
    # Print validation results
    # --------------------------------------------------------

    passed = True

    for name, condition in checks.items():

        if condition:
            print(f"[PASS] {name}")

        else:
            print(f"[FAIL] {name}")
            passed = False

    # --------------------------------------------------------
    # Final Result
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    if passed:

        print(
            "MEMORY-AWARE SYNTHESIS TEST PASSED"
        )

    else:

        print(
            "MEMORY-AWARE SYNTHESIS TEST FAILED"
        )

        raise SystemExit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()