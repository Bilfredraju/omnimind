import os
import sys


# ============================================================
# PROJECT ROOT
# ============================================================
#
# When this file is executed as:
#
#     python scripts\test_memory_augmented_planning.py
#
# Python starts with "scripts" on sys.path. Add the OmniMind
# project root so imports such as "agents.planner" work.
#

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from agents.planner import PlannerAgent
from agents.state import AgentState


def print_separator():
    print("=" * 70)


def main():
    print_separator()
    print("OMNIMIND MEMORY-AUGMENTED PLANNING TEST")
    print_separator()

    # --------------------------------------------------
    # Simulated remembered context
    # --------------------------------------------------
    #
    # In the real OmniMind graph this information comes from:
    #
    # MemoryAgent
    #      ↓
    # memory_recall_node
    #      ↓
    # AgentState
    #      ↓
    # PlannerAgent
    #
    # We simulate the retrieved state here so the planner
    # can be tested independently.
    # --------------------------------------------------

    memory_results = [
        {
            "memory_id": "test-qdrant-decision",
            "text": (
                "I decided to use Qdrant as the vector database "
                "for my OmniMind project."
            ),
            "metadata": {
                "type": "decision",
                "importance": 1.0,
                "source": "conversation",
            },
            "score": 0.91,
        }
    ]

    memory_context = (
        "Memory 1:\n"
        "I decided to use Qdrant as the vector database "
        "for my OmniMind project."
    )

    # --------------------------------------------------
    # Simulated temporal memory
    # --------------------------------------------------

    temporal_memory_results = [
        {
            "memory_id": "test-qdrant-decision",
            "text": (
                "I decided to use Qdrant as the vector database "
                "for my OmniMind project."
            ),
            "metadata": {
                "type": "decision",
                "importance": 1.0,
                "status": "historical",
            },
            "score": 0.88,
        }
    ]

    temporal_memory_context = (
        "Temporal Memory:\n"
        "Topic: OmniMind Vector Database\n"
        "Historical decision: Qdrant was selected as the "
        "vector database."
    )

    temporal_intent = {
        "has_time_filter": False,
        "is_current": False,
        "expression": None,
    }

    # --------------------------------------------------
    # Build AgentState
    # --------------------------------------------------

    state: AgentState = {
        "query": (
            "How should I implement vector retrieval "
            "for my OmniMind project?"
        ),

        "memory_results": memory_results,

        "memory_context": memory_context,

        "temporal_memory_results": (
            temporal_memory_results
        ),

        "temporal_memory_context": (
            temporal_memory_context
        ),

        "temporal_intent": temporal_intent,
    }

    # --------------------------------------------------
    # Initialize planner
    # --------------------------------------------------

    print()
    print("Initializing PlannerAgent...")

    planner = PlannerAgent()

    print("PlannerAgent initialized successfully.")

    # --------------------------------------------------
    # Execute planning
    # --------------------------------------------------

    result = planner.plan(state)

    # --------------------------------------------------
    # QUERY
    # --------------------------------------------------

    print()
    print_separator()
    print("QUERY")
    print_separator()

    print(
        result.get(
            "query",
            state["query"],
        )
    )

    # --------------------------------------------------
    # SELECTED ROUTE
    # --------------------------------------------------

    print()
    print_separator()
    print("SELECTED ROUTE")
    print_separator()

    print(
        result.get(
            "route",
            "unknown",
        )
    )

    # --------------------------------------------------
    # MEMORY RETRIEVED
    # --------------------------------------------------

    print()
    print_separator()
    print("MEMORY RETRIEVED")
    print_separator()

    print(
        f"Semantic memories: "
        f"{len(result.get('memory_results', []))}"
    )

    print(
        f"Temporal memories: "
        f"{len(result.get('temporal_memory_results', []))}"
    )

    # --------------------------------------------------
    # MEMORY-AWARE PLAN
    # --------------------------------------------------

    print()
    print_separator()
    print("MEMORY-AWARE PLAN")
    print_separator()

    plan = result.get(
        "plan",
        [],
    )

    for index, step in enumerate(
        plan,
        start=1,
    ):
        print(f"{index}. {step}")

    # --------------------------------------------------
    # PLANNING MEMORY CONTEXT
    # --------------------------------------------------

    print()
    print_separator()
    print("PLANNING MEMORY CONTEXT")
    print_separator()

    planning_memory_context = result.get(
        "planning_memory_context",
        "",
    )

    print(
        planning_memory_context
    )

    # --------------------------------------------------
    # TEST ASSERTIONS
    # --------------------------------------------------

    print()
    print_separator()
    print("VALIDATING MEMORY-AUGMENTED PLANNING")
    print_separator()

    # Route should remain deterministic.
    assert result.get("route") == "rag", (
        "Expected RAG route for a normal project question."
    )

    print("[PASS] Deterministic RAG route selected.")

    # Planner should produce a plan.
    assert plan, (
        "Planner returned an empty plan."
    )

    print("[PASS] Planner generated an execution plan.")

    # Plan should explicitly acknowledge memory.
    assert any(
        "memory" in step.lower()
        for step in plan
    ), (
        "Planner did not add a memory-aware planning step."
    )

    print(
        "[PASS] Planner added a memory-aware planning step."
    )

    # Planning memory context should exist.
    assert planning_memory_context.strip(), (
        "planning_memory_context is empty."
    )

    print(
        "[PASS] planning_memory_context was generated."
    )

    # The remembered decision should reach the planner.
    assert "Qdrant" in planning_memory_context, (
        "Remembered Qdrant decision was not passed "
        "into planning context."
    )

    print(
        "[PASS] Remembered Qdrant decision reached the planner."
    )

    # Temporal memory should also be available.
    assert result.get(
        "temporal_memory_results"
    ), (
        "Temporal memory was not available to the planner."
    )

    print(
        "[PASS] Temporal memory was available to the planner."
    )

    # --------------------------------------------------
    # SUCCESS
    # --------------------------------------------------

    print()
    print_separator()
    print("TEST RESULT")
    print_separator()

    print(
        "MEMORY-AUGMENTED PLANNING TEST PASSED"
    )

    print_separator()


if __name__ == "__main__":
    main()