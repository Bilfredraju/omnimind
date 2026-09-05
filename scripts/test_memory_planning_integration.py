import os
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from agents.graph import OmniMindGraph


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/sample.pdf"


def print_separator():
    print("=" * 70)


# ============================================================
# TEST
# ============================================================

def main():
    print_separator()
    print("OMNIMIND MEMORY-PLANNING GRAPH INTEGRATION TEST")
    print_separator()

    # --------------------------------------------------------
    # Initialize graph
    # --------------------------------------------------------

    print()
    print("Initializing OmniMind...")

    graph = OmniMindGraph(
        pdf_path=PDF_PATH
    )

    print("OmniMind initialized successfully.")

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------
    #
    # This query is intentionally related to the remembered
    # Qdrant decision used in the previous planning test.
    #
    # The important part is that we DO NOT manually place
    # memory into AgentState here.
    #
    # The real graph must:
    #
    # Query
    #   ↓
    # MemoryAgent
    #   ↓
    # Memory Recall
    #   ↓
    # Planner
    #
    # --------------------------------------------------------

    query = (
        "How should I implement vector retrieval "
        "for my OmniMind project?"
    )

    state = {
        "query": query,
    }

    # --------------------------------------------------------
    # Execute graph
    # --------------------------------------------------------

    print()
    print("Executing OmniMind graph...")

    result = graph.run(state)

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    print()
    print_separator()
    print("QUERY")
    print_separator()

    print(
        result.get(
            "query",
            query,
        )
    )

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    print()
    print_separator()
    print("MEMORY RECALL")
    print_separator()

    semantic_memories = result.get(
        "memory_results",
        [],
    )

    temporal_memories = result.get(
        "temporal_memory_results",
        [],
    )

    print(
        f"Semantic memories retrieved: "
        f"{len(semantic_memories)}"
    )

    print(
        f"Temporal memories retrieved: "
        f"{len(temporal_memories)}"
    )

    # --------------------------------------------------------
    # PLANNING
    # --------------------------------------------------------

    print()
    print_separator()
    print("PLANNER")
    print_separator()

    print(
        "Selected route:",
        result.get(
            "route",
            "unknown",
        ),
    )

    print()
    print("Execution plan:")

    plan = result.get(
        "plan",
        [],
    )

    for index, step in enumerate(
        plan,
        start=1,
    ):
        print(
            f"{index}. {step}"
        )

    # --------------------------------------------------------
    # PLANNING MEMORY CONTEXT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DOWNSTREAM RESULTS
    # --------------------------------------------------------

    print()
    print_separator()
    print("DOWNSTREAM EXECUTION")
    print_separator()

    print(
        "RAG results:",
        len(
            result.get(
                "rag_results",
                [],
            )
        ),
    )

    print(
        "Research results:",
        len(
            result.get(
                "research_results",
                [],
            )
        ),
    )

    print(
        "Current step:",
        result.get(
            "current_step",
            "",
        ),
    )

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    print()
    print_separator()
    print("FINAL ANSWER")
    print_separator()

    final_answer = result.get(
        "final_answer",
        "",
    )

    print(final_answer)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print_separator()
    print("VALIDATING END-TO-END MEMORY PLANNING")
    print_separator()

    # 1. Memory recall must execute.
    assert (
        "memory_results" in result
    ), (
        "Semantic memory was not propagated "
        "through the graph."
    )

    print(
        "[PASS] Semantic memory reached the graph state."
    )

    # 2. Temporal memory field must exist.
    assert (
        "temporal_memory_results" in result
    ), (
        "Temporal memory was not propagated "
        "through the graph."
    )

    print(
        "[PASS] Temporal memory reached the graph state."
    )

    # 3. Planner must execute.
    assert result.get(
        "plan"
    ), (
        "Planner returned an empty plan."
    )

    print(
        "[PASS] Planner generated an execution plan."
    )

    # 4. Planner should have received memory.
    assert result.get(
        "planning_memory_context",
        "",
    ).strip(), (
        "Planner did not receive memory context."
    )

    print(
        "[PASS] Planner received memory context."
    )

    # 5. The remembered Qdrant decision should reach
    # the planner in the real graph.
    assert "Qdrant" in result.get(
        "planning_memory_context",
        "",
    ), (
        "The remembered Qdrant decision did not "
        "reach the planner."
    )

    print(
        "[PASS] Remembered Qdrant decision reached planner."
    )

    # 6. The memory-aware planning step should exist.
    assert any(
        "memory" in step.lower()
        for step in result.get(
            "plan",
            [],
        )
    ), (
        "Planner did not create a memory-aware step."
    )

    print(
        "[PASS] Memory-aware planning step was generated."
    )

    # 7. Downstream execution must continue.
    assert result.get(
        "current_step"
    ) == "memory_write_complete", (
        "Graph did not complete the full pipeline."
    )

    print(
        "[PASS] Graph completed through Memory Write."
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print()
    print_separator()
    print("TEST RESULT")
    print_separator()

    print(
        "MEMORY-PLANNING GRAPH INTEGRATION TEST PASSED"
    )

    print_separator()

    # --------------------------------------------------------
    # Close graph
    # --------------------------------------------------------

    print()
    print("Closing OmniMind...")

    graph.close()

    print("OmniMind closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()