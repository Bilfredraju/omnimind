import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from agents.state import AgentState
from agents.graph import OmniMindGraph


# ============================================================
# PDF PATH
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


# ============================================================
# TEST RUNNER
# ============================================================

def run_test(
    query: str,
):
    """
    Run one complete OmniMind graph execution.

    A fresh graph instance is created for every test so
    MCP server processes are completely isolated.
    """

    print("\n")
    print("=" * 60)
    print("QUERY")
    print("=" * 60)
    print(query)

    # --------------------------------------------------------
    # Create a fresh graph for this test
    # --------------------------------------------------------

    print("\nInitializing OmniMind...")

    omnimind = OmniMindGraph(
        pdf_path=str(PDF_PATH)
    )

    state: AgentState = {
        "query": query,
        "plan": [],
        "current_step": "starting",
        "route": "",
        "research_results": [],
        "rag_results": [],
        "analysis": "",
        "final_answer": "",
        "sources": [],
        "error": "",
    }

    try:

        # ----------------------------------------------------
        # Execute graph
        # ----------------------------------------------------

        result = omnimind.run(state)

        # ----------------------------------------------------
        # Route
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("SELECTED ROUTE")
        print("=" * 60)

        print(
            result.get(
                "route",
                "unknown",
            )
        )

        # ----------------------------------------------------
        # Plan
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("PLAN")
        print("=" * 60)

        for index, step in enumerate(
            result.get("plan", []),
            start=1,
        ):
            print(
                f"{index}. {step}"
            )

        # ----------------------------------------------------
        # RAG results
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("RAG RESULTS")
        print("=" * 60)

        print(
            len(
                result.get(
                    "rag_results",
                    [],
                )
            )
        )

        # ----------------------------------------------------
        # Research results
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("RESEARCH RESULTS")
        print("=" * 60)

        print(
            len(
                result.get(
                    "research_results",
                    [],
                )
            )
        )

        # ----------------------------------------------------
        # Final answer
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(
            result.get(
                "final_answer",
                "",
            )
        )

        # ----------------------------------------------------
        # Current step
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("CURRENT STEP")
        print("=" * 60)

        print(
            result.get(
                "current_step",
                "",
            )
        )

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        if result.get("error"):

            print("\n" + "=" * 60)
            print("ERROR")
            print("=" * 60)

            print(
                result.get(
                    "error",
                    "",
                )
            )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        seen = set()

        for source in result.get(
            "sources",
            [],
        ):

            key = (
                source.get(
                    "source",
                    "",
                ),
                source.get(
                    "url",
                    "",
                ),
                source.get(
                    "page",
                    "",
                ),
                source.get(
                    "chunk",
                    "",
                ),
            )

            if key in seen:
                continue

            seen.add(key)

            print(
                f"- {source.get('source', '')}"
            )

            if source.get("url"):

                print(
                    f"  URL: "
                    f"{source['url']}"
                )

            if source.get("page"):

                print(
                    f"  Page: "
                    f"{source['page']}"
                )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("GRAPH EXECUTION SUCCESSFUL")
        print("=" * 60)

        return result

    finally:

        # ----------------------------------------------------
        # Close this graph before starting another test
        # ----------------------------------------------------

        print("\nClosing OmniMind...")

        omnimind.close()

        print("OmniMind closed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("OMNIMIND CONDITIONAL LANGGRAPH TEST")
    print("=" * 60)

    print()
    print(
        "Each test uses a fresh graph instance."
    )

    print()
    print(
        "Document and Research MCP processes "
        "are isolated between tests."
    )

    # --------------------------------------------------------
    # TEST 1 — RAG
    # --------------------------------------------------------

    run_test(
        "What datasets were used to evaluate "
        "the RAG models?"
    )

    # --------------------------------------------------------
    # TEST 2 — RESEARCH
    # --------------------------------------------------------

    run_test(
        "What are the latest developments "
        "in Retrieval-Augmented Generation?"
    )

    # --------------------------------------------------------
    # TEST 3 — BOTH
    # --------------------------------------------------------

    run_test(
        "Compare the RAG approach in my document "
        "with recent developments in RAG."
    )