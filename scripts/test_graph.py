import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState
from agents.graph import OmniMindGraph


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


def run_test(query: str):

    print("\n")
    print("=" * 60)
    print("QUERY")
    print("=" * 60)
    print(query)

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

    omnimind = OmniMindGraph(
        pdf_path=str(PDF_PATH)
    )

    try:

        result = omnimind.run(
            state
        )

        print("\n" + "=" * 60)
        print("SELECTED ROUTE")
        print("=" * 60)

        print(
            result.get(
                "route",
                "unknown",
            )
        )

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

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(
            result.get(
                "final_answer",
                "",
            )
        )

        print("\n" + "=" * 60)
        print("CURRENT STEP")
        print("=" * 60)

        print(
            result.get(
                "current_step",
                "",
            )
        )

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
                    f"  URL: {source['url']}"
                )

            if source.get("page"):
                print(
                    f"  Page: {source['page']}"
                )

        print("\n" + "=" * 60)
        print("GRAPH EXECUTION SUCCESSFUL")
        print("=" * 60)

    finally:

        omnimind.close()


if __name__ == "__main__":

    print("=" * 60)
    print("OMNIMIND CONDITIONAL LANGGRAPH TEST")
    print("=" * 60)

    # Test 1: Document-based question
    run_test(
        "What datasets were used to evaluate "
        "the RAG models?"
    )

    # Test 2: External research question
    run_test(
        "What are the latest developments "
        "in Retrieval-Augmented Generation?"
    )

    # Test 3: Combined question
    run_test(
        "Compare the RAG approach in my document "
        "with recent developments in RAG."
    )