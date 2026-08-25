import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState
from agents.research_agent import ResearchAgent


print("=" * 60)
print("OMNIMIND RESEARCH AGENT TEST")
print("=" * 60)


state: AgentState = {
    "query": (
        "latest developments in "
        "Retrieval-Augmented Generation"
    ),
    "plan": [
        "Perform external web research",
        "Analyze research results",
        "Generate the final answer",
    ],
    "current_step": "research",
    "route": "research",
    "research_results": [],
    "rag_results": [],
    "analysis": "",
    "final_answer": "",
    "sources": [],
}


print("\nQuery:")
print(state["query"])

print("\nCalling Research Agent...")
print("Research Agent → MCP Client → MCP Server → Web")


agent = ResearchAgent()

updated_state = agent.run(
    state
)


if updated_state["current_step"] == "research_failed":

    print("\n" + "=" * 60)
    print("RESEARCH AGENT FAILED")
    print("=" * 60)

    print(
        updated_state.get(
            "error",
            "Unknown error",
        )
    )

    raise SystemExit(1)


print("\n" + "=" * 60)
print("RESEARCH RESULTS")
print("=" * 60)


results = updated_state.get(
    "research_results",
    [],
)


print(
    f"\nResults found: {len(results)}"
)


for index, result in enumerate(
    results,
    start=1,
):

    print(f"\nResult {index}")

    print(
        f"Title: "
        f"{result.get('title', '')}"
    )

    print(
        f"URL: "
        f"{result.get('url', '')}"
    )

    print(
        f"Snippet: "
        f"{result.get('snippet', '')}"
    )


print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)


for source in updated_state.get(
    "sources",
    [],
):

    print(
        f"- {source['source']}"
    )

    print(
        f"  {source['url']}"
    )


print("\nCurrent Step:")
print(
    updated_state["current_step"]
)


print("\n" + "=" * 60)
print("RESEARCH AGENT SUCCESSFUL")
print("=" * 60)