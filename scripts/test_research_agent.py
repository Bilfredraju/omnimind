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
        "Find recent information about "
        "Retrieval-Augmented Generation."
    ),
    "plan": [
        "Perform external research",
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


agent = ResearchAgent()

updated_state = agent.run(
    state
)


print("\nQuery:")
print(updated_state["query"])


print("\nResearch Results:")
print("-" * 60)

for index, result in enumerate(
    updated_state["research_results"],
    start=1,
):
    print(f"\nResult {index}")

    print(
        f"Title: {result['title']}"
    )

    print(
        f"Source: {result['source']}"
    )

    print(
        f"Snippet: {result['snippet']}"
    )


print("\nCurrent Step:")
print(
    updated_state["current_step"]
)


print("\n" + "=" * 60)
print("RESEARCH AGENT SUCCESSFUL")
print("=" * 60)