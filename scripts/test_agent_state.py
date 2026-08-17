import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState


print("=" * 60)
print("OMNIMIND AGENT STATE TEST")
print("=" * 60)


state: AgentState = {
    "query": "What datasets were used to evaluate the RAG models?",
    "plan": [
        "Retrieve relevant information",
        "Analyze retrieved information",
        "Generate final answer",
    ],
    "current_step": "planning",
    "research_results": [],
    "rag_results": [],
    "analysis": "",
    "final_answer": "",
    "sources": [],
}


print("\nQuery:")
print(state["query"])

print("\nPlan:")

for index, step in enumerate(
    state["plan"],
    start=1,
):
    print(f"{index}. {step}")


print("\nCurrent step:")
print(state["current_step"])


print("\n" + "=" * 60)
print("AGENT STATE SUCCESSFUL")
print("=" * 60)