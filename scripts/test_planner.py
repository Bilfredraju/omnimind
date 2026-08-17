import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState
from agents.planner import PlannerAgent


print("=" * 60)
print("OMNIMIND PLANNER AGENT TEST")
print("=" * 60)


state: AgentState = {
    "query": (
        "What datasets were used to evaluate "
        "the RAG models?"
    ),
    "plan": [],
    "current_step": "planning",
    "research_results": [],
    "rag_results": [],
    "analysis": "",
    "final_answer": "",
    "sources": [],
}


planner = PlannerAgent()

updated_state = planner.plan(state)


print("\nUser Query:")
print(updated_state["query"])


print("\nGenerated Plan:")
print("-" * 60)

for index, step in enumerate(
    updated_state["plan"],
    start=1,
):
    print(f"{index}. {step}")


print("\nCurrent Step:")
print(updated_state["current_step"])
print("\nSelected Route:")
print(updated_state["route"])


print("\n" + "=" * 60)
print("PLANNER AGENT SUCCESSFUL")
print("=" * 60)