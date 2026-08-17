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


print("=" * 60)
print("OMNIMIND LANGGRAPH TEST")
print("=" * 60)


state: AgentState = {
    "query": (
        "What datasets were used to evaluate "
        "the RAG models?"
    ),
    "plan": [],
    "current_step": "starting",
    "research_results": [],
    "rag_results": [],
    "analysis": "",
    "final_answer": "",
    "sources": [],
}


print("\nBuilding OmniMind graph...")

omnimind = OmniMindGraph(
    pdf_path=str(PDF_PATH)
)


print("Graph created successfully.")


print("\nExecuting graph...")


result = omnimind.run(
    state
)


print("\n" + "=" * 60)
print("PLANNER OUTPUT")
print("=" * 60)

for index, step in enumerate(
    result["plan"],
    start=1,
):
    print(
        f"{index}. {step}"
    )


print("\n" + "=" * 60)
print("RAG EVIDENCE")
print("=" * 60)

print(
    f"Evidence items: "
    f"{len(result['rag_results'])}"
)


print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

print(
    result["analysis"]
)


print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(
    result["final_answer"]
)


print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

seen = set()

for source in result["sources"]:

    key = (
        source["source"],
        source["page"],
        source["chunk"],
    )

    if key in seen:
        continue

    seen.add(key)

    print(
        f"- {source['source']} "
        f"| Page {source['page']} "
        f"| Chunk {source['chunk']}"
    )


print("\n" + "=" * 60)
print("CURRENT STATE")
print("=" * 60)

print(
    result["current_step"]
)


omnimind.close()


print("\n" + "=" * 60)
print("LANGGRAPH ORCHESTRATION SUCCESSFUL")
print("=" * 60)