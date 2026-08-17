import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState
from agents.rag_agent import RAGAgent
from agents.analysis_agent import AnalysisAgent
from agents.synthesis_agent import SynthesisAgent


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


print("=" * 60)
print("OMNIMIND SYNTHESIS AGENT TEST")
print("=" * 60)


# ---------------------------------------------------------
# Initial state
# ---------------------------------------------------------

state: AgentState = {
    "query": (
        "What datasets were used to evaluate "
        "the RAG models?"
    ),
    "plan": [
        "Retrieve relevant information",
        "Analyze retrieved information",
        "Generate final answer",
    ],
    "current_step": "rag",
    "research_results": [],
    "rag_results": [],
    "analysis": "",
    "final_answer": "",
    "sources": [],
}


# ---------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------

rag_agent = RAGAgent(
    pdf_path=str(PDF_PATH)
)

state = rag_agent.run(state)

rag_agent.close()

print(
    f"\nRAG evidence collected: "
    f"{len(state['rag_results'])}"
)


# ---------------------------------------------------------
# Analysis Agent
# ---------------------------------------------------------

analysis_agent = AnalysisAgent()

state = analysis_agent.run(state)

print(
    "\nAnalysis completed."
)


# ---------------------------------------------------------
# Synthesis Agent
# ---------------------------------------------------------

synthesis_agent = SynthesisAgent()

state = synthesis_agent.run(state)


# ---------------------------------------------------------
# Display final answer
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("OMNIMIND FINAL ANSWER")
print("=" * 60)

print(
    state["final_answer"]
)


print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

seen = set()

for source in state["sources"]:

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


print("\nCurrent Step:")
print(
    state["current_step"]
)


print("\n" + "=" * 60)
print("SYNTHESIS AGENT SUCCESSFUL")
print("=" * 60)