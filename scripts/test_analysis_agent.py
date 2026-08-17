import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState
from agents.rag_agent import RAGAgent
from agents.analysis_agent import AnalysisAgent


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


print("=" * 60)
print("OMNIMIND ANALYSIS AGENT TEST")
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


# ---------------------------------------------------------
# Display analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS RESULT")
print("=" * 60)

print(state["analysis"])

print("\nCurrent Step:")
print(state["current_step"])


print("\n" + "=" * 60)
print("ANALYSIS AGENT SUCCESSFUL")
print("=" * 60)