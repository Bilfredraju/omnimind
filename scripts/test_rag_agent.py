import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState
from agents.rag_agent import RAGAgent


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


print("=" * 60)
print("OMNIMIND RAG AGENT TEST")
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
# Create RAG Agent
# ---------------------------------------------------------

agent = RAGAgent(
    pdf_path=str(PDF_PATH)
)


# ---------------------------------------------------------
# Run agent
# ---------------------------------------------------------

updated_state = agent.run(
    state
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\nUser Query:")
print(updated_state["query"])


print("\nRAG Evidence:")
print("-" * 60)


for index, result in enumerate(
    updated_state["rag_results"],
    start=1,
):

    print(f"\nEvidence {index}")

    print(
        f"Score: {result['score']:.4f}"
    )

    print(
        f"Source: {result['source']}"
    )

    print(
        f"Page: {result['page']}"
    )

    print(
        f"Chunk: {result['chunk']}"
    )

    print("\nText:")
    print(
        result["text"][:500]
    )


print("\nCurrent Step:")
print(
    updated_state["current_step"]
)
print("\nError:")
print(
    updated_state.get("error", "")
)


agent.close()


print("\n" + "=" * 60)
print("RAG AGENT SUCCESSFUL")
print("=" * 60)