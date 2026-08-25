import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.analysis_agent import AnalysisAgent
from agents.state import AgentState


agent = AnalysisAgent()


def run_test(
    title: str,
    state: AgentState,
):
    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)

    result = agent.run(state)

    print("\nAnalysis:")
    print("-" * 60)
    print(result["analysis"])

    print("\nCurrent Step:")
    print(result["current_step"])


# ----------------------------------------------------------
# Test 1 — RAG only
# ----------------------------------------------------------

run_test(
    "TEST 1 — RAG EVIDENCE",
    {
        "query": (
            "What datasets were used to evaluate "
            "the RAG models?"
        ),
        "route": "rag",
        "rag_results": [
            {
                "source": "sample.pdf",
                "page": 4,
                "chunk": 5,
                "score": 0.82,
                "text": (
                    "The RAG models were evaluated "
                    "on Natural Questions, TriviaQA, "
                    "WebQuestions, CuratedTrec and "
                    "MSMARCO."
                ),
            }
        ],
        "research_results": [],
        "sources": [],
    },
)


# ----------------------------------------------------------
# Test 2 — Research only
# ----------------------------------------------------------

run_test(
    "TEST 2 — WEB RESEARCH",
    {
        "query": (
            "What are recent developments "
            "in Retrieval-Augmented Generation?"
        ),
        "route": "research",
        "rag_results": [],
        "research_results": [
            {
                "title": (
                    "Retrieval-Augmented Generation: "
                    "A Comprehensive Survey"
                ),
                "url": (
                    "https://arxiv.org/"
                ),
                "snippet": (
                    "Recent RAG research explores "
                    "retrieval quality, grounding "
                    "and architecture improvements."
                ),
            }
        ],
        "sources": [],
    },
)


# ----------------------------------------------------------
# Test 3 — Both
# ----------------------------------------------------------

run_test(
    "TEST 3 — RAG + WEB RESEARCH",
    {
        "query": (
            "Compare the RAG approach in my document "
            "with recent developments in RAG."
        ),
        "route": "both",
        "rag_results": [
            {
                "source": "sample.pdf",
                "page": 4,
                "chunk": 5,
                "score": 0.82,
                "text": (
                    "The original RAG approach combines "
                    "retrieval with generation using "
                    "non-parametric memory."
                ),
            }
        ],
        "research_results": [
            {
                "title": (
                    "Modern RAG Architectures"
                ),
                "url": (
                    "https://arxiv.org/"
                ),
                "snippet": (
                    "Modern RAG systems increasingly "
                    "focus on retrieval quality, "
                    "context selection and grounding."
                ),
            }
        ],
        "sources": [],
    },
)


print("\n")
print("=" * 60)
print("ANALYSIS AGENT TESTS COMPLETE")
print("=" * 60)