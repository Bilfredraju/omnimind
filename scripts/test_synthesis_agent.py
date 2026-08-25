import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.synthesis_agent import SynthesisAgent
from agents.state import AgentState


agent = SynthesisAgent()


def run_test(
    title: str,
    state: AgentState,
):
    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)

    result = agent.run(state)

    print("\nFINAL ANSWER")
    print("-" * 60)
    print(result["final_answer"])

    print("\nSOURCES")
    print("-" * 60)

    for source in result.get(
        "sources",
        [],
    ):
        print(source)

    print("\nCURRENT STEP:")
    print(result["current_step"])


# ----------------------------------------------------------
# Test 1 — RAG only
# ----------------------------------------------------------

run_test(
    "TEST 1 — RAG SYNTHESIS",
    {
        "query": (
            "What datasets were used to evaluate "
            "the RAG models?"
        ),
        "route": "rag",
        "analysis": (
            "The document states that the RAG models "
            "were evaluated on Natural Questions, "
            "TriviaQA, WebQuestions, CuratedTrec "
            "and MSMARCO."
        ),
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
    "TEST 2 — WEB SYNTHESIS",
    {
        "query": (
            "What are recent developments "
            "in Retrieval-Augmented Generation?"
        ),
        "route": "research",
        "analysis": (
            "Recent research focuses on improving "
            "retrieval quality, grounding and "
            "context selection."
        ),
        "rag_results": [],
        "research_results": [
            {
                "title": (
                    "Retrieval-Augmented Generation "
                    "Survey"
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
    "TEST 3 — RAG + WEB SYNTHESIS",
    {
        "query": (
            "Compare the RAG approach in my document "
            "with recent developments in RAG."
        ),
        "route": "both",
        "analysis": (
            "The document describes RAG as a system "
            "combining retrieval with generation and "
            "non-parametric memory. Recent research "
            "focuses on improving retrieval quality, "
            "context selection and grounding."
        ),
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
print("SYNTHESIS AGENT TESTS COMPLETE")
print("=" * 60)