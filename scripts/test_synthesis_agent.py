import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
    )
)

from agents.synthesis_agent import SynthesisAgent


agent = SynthesisAgent()


# ============================================================
# TEST 1 — RAG ONLY
# ============================================================

print("=" * 60)
print("TEST 1 — RAG ONLY")
print("=" * 60)

state = {
    "query": "What datasets were used to evaluate the RAG models?",
    "route": "rag",
    "analysis": (
        "The document evidence identifies Natural Questions, "
        "TriviaQA, WebQuestions, and CuratedTrec."
    ),
    "rag_results": [
        {
            "source": "sample.pdf",
            "page": 4,
            "chunk": 5,
            "score": 0.91,
            "text": (
                "We evaluate RAG models on Natural Questions, "
                "TriviaQA, WebQuestions and CuratedTrec."
            ),
        }
    ],
    "research_results": [],
}

result = agent.run(state)

print("\nFINAL ANSWER:")
print(result["final_answer"])

print("\nSOURCES:")
print(result["sources"])


# ============================================================
# TEST 2 — RESEARCH ONLY
# ============================================================

print("\n" + "=" * 60)
print("TEST 2 — RESEARCH ONLY")
print("=" * 60)

state = {
    "query": "What are the latest developments in RAG?",
    "route": "research",
    "analysis": (
        "The supplied web evidence mentions hybrid retrieval "
        "and multimodal evidence."
    ),
    "rag_results": [],
    "research_results": [
        {
            "title": "Modern RAG Survey",
            "url": "https://example.com/rag",
            "snippet": (
                "Recent RAG systems increasingly use "
                "hybrid retrieval and multimodal evidence."
            ),
        }
    ],
}

result = agent.run(state)

print("\nFINAL ANSWER:")
print(result["final_answer"])

print("\nSOURCES:")
print(result["sources"])


# ============================================================
# TEST 3 — BOTH
# ============================================================

print("\n" + "=" * 60)
print("TEST 3 — BOTH")
print("=" * 60)

state = {
    "query": (
        "Compare the RAG approach in my document "
        "with recent developments in RAG."
    ),
    "route": "both",
    "analysis": (
        "The document describes BART, DPR and a Wikipedia "
        "index. The external evidence mentions hybrid "
        "retrieval and multimodal evidence."
    ),
    "rag_results": [
        {
            "source": "sample.pdf",
            "page": 2,
            "chunk": 3,
            "score": 0.89,
            "text": (
                "The original RAG architecture combines "
                "BART with DPR and a Wikipedia index."
            ),
        }
    ],
    "research_results": [
        {
            "title": "Modern RAG Survey",
            "url": "https://example.com/rag",
            "snippet": (
                "Modern RAG systems increasingly use "
                "hybrid retrieval and multimodal evidence."
            ),
        }
    ],
}

result = agent.run(state)

print("\nFINAL ANSWER:")
print(result["final_answer"])

print("\nSOURCES:")
print(result["sources"])


print("\n" + "=" * 60)
print("SYNTHESIS AGENT TEST COMPLETE")
print("=" * 60)