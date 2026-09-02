from agents.analysis_agent import AnalysisAgent


agent = AnalysisAgent()


# ============================================================
# TEST 1 — RAG ONLY
# ============================================================

print("=" * 60)
print("TEST 1 — RAG ONLY")
print("=" * 60)

state = {
    "query": "What datasets were used to evaluate the RAG models?",
    "route": "rag",
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

print("\nAnalysis:")
print(result["analysis"])

print("\nCurrent Step:")
print(result["current_step"])


# ============================================================
# TEST 2 — RESEARCH ONLY
# ============================================================

print("\n" + "=" * 60)
print("TEST 2 — RESEARCH ONLY")
print("=" * 60)

state = {
    "query": "What are the latest developments in RAG?",
    "route": "research",
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

print("\nAnalysis:")
print(result["analysis"])

print("\nCurrent Step:")
print(result["current_step"])


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

print("\nAnalysis:")
print(result["analysis"])

print("\nCurrent Step:")
print(result["current_step"])


print("\n" + "=" * 60)
print("ANALYSIS AGENT TEST COMPLETE")
print("=" * 60)