import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.synthesis_agent import SynthesisAgent


# ============================================================
# TEST 1
# ============================================================

def test_structured_document_citations():
    """
    Validate that SynthesisAgent converts structured RAG citations
    into LLM evidence and final source records correctly.
    """

    agent = SynthesisAgent()

    rag_results = [
        {
            "text": "The models were evaluated on Natural Questions.",
            "score": 0.91,
            "rerank_score": 0.88,
            "hybrid_score": 0.02,
            "citation": {
                "citation_id": "[1]",
                "document_name": "sample.pdf",
                "source": "sample.pdf",
                "page": 1,
                "chunk_id": "chunk-test-001",
            },
        },
        {
            "text": "Additional evaluation was performed on WebQuestions.",
            "score": 0.82,
            "rerank_score": 0.79,
            "hybrid_score": 0.018,
            "citation": {
                "citation_id": "[2]",
                "document_name": "sample.pdf",
                "source": "sample.pdf",
                "page": 4,
                "chunk_id": "chunk-test-002",
            },
        },
    ]

    evidence, sources = agent._build_rag_evidence(
        rag_results
    )

    # --------------------------------------------------
    # Evidence validation
    # --------------------------------------------------

    assert "[RAG Source [1]]" in evidence
    assert "[RAG Source [2]]" in evidence

    assert "Document: sample.pdf" in evidence
    assert "Page: 1" in evidence
    assert "Page: 4" in evidence

    assert "Chunk ID: chunk-test-001" in evidence
    assert "Chunk ID: chunk-test-002" in evidence

    assert "Natural Questions" in evidence
    assert "WebQuestions" in evidence

    # --------------------------------------------------
    # Source record validation
    # --------------------------------------------------

    assert len(sources) == 2

    assert sources[0]["citation_id"] == "[1]"
    assert sources[0]["document_name"] == "sample.pdf"
    assert sources[0]["source"] == "sample.pdf"
    assert sources[0]["page"] == 1
    assert sources[0]["chunk_id"] == "chunk-test-001"
    assert sources[0]["type"] == "document"

    assert sources[1]["citation_id"] == "[2]"
    assert sources[1]["document_name"] == "sample.pdf"
    assert sources[1]["source"] == "sample.pdf"
    assert sources[1]["page"] == 4
    assert sources[1]["chunk_id"] == "chunk-test-002"
    assert sources[1]["type"] == "document"

    print("Structured citation validation: PASSED")


# ============================================================
# TEST 2
# ============================================================

def test_legacy_rag_result_fallback():
    """
    Validate backward compatibility when a RAG result does not
    contain a structured citation object.
    """

    agent = SynthesisAgent()

    rag_results = [
        {
            "text": "Legacy document evidence.",
            "score": 0.75,
            "source": "legacy.pdf",
            "page": 7,
            "chunk_id": "legacy-chunk-001",
        }
    ]

    evidence, sources = agent._build_rag_evidence(
        rag_results
    )

    assert "[RAG Source [1]]" in evidence
    assert "Document: legacy.pdf" in evidence
    assert "Source: legacy.pdf" in evidence
    assert "Page: 7" in evidence
    assert "Chunk ID: legacy-chunk-001" in evidence

    assert len(sources) == 1
    assert sources[0]["citation_id"] == "[1]"
    assert sources[0]["source"] == "legacy.pdf"
    assert sources[0]["document_name"] == "legacy.pdf"
    assert sources[0]["page"] == 7
    assert sources[0]["chunk_id"] == "legacy-chunk-001"
    assert sources[0]["type"] == "document"

    print("Legacy citation fallback: PASSED")


# ============================================================
# TEST 3
# ============================================================

def test_metadata_fallback():
    """
    Validate citation extraction from nested metadata when the
    explicit citation object is unavailable.
    """

    agent = SynthesisAgent()

    rag_results = [
        {
            "text": "Metadata-backed document evidence.",
            "score": 0.70,
            "metadata": {
                "source": "metadata.pdf",
                "document_name": "metadata.pdf",
                "page": 12,
                "chunk_id": "metadata-chunk-001",
            },
        }
    ]

    evidence, sources = agent._build_rag_evidence(
        rag_results
    )

    assert "[RAG Source [1]]" in evidence
    assert "Document: metadata.pdf" in evidence
    assert "Source: metadata.pdf" in evidence
    assert "Page: 12" in evidence
    assert "Chunk ID: metadata-chunk-001" in evidence

    assert len(sources) == 1
    assert sources[0]["citation_id"] == "[1]"
    assert sources[0]["document_name"] == "metadata.pdf"
    assert sources[0]["source"] == "metadata.pdf"
    assert sources[0]["page"] == 12
    assert sources[0]["chunk_id"] == "metadata-chunk-001"
    assert sources[0]["type"] == "document"

    print("Metadata fallback: PASSED")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_structured_document_citations()
    test_legacy_rag_result_fallback()
    test_metadata_fallback()

    print()
    print("=" * 60)
    print("PHASE 19.6.1 CITATION VALIDATION: PASSED")
    print("=" * 60)