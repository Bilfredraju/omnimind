"""
OmniMind RAG Agent Integration Test

Validates that the RAGAgent can:
1. Receive a user query.
2. Retrieve document evidence through the MCP document client.
3. Consume the persistent Qdrant retrieval schema.
4. Preserve document metadata and citations.
5. Produce usable RAG evidence.

This test is intentionally deterministic and does not call the LLM directly.
"""

from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from agents.rag_agent import RAGAgent


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

QUERY = "What datasets were used to evaluate the RAG models?"
TOP_K = 5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_document_name(result: dict) -> str:
    """
    Extract the document name from the current persistent MCP schema.
    Supports both current and legacy result formats.
    """

    metadata = result.get("metadata") or {}

    return (
        result.get("document_name")
        or metadata.get("document_name")
        or result.get("document")
        or metadata.get("source")
        or "Unknown"
    )


def get_page(result: dict):
    """
    Extract page number from the current persistent MCP schema.
    """

    metadata = result.get("metadata") or {}

    return (
        result.get("page")
        or result.get("page_number")
        or metadata.get("page")
        or metadata.get("page_number")
        or "N/A"
    )


def get_chunk_id(result: dict) -> str:
    """
    Extract the stable chunk identifier.

    Current persistent schema:
        chunk_id

    Legacy compatibility:
        chunk
        result_id
    """

    metadata = result.get("metadata") or {}

    return (
        result.get("chunk_id")
        or metadata.get("chunk_id")
        or result.get("chunk")
        or result.get("result_id")
        or "N/A"
    )


def get_text(result: dict) -> str:
    """
    Extract retrieved text from the current or legacy result schema.
    """

    return (
        result.get("text")
        or result.get("chunk")
        or result.get("content")
        or ""
    )


def get_citation(result: dict):
    """
    Extract structured citation metadata.
    """

    citation = result.get("citation")

    if citation:
        return citation

    metadata = result.get("metadata") or {}

    return metadata.get("citation")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("OMNIMIND RAG AGENT TEST")
    print("=" * 60)

    print()
    print("User Query:")
    print(QUERY)

    # -----------------------------------------------------------------------
    # Create RAG agent
    # -----------------------------------------------------------------------

    agent = RAGAgent()

    # -----------------------------------------------------------------------
    # Build minimal AgentState
    # -----------------------------------------------------------------------

    state = {
        "query": QUERY,
        "memory_context": "",
        "temporal_memory_context": "",
        "rag_results": [],
        "current_step": "",
        "error": "",
    }

    # -----------------------------------------------------------------------
    # Execute RAG agent
    # -----------------------------------------------------------------------

    try:
        result_state = agent.run(state)
    except Exception as exc:
        print()
        print("RAG Agent execution: FAILED")
        print(f"Error: {exc}")
        raise

    # -----------------------------------------------------------------------
    # Validate execution
    # -----------------------------------------------------------------------

    error = result_state.get("error")

    if error:
        print()
        print("RAG Agent execution: FAILED")
        print(f"Error: {error}")
        raise RuntimeError(error)

    results = result_state.get("rag_results") or []

    if not results:
        print()
        print("RAG Agent execution: FAILED")
        print("No RAG results were returned.")
        raise AssertionError("RAGAgent returned no retrieval results.")

    print()
    print("RAG Agent execution: PASSED")

    print()
    print("RAG Evidence:")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # Display and validate evidence
    # -----------------------------------------------------------------------

    valid_results = 0
    valid_citations = 0

    for index, evidence in enumerate(results[:TOP_K], start=1):

        document_name = get_document_name(evidence)
        page = get_page(evidence)
        chunk_id = get_chunk_id(evidence)
        text = get_text(evidence)
        citation = get_citation(evidence)

        print()
        print(f"Evidence {index}")

        if evidence.get("score") is not None:
            print(f"Score: {evidence.get('score'):.4f}")

        if evidence.get("rerank_score") is not None:
            print(f"Rerank Score: {evidence.get('rerank_score'):.6f}")

        print(f"Document: {document_name}")
        print(f"Page: {page}")
        print(f"Chunk: {chunk_id}")

        if citation:
            print(f"Citation: {citation}")

        # Display a compact text preview.
        preview = " ".join(text.split())

        if len(preview) > 300:
            preview = preview[:300] + "..."

        print(f"Text: {preview}")

        # ---------------------------------------------------------------
        # Evidence validation
        # ---------------------------------------------------------------

        if text.strip():
            valid_results += 1

        if citation:
            valid_citations += 1

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------

    assert valid_results == min(TOP_K, len(results)), (
        "One or more RAG results contain no retrievable text."
    )

    assert valid_citations == min(TOP_K, len(results)), (
        "One or more RAG results are missing structured citations."
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("-" * 60)
    print("Validation Summary")
    print("-" * 60)

    print(f"Results returned: {len(results)}")
    print(f"Valid evidence: {valid_results}")
    print(f"Structured citations: {valid_citations}")

    print()
    print("Retrieval metadata:")

    retrieval = result_state.get("retrieval")

    if retrieval:
        print(retrieval)
    else:
        print("Available through individual RAG result metadata.")

    print()
    print("=" * 60)
    print("RAG AGENT TEST: PASSED")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    try:
        agent.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()