"""
OmniMind Citation Evaluation

Phase 19.7.2
End-to-end validation of document citations produced by the full graph.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.graph import OmniMindGraph
from evaluation.metrics import (
    extract_document_citations,
    extract_web_citations,
    citation_coverage,
)


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample.pdf"


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Sample PDF not found: {PDF_PATH}")

    graph = OmniMindGraph(pdf_path=str(PDF_PATH))

    query = "What datasets were used to evaluate the RAG models?"

    state = graph.run({"query": query})

    answer = state.get("final_answer", "")
    sources = state.get("sources", [])
    rag_results = state.get("rag_results", [])

    print("=" * 70)
    print("OMNIMIND END-TO-END CITATION EVALUATION")
    print("=" * 70)

    print(f"\nQuery:\n{query}")

    print("\nGraph State:")
    print(f"  Current Step : {state.get('current_step')}")
    print(f"  Error        : {state.get('error')}")

    if not answer:
        raise AssertionError("Final answer is empty.")

    print("\nFinal Answer:")
    print(answer)

    document_citations = extract_document_citations(answer)
    web_citations = extract_web_citations(answer)

    print("\nDetected Citations:")
    print(f"  Document citations : {document_citations}")
    print(f"  Web citations      : {web_citations}")

    print("\nEvidence:")
    print(f"  RAG results   : {len(rag_results)}")
    print(f"  Source records: {len(sources)}")

    document_source_count = sum(
        1
        for source in sources
        if source.get("type") == "document"
    )

    print(f"  Document sources: {document_source_count}")

    coverage = citation_coverage(
        answer,
        expected_document_citations=document_source_count,
        expected_web_citations=0,
    )

    print(f"\nCitation Coverage: {coverage:.3f}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    assert state.get("error") in (None, ""), (
        f"Graph returned an error: {state.get('error')}"
    )

    assert state.get("current_step") in (
        "memory_write_complete",
        "synthesis_complete",
    ), (
        f"Unexpected final graph step: {state.get('current_step')}"
    )

    assert len(rag_results) > 0, (
        "Expected at least one RAG result."
    )

    assert len(sources) > 0, (
        "Expected at least one final source record."
    )

    assert len(document_citations) > 0, (
        "Expected at least one document citation in final answer."
    )

    assert coverage > 0.0, (
        "Citation coverage should be greater than zero."
    )

    # Every detected numeric citation should correspond to a source record.
    source_citation_ids = {
        source.get("citation_id")
        for source in sources
        if source.get("type") == "document"
    }

    missing = [
        citation
        for citation in document_citations
        if citation not in source_citation_ids
    ]

    assert not missing, (
        f"Answer contains citations without matching source records: {missing}"
    )

    print("\n" + "=" * 70)
    print("PHASE 19.7.2 END-TO-END CITATION EVALUATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()