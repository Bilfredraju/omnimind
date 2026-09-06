"""
OmniMind Citation Evaluation

Phase 19.7.4
End-to-end validation of document citations and citation quality
produced by the full OmniMind graph.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.graph import OmniMindGraph
from evaluation.metrics import (
    citation_coverage,
    citation_quality,
    extract_document_citations,
    extract_web_citations,
)


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample.pdf"


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Sample PDF not found: {PDF_PATH}"
        )

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

    # ------------------------------------------------------------------
    # GRAPH STATE
    # ------------------------------------------------------------------

    print("\nGraph State:")
    print(f"  Current Step : {state.get('current_step')}")
    print(f"  Error        : {state.get('error')}")

    # ------------------------------------------------------------------
    # BASIC VALIDATION
    # ------------------------------------------------------------------

    if not answer:
        raise AssertionError("Final answer is empty.")

    print("\nFinal Answer:")
    print(answer)

    # ------------------------------------------------------------------
    # CITATION EXTRACTION
    # ------------------------------------------------------------------

    document_citations = extract_document_citations(answer)
    web_citations = extract_web_citations(answer)

    print("\nDetected Citations:")
    print(f"  Document citations : {document_citations}")
    print(f"  Web citations      : {web_citations}")

    # ------------------------------------------------------------------
    # EVIDENCE INFORMATION
    # ------------------------------------------------------------------

    print("\nEvidence:")
    print(f"  RAG results   : {len(rag_results)}")
    print(f"  Source records: {len(sources)}")

    document_source_count = sum(
        1
        for source in sources
        if source.get("type") == "document"
    )

    print(f"  Document sources: {document_source_count}")

    # ------------------------------------------------------------------
    # CITATION COVERAGE
    # ------------------------------------------------------------------

    coverage = citation_coverage(
        answer,
        expected_document_citations=document_source_count,
        expected_web_citations=0,
    )

    # ------------------------------------------------------------------
    # EXPECTED CITATION IDS
    # ------------------------------------------------------------------

    expected_citation_ids = {
        source.get("citation_id")
        for source in sources
        if source.get("type") == "document"
        and source.get("citation_id")
    }

    # ------------------------------------------------------------------
    # CITATION QUALITY
    # ------------------------------------------------------------------

    quality = citation_quality(
        answer,
        sources,
        expected_citation_ids=expected_citation_ids,
    )

    print(f"\nCitation Coverage: {coverage:.3f}")

    print("\nCitation Quality:")
    print(
        f"  Precision    : "
        f"{quality['citation_precision']:.3f}"
    )
    print(
        f"  Recall       : "
        f"{quality['citation_recall']:.3f}"
    )
    print(
        f"  Correctness  : "
        f"{quality['citation_correctness']:.3f}"
    )
    print(
        f"  Completeness : "
        f"{quality['citation_completeness']:.3f}"
    )
    print(
        f"  F1           : "
        f"{quality['citation_f1']:.3f}"
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    assert state.get("error") in (None, ""), (
        f"Graph returned an error: {state.get('error')}"
    )

    assert state.get("current_step") in (
        "memory_write_complete",
        "synthesis_complete",
    ), (
        f"Unexpected final graph step: "
        f"{state.get('current_step')}"
    )

    assert len(rag_results) > 0, (
        "Expected at least one RAG result."
    )

    assert len(sources) > 0, (
        "Expected at least one final source record."
    )

    assert len(document_citations) > 0, (
        "Expected at least one document citation "
        "in final answer."
    )

    assert coverage > 0.0, (
        "Citation coverage should be greater than zero."
    )

    # ------------------------------------------------------------------
    # CITATION ID VALIDATION
    # ------------------------------------------------------------------

    source_citation_ids = {
        source.get("citation_id")
        for source in sources
        if source.get("type") == "document"
        and source.get("citation_id")
    }

    missing = [
        citation
        for citation in document_citations
        if citation not in source_citation_ids
    ]

    assert not missing, (
        "Answer contains citations without matching "
        f"source records: {missing}"
    )

    # ------------------------------------------------------------------
    # CITATION QUALITY VALIDATION
    # ------------------------------------------------------------------

    assert quality["citation_precision"] >= 0.0, (
        "Citation precision must be >= 0."
    )

    assert quality["citation_precision"] <= 1.0, (
        "Citation precision must be <= 1."
    )

    assert quality["citation_recall"] >= 0.0, (
        "Citation recall must be >= 0."
    )

    assert quality["citation_recall"] <= 1.0, (
        "Citation recall must be <= 1."
    )

    assert quality["citation_correctness"] >= 0.0, (
        "Citation correctness must be >= 0."
    )

    assert quality["citation_correctness"] <= 1.0, (
        "Citation correctness must be <= 1."
    )

    assert quality["citation_completeness"] >= 0.0, (
        "Citation completeness must be >= 0."
    )

    assert quality["citation_completeness"] <= 1.0, (
        "Citation completeness must be <= 1."
    )

    assert quality["citation_f1"] >= 0.0, (
        "Citation F1 must be >= 0."
    )

    assert quality["citation_f1"] <= 1.0, (
        "Citation F1 must be <= 1."
    )

    # ------------------------------------------------------------------
    # SUCCESS
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PHASE 19.7.4 END-TO-END CITATION QUALITY "
        "EVALUATION: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()