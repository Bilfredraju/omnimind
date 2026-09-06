"""
OmniMind Grounding Evaluation

Phase 19.8.2
Deterministic evaluation of retrieved RAG evidence and grounding
infrastructure without invoking an LLM.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.document_server import get_knowledge_base
from evaluation.metrics import (
    answer_grounding_score,
    citation_evidence_alignment,
    evidence_utilization,
    grounding_quality,
)


QUERY = "What datasets were used to evaluate the RAG models?"

TOP_K = 5


def main():
    print("=" * 70)
    print("OMNIMIND RAG EVIDENCE GROUNDING EVALUATION")
    print("=" * 70)

    print(f"\nQuery:")
    print(QUERY)

    kb = get_knowledge_base()

    results = kb.search(
        QUERY,
        top_k=TOP_K,
    )

    if not results:
        raise AssertionError(
            "No RAG results were returned."
        )

    print("\nRetrieval:")
    print(f"  Requested top-k : {TOP_K}")
    print(f"  Results returned : {len(results)}")

    # ------------------------------------------------------------------
    # Extract evidence
    # ------------------------------------------------------------------

    evidence_texts = []
    evidence_records = []

    for index, result in enumerate(results, start=1):
        text = result.get("text", "")

        citation = result.get("citation")

        if not text:
            metadata = result.get("metadata", {})

            if isinstance(metadata, dict):
                text = metadata.get("text", "")

        if text:
            evidence_texts.append(text)

        if isinstance(citation, dict):
            evidence_records.append(citation)

        print(
            f"\nResult {index}:"
        )
        print(
            f"  Citation : "
            f"{citation.get('citation_id') if citation else None}"
        )
        print(
            f"  Document : "
            f"{citation.get('document_name') if citation else None}"
        )
        print(
            f"  Page     : "
            f"{citation.get('page') if citation else None}"
        )
        print(
            f"  Chunk ID  : "
            f"{citation.get('chunk_id') if citation else None}"
        )
        print(
            f"  Text chars: {len(text)}"
        )

    assert len(evidence_texts) > 0, (
        "No evidence text was extracted."
    )

    assert len(evidence_records) > 0, (
        "No structured citation records were extracted."
    )

    # ------------------------------------------------------------------
    # Build deterministic synthetic answer
    # ------------------------------------------------------------------
    #
    # This answer is deliberately constructed from the first retrieved
    # evidence chunk. It validates that the grounding metrics correctly
    # recognize evidence-supported content and citation alignment.
    #
    # It is NOT an LLM-generated answer.
    # ------------------------------------------------------------------

    first_evidence = evidence_texts[0]

    words = first_evidence.split()

    if len(words) > 40:
        grounded_fragment = " ".join(words[:40])
    else:
        grounded_fragment = first_evidence

    first_citation_id = evidence_records[0].get(
        "citation_id"
    )

    synthetic_answer = (
        f"{grounded_fragment} "
        f"{first_citation_id}"
    )

    print("\nDeterministic Grounded Answer:")
    print(synthetic_answer)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    utilization = evidence_utilization(
        synthetic_answer,
        evidence_texts,
    )

    grounding = answer_grounding_score(
        synthetic_answer,
        evidence_texts,
    )

    alignment = citation_evidence_alignment(
        synthetic_answer,
        evidence_records,
        evidence_records,
    )

    quality = grounding_quality(
        synthetic_answer,
        evidence_records,
        evidence_texts,
        evidence_records=evidence_records,
    )

    print("\nGrounding Metrics:")
    print(
        f"  Evidence Utilization : "
        f"{utilization:.3f}"
    )
    print(
        f"  Answer Grounding     : "
        f"{grounding:.3f}"
    )
    print(
        f"  Citation Alignment   : "
        f"{alignment:.3f}"
    )
    print(
        f"  Grounding Score      : "
        f"{quality['grounding_score']:.3f}"
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    assert utilization > 0.0, (
        "Evidence utilization should be greater than zero."
    )

    assert grounding > 0.0, (
        "Answer grounding should be greater than zero."
    )

    assert alignment == 1.0, (
        "Citation alignment should be 1.0 for the "
        "synthetically grounded answer."
    )

    assert quality["grounding_score"] > 0.0, (
        "Overall grounding score should be greater than zero."
    )

    for metric_name, value in quality.items():
        assert 0.0 <= value <= 1.0, (
            f"{metric_name} must be between 0 and 1, "
            f"got {value}"
        )

    print("\n" + "=" * 70)
    print(
        "PHASE 19.8.2 RAG EVIDENCE GROUNDING "
        "EVALUATION: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()