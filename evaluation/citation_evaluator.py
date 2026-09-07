"""
Unified citation, grounding, and claim-level RAG evaluation.

Phase 19.8.4

This evaluator combines:
- Citation quality
- Evidence grounding
- Claim-level grounding

It intentionally avoids LLM generation so evaluation can run
deterministically without consuming Groq quota.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from evaluation.metrics import (
    citation_coverage,
    citation_quality,
    grounding_quality,
    claim_grounding_quality,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_QUERY = "What datasets were used to evaluate the RAG models?"

DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_text(value: Any) -> str:
    """Convert a value to clean text."""

    if value is None:
        return ""

    return str(value).strip()


def _extract_rag_results(
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract RAG retrieval results from graph state."""

    results = state.get("rag_results") or []

    if not isinstance(results, list):
        return []

    return [
        result
        for result in results
        if isinstance(result, dict)
    ]


def _extract_source_records(
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract source records and normalize them to dictionaries.

    Some evaluation paths may expose source information as strings.
    Those values are converted into structured records so the metric
    functions always receive the expected input shape.
    """

    sources = state.get("sources") or []

    if not isinstance(sources, list):
        return []

    normalized: List[Dict[str, Any]] = []

    for source in sources:

        if isinstance(source, dict):
            normalized.append(source)
            continue

        if isinstance(source, str):
            normalized.append(
                {
                    "citation_id": None,
                    "source": source,
                    "text": source,
                }
            )

    return normalized


def _extract_evidence_records(
    rag_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert RAG results into structured evidence records.

    Every returned record is guaranteed to be a dictionary.
    """

    evidence_records: List[Dict[str, Any]] = []

    for result in rag_results:

        if not isinstance(result, dict):
            continue

        text = _safe_text(
            result.get("text")
        )

        if not text:
            continue

        citation = result.get("citation")

        citation_id = None

        if isinstance(citation, dict):
            citation_id = citation.get(
                "citation_id"
            )

        if not citation_id:
            citation_id = result.get(
                "citation_id"
            )

        evidence_records.append(
            {
                "text": text,
                "citation_id": citation_id,
                "result_id": result.get("result_id"),
                "chunk_id": result.get("chunk_id"),
                "document_id": result.get("document_id"),
                "source": result.get("source"),
                "document_name": result.get("document_name"),
                "page": result.get("page"),
                "metadata": (
                    result.get("metadata")
                    if isinstance(
                        result.get("metadata"),
                        dict,
                    )
                    else {}
                ),
                "citation": citation,
            }
        )

    return evidence_records


def _extract_evidence_texts(
    evidence_records: List[Dict[str, Any]],
) -> List[str]:
    """Extract non-empty evidence text."""

    return [
        record["text"]
        for record in evidence_records
        if isinstance(record, dict)
        and _safe_text(record.get("text"))
    ]


def _build_expected_citation_ids(
    evidence_records: List[Dict[str, Any]],
) -> List[str]:
    """
    Build expected citation IDs from retrieved evidence.

    For this deterministic evaluation, all retrieved citations are
    treated as the expected citation universe.

    This is intentionally conservative and deterministic. A later
    evaluation phase can introduce relevance-aware expected citations.
    """

    ids: List[str] = []

    for record in evidence_records:

        if not isinstance(record, dict):
            continue

        citation_id = record.get(
            "citation_id"
        )

        if (
            citation_id
            and citation_id not in ids
        ):
            ids.append(
                str(citation_id)
            )

    return ids


def _ensure_source_citations(
    source_records: List[Dict[str, Any]],
    evidence_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ensure source records contain structured citation IDs.

    The real MCP retrieval results already contain citation objects.
    This helper keeps source records aligned with those citations.
    """

    if not source_records:
        return list(evidence_records)

    normalized: List[Dict[str, Any]] = []

    for index, source in enumerate(
        source_records
    ):

        if not isinstance(source, dict):
            continue

        record = dict(source)

        citation_id = record.get(
            "citation_id"
        )

        if not citation_id:

            citation = record.get(
                "citation"
            )

            if isinstance(
                citation,
                dict,
            ):
                citation_id = citation.get(
                    "citation_id"
                )

        if (
            not citation_id
            and index < len(evidence_records)
        ):

            evidence = evidence_records[index]

            if isinstance(
                evidence,
                dict,
            ):
                citation_id = evidence.get(
                    "citation_id"
                )

                if not record.get("citation"):
                    record["citation"] = evidence.get(
                        "citation"
                    )

                if not record.get("text"):
                    record["text"] = evidence.get(
                        "text"
                    )

        if citation_id:
            record["citation_id"] = str(
                citation_id
            )

        normalized.append(
            record
        )

    return normalized


# ---------------------------------------------------------------------------
# Deterministic evidence sentence extraction
# ---------------------------------------------------------------------------

def _first_sentence(text: str) -> str:
    """
    Return the first reasonably complete sentence from evidence text.

    This keeps the deterministic evaluation answer compact while
    ensuring that the generated fixture remains directly grounded
    in retrieved document content.
    """

    text = _safe_text(text)

    if not text:
        return ""

    normalized = " ".join(
        text.split()
    )

    if not normalized:
        return ""

    # Prefer a normal sentence boundary.
    for separator in (". ", "? ", "! "):

        if separator in normalized:

            sentence = normalized.split(
                separator,
                1,
            )[0].strip()

            if sentence:
                return sentence + separator.strip()

    # If there is no sentence boundary, use the complete text.
    return normalized


def _build_deterministic_answer(
    evidence_records: List[Dict[str, Any]],
    max_evidence: int = 3,
) -> str:
    """
    Build a deterministic answer directly from retrieved evidence.

    No LLM is used.

    Only the first few evidence records are included so the evaluation
    remains readable while still testing multiple citations and claims.
    """

    answer_parts: List[str] = []

    for record in evidence_records[:max_evidence]:

        if not isinstance(record, dict):
            continue

        text = _safe_text(
            record.get("text")
        )

        if not text:
            continue

        sentence = _first_sentence(
            text
        )

        if not sentence:
            continue

        citation_id = _safe_text(
            record.get("citation_id")
        )

        if citation_id:
            answer_parts.append(
                f"{sentence} {citation_id}"
            )
        else:
            answer_parts.append(
                sentence
            )

    if answer_parts:
        return " ".join(
            answer_parts
        )

    return (
        "No textual evidence was retrieved for this query."
    )


# ---------------------------------------------------------------------------
# Unified evaluation
# ---------------------------------------------------------------------------

def evaluate_state(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate a completed OmniMind evaluation state.

    Returns a unified report containing:

    Citation:
    - coverage
    - precision
    - recall
    - correctness
    - completeness
    - F1

    Grounding:
    - evidence utilization
    - answer grounding
    - citation-evidence alignment
    - grounding score

    Claim-level grounding:
    - claim count
    - supported claims
    - unsupported claims
    - supported ratio
    - unsupported ratio
    - average claim support
    - claim grounding score
    """

    answer = _safe_text(
        state.get("final_answer")
        or state.get("answer")
        or ""
    )

    # ---------------------------------------------------------------
    # Retrieve evaluation inputs
    # ---------------------------------------------------------------

    rag_results = _extract_rag_results(
        state
    )

    raw_source_records = _extract_source_records(
        state
    )

    evidence_records = _extract_evidence_records(
        rag_results
    )

    evidence_texts = _extract_evidence_texts(
        evidence_records
    )

    # Normalize source records so the metrics always receive
    # dictionaries instead of strings.
    source_records = _ensure_source_citations(
        raw_source_records,
        evidence_records,
    )

    expected_citation_ids = _build_expected_citation_ids(
        evidence_records
    )

    # ---------------------------------------------------------------
    # Citation evaluation
    # ---------------------------------------------------------------

    coverage = citation_coverage(
        answer,
        len(expected_citation_ids),
    )

    citations = citation_quality(
        answer,
        source_records,
        expected_citation_ids,
    )

    # ---------------------------------------------------------------
    # Grounding evaluation
    # ---------------------------------------------------------------

    grounding = grounding_quality(
        answer,
        source_records,
        evidence_texts,
        evidence_records,
    )

    # ---------------------------------------------------------------
    # Claim-level grounding
    # ---------------------------------------------------------------

    claims = claim_grounding_quality(
        answer,
        evidence_texts,
        threshold=0.5,
    )

    # ---------------------------------------------------------------
    # Unified score
    # ---------------------------------------------------------------

    citation_f1 = float(
        citations.get(
            "citation_f1",
            0.0,
        )
    )

    grounding_score = float(
        grounding.get(
            "grounding_score",
            0.0,
        )
    )

    claim_grounding_score = float(
        claims.get(
            "claim_grounding_score",
            0.0,
        )
    )

    unified_score = (
        citation_f1
        + grounding_score
        + claim_grounding_score
    ) / 3.0

    # ---------------------------------------------------------------
    # Return complete report
    # ---------------------------------------------------------------

    return {
        "query": state.get(
            "query"
        ),

        "answer": answer,

        "retrieval": {
            "result_count": len(
                rag_results
            ),

            "evidence_count": len(
                evidence_records
            ),

            "source_count": len(
                source_records
            ),

            "expected_citation_ids": (
                expected_citation_ids
            ),
        },

        "citation": {
            "coverage": coverage,
            **citations,
        },

        "grounding": grounding,

        "claim_grounding": claims,

        "unified_rag_quality": unified_score,
    }


# ---------------------------------------------------------------------------
# Deterministic evaluation from real retrieval
# ---------------------------------------------------------------------------

def evaluate_real_retrieval(
    query: str = DEFAULT_QUERY,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Run real document retrieval and evaluate a deterministic,
    evidence-grounded answer.

    No LLM call is made.

    The answer is constructed directly from the retrieved evidence
    so that the evaluator measures citation, grounding, and
    claim-level quality without introducing an intentionally
    unsupported claim.
    """

    from mcp_servers.document_server import (
        get_knowledge_base,
    )

    knowledge_base = get_knowledge_base()

    # ---------------------------------------------------------------
    # Real document retrieval
    # ---------------------------------------------------------------

    results = knowledge_base.search(
        query,
        top_k=top_k,
    )

    rag_results = [
        result
        for result in results
        if isinstance(result, dict)
    ]

    # ---------------------------------------------------------------
    # Extract evidence
    # ---------------------------------------------------------------

    evidence_records = _extract_evidence_records(
        rag_results
    )

    # ---------------------------------------------------------------
    # Construct deterministic evidence-grounded answer
    # ---------------------------------------------------------------

    answer = _build_deterministic_answer(
        evidence_records,
        max_evidence=min(
            3,
            len(evidence_records),
        ),
    )

    # ---------------------------------------------------------------
    # Build synthetic evaluation state
    # ---------------------------------------------------------------

    synthetic_state = {
        "query": query,

        "final_answer": answer,

        "rag_results": rag_results,

        "sources": [
            record
            for record in evidence_records
            if isinstance(record, dict)
            and record.get("citation_id")
        ],
    }

    # ---------------------------------------------------------------
    # Run unified evaluation
    # ---------------------------------------------------------------

    return evaluate_state(
        synthetic_state
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_report(
    report: Dict[str, Any],
) -> None:
    """Print a readable unified evaluation report."""

    print(
        "\n"
        + "=" * 72
    )

    print(
        "OMNIMIND — UNIFIED RAG QUALITY EVALUATION"
    )

    print(
        "=" * 72
    )

    # ---------------------------------------------------------------
    # Query
    # ---------------------------------------------------------------

    print("\nQuery:")

    print(
        report.get("query")
    )

    # ---------------------------------------------------------------
    # Answer
    # ---------------------------------------------------------------

    print("\nAnswer:")

    print(
        report.get("answer")
    )

    # ---------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------

    retrieval = report.get(
        "retrieval",
        {},
    )

    print("\nRetrieval:")

    print(
        f"  Results:             "
        f"{retrieval.get('result_count', 0)}"
    )

    print(
        f"  Evidence:            "
        f"{retrieval.get('evidence_count', 0)}"
    )

    print(
        f"  Sources:             "
        f"{retrieval.get('source_count', 0)}"
    )

    print(
        f"  Expected citations:  "
        f"{retrieval.get('expected_citation_ids', [])}"
    )

    # ---------------------------------------------------------------
    # Citation quality
    # ---------------------------------------------------------------

    citation = report.get(
        "citation",
        {},
    )

    print("\nCitation Quality:")

    print(
        f"  Coverage:            "
        f"{citation.get('coverage', 0.0):.3f}"
    )

    print(
        f"  Precision:           "
        f"{citation.get('citation_precision', 0.0):.3f}"
    )

    print(
        f"  Recall:              "
        f"{citation.get('citation_recall', 0.0):.3f}"
    )

    print(
        f"  Correctness:         "
        f"{citation.get('citation_correctness', 0.0):.3f}"
    )

    print(
        f"  Completeness:        "
        f"{citation.get('citation_completeness', 0.0):.3f}"
    )

    print(
        f"  F1:                  "
        f"{citation.get('citation_f1', 0.0):.3f}"
    )

    # ---------------------------------------------------------------
    # Grounding quality
    # ---------------------------------------------------------------

    grounding = report.get(
        "grounding",
        {},
    )

    print("\nGrounding Quality:")

    print(
        f"  Evidence utilization: "
        f"{grounding.get('evidence_utilization', 0.0):.3f}"
    )

    print(
        f"  Answer grounding:      "
        f"{grounding.get('answer_grounding', 0.0):.3f}"
    )

    print(
        f"  Citation alignment:    "
        f"{grounding.get('citation_evidence_alignment', 0.0):.3f}"
    )

    print(
        f"  Grounding score:       "
        f"{grounding.get('grounding_score', 0.0):.3f}"
    )

    # ---------------------------------------------------------------
    # Claim-level grounding
    # ---------------------------------------------------------------

    claims = report.get(
        "claim_grounding",
        {},
    )

    print("\nClaim-Level Grounding:")

    print(
        f"  Claims:                "
        f"{claims.get('claim_count', 0)}"
    )

    print(
        f"  Supported claims:      "
        f"{claims.get('supported_claims', 0)}"
    )

    print(
        f"  Unsupported claims:    "
        f"{claims.get('unsupported_claims', 0)}"
    )

    print(
        f"  Supported ratio:       "
        f"{claims.get('supported_claim_ratio', 0.0):.3f}"
    )

    print(
        f"  Unsupported ratio:     "
        f"{claims.get('unsupported_claim_ratio', 0.0):.3f}"
    )

    print(
        f"  Average claim support: "
        f"{claims.get('average_claim_support', 0.0):.3f}"
    )

    print(
        f"  Claim grounding:       "
        f"{claims.get('claim_grounding_score', 0.0):.3f}"
    )

    # ---------------------------------------------------------------
    # Unified score
    # ---------------------------------------------------------------

    print("\nUnified RAG Quality:")

    print(
        f"  Score:                 "
        f"{report.get('unified_rag_quality', 0.0):.3f}"
    )

    print(
        "=" * 72
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    report = evaluate_real_retrieval()

    print_report(
        report
    )