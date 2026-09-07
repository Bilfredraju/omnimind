"""
Deterministic evaluation of multi-query retrieval.

Compares baseline hybrid retrieval against multi-query hybrid retrieval
using the real MCP document knowledge base.

No Groq or LLM calls are required.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from mcp_servers.document_server import DocumentKnowledgeBase
from rag.retrieval.query_expansion import QueryExpander


DEFAULT_QUERY = (
    "What datasets were used to evaluate the RAG models?"
)


def _get_page(result: dict[str, Any]) -> Any:
    metadata = result.get("metadata")

    if isinstance(metadata, dict):
        if metadata.get("page") is not None:
            return metadata.get("page")

        if metadata.get("page_number") is not None:
            return metadata.get("page_number")

    return result.get("page")


def _get_chunk_id(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata")

    if isinstance(metadata, dict):
        chunk_id = metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

    chunk_id = result.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    result_id = result.get("result_id")

    if result_id:
        return str(result_id)

    return None


def _evaluate_pages(
    results: list[dict[str, Any]],
    expected_pages: set[Any],
) -> dict[str, float]:
    if not results:
        return {
            "hit_rate": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "mrr": 0.0,
        }

    retrieved_pages = [
        _get_page(result)
        for result in results
    ]

    retrieved_pages = [
        page
        for page in retrieved_pages
        if page is not None
    ]

    hits = [
        page
        for page in retrieved_pages
        if page in expected_pages
    ]

    hit_rate = (
        1.0
        if hits
        else 0.0
    )

    recall = (
        len(set(hits)) / len(expected_pages)
        if expected_pages
        else 0.0
    )

    precision = (
        len(hits) / len(retrieved_pages)
        if retrieved_pages
        else 0.0
    )

    mrr = 0.0

    for rank, page in enumerate(
        retrieved_pages,
        start=1,
    ):
        if page in expected_pages:
            mrr = 1.0 / rank
            break

    return {
        "hit_rate": hit_rate,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
    }


def evaluate(
    kb: DocumentKnowledgeBase,
    query: str = DEFAULT_QUERY,
    top_k: int = 5,
) -> dict[str, Any]:

    # Pages known to contain the answer for the evaluation query.
    expected_pages = {
        2,
        4,
        6,
        18,
    }

    # ---------------------------------------------------------
    # Baseline hybrid retrieval
    # ---------------------------------------------------------

    baseline = kb.hybrid_search(
        query,
        top_k=top_k,
    )

    baseline_metrics = _evaluate_pages(
        baseline[:top_k],
        expected_pages,
    )

    # ---------------------------------------------------------
    # Multi-query retrieval
    # ---------------------------------------------------------

    expanded_queries = QueryExpander(
        max_queries=3
    ).expand(query)

    multi_query = kb.multi_query_search(
        query,
        top_k=top_k,
    )

    multi_metrics = _evaluate_pages(
        multi_query[:top_k],
        expected_pages,
    )

    # ---------------------------------------------------------
    # Unique chunk counts
    # ---------------------------------------------------------

    baseline_chunks = {
        chunk_id
        for chunk_id in (
            _get_chunk_id(result)
            for result in baseline[:top_k]
        )
        if chunk_id
    }

    multi_chunks = {
        chunk_id
        for chunk_id in (
            _get_chunk_id(result)
            for result in multi_query[:top_k]
        )
        if chunk_id
    }

    return {
        "query": query,
        "top_k": top_k,
        "expanded_queries": expanded_queries,
        "query_count": len(expanded_queries),
        "baseline": {
            **baseline_metrics,
            "result_count": len(baseline[:top_k]),
            "unique_chunks": len(baseline_chunks),
        },
        "multi_query": {
            **multi_metrics,
            "result_count": len(multi_query[:top_k]),
            "unique_chunks": len(multi_chunks),
        },
        "delta": {
            "hit_rate": (
                multi_metrics["hit_rate"]
                - baseline_metrics["hit_rate"]
            ),
            "recall": (
                multi_metrics["recall"]
                - baseline_metrics["recall"]
            ),
            "precision": (
                multi_metrics["precision"]
                - baseline_metrics["precision"]
            ),
            "mrr": (
                multi_metrics["mrr"]
                - baseline_metrics["mrr"]
            ),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print("OMNIMIND — PHASE 19.9.4 MULTI-QUERY RETRIEVAL EVALUATION")
    print("=" * 72)

    print()
    print(f"Query: {report['query']}")
    print(f"Top K: {report['top_k']}")
    print(f"Query variations: {report['query_count']}")

    print()
    print("-" * 72)
    print("QUERY VARIATIONS")
    print("-" * 72)

    for index, query in enumerate(
        report["expanded_queries"],
        start=1,
    ):
        print(f"{index}. {query}")

    print()
    print("-" * 72)
    print("BASELINE — HYBRID RRF")
    print("-" * 72)

    baseline = report["baseline"]

    print(
        f"Hit Rate:       {baseline['hit_rate']:.3f}"
    )
    print(
        f"Recall:         {baseline['recall']:.3f}"
    )
    print(
        f"Precision:      {baseline['precision']:.3f}"
    )
    print(
        f"MRR:            {baseline['mrr']:.3f}"
    )
    print(
        f"Results:        {baseline['result_count']}"
    )
    print(
        f"Unique chunks:  {baseline['unique_chunks']}"
    )

    print()
    print("-" * 72)
    print("MULTI-QUERY — HYBRID RRF")
    print("-" * 72)

    multi = report["multi_query"]

    print(
        f"Hit Rate:       {multi['hit_rate']:.3f}"
    )
    print(
        f"Recall:         {multi['recall']:.3f}"
    )
    print(
        f"Precision:      {multi['precision']:.3f}"
    )
    print(
        f"MRR:            {multi['mrr']:.3f}"
    )
    print(
        f"Results:        {multi['result_count']}"
    )
    print(
        f"Unique chunks:  {multi['unique_chunks']}"
    )

    print()
    print("-" * 72)
    print("DELTA — MULTI-QUERY MINUS BASELINE")
    print("-" * 72)

    delta = report["delta"]

    print(
        f"Hit Rate:       {delta['hit_rate']:+.3f}"
    )
    print(
        f"Recall:         {delta['recall']:+.3f}"
    )
    print(
        f"Precision:      {delta['precision']:+.3f}"
    )
    print(
        f"MRR:            {delta['mrr']:+.3f}"
    )

    print()
    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)


def main() -> None:
    document = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "sample.pdf"
    )

    kb = DocumentKnowledgeBase(document)

    report = evaluate(kb)

    print_report(report)


if __name__ == "__main__":
    main()