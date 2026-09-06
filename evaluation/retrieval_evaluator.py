"""
OmniMind Retrieval Evaluation

Evaluates:
- Semantic retrieval
- BM25 retrieval
- Hybrid RRF retrieval
- Hybrid RRF + Cross-Encoder reranking

Metrics:
- Hit Rate@K
- Recall@K
- Precision@K
- MRR
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.document_server import get_knowledge_base


@dataclass
class EvaluationCase:
    query: str
    relevant_pages: set[int]


EVALUATION_CASES = [
    EvaluationCase(
        query="What datasets were used to evaluate the RAG models?",
        relevant_pages={1, 4},
    ),
    EvaluationCase(
        query="What is retrieval augmented generation?",
        relevant_pages={1, 2},
    ),
    EvaluationCase(
        query="What are the main components of a RAG system?",
        relevant_pages={1, 2, 3},
    ),
    EvaluationCase(
        query="How are documents split into chunks?",
        relevant_pages={2, 3},
    ),
    EvaluationCase(
        query="What embedding model was used?",
        relevant_pages={1, 2},
    ),
]


def _metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Safely extract result metadata."""

    metadata = result.get("metadata")

    if isinstance(metadata, dict):
        return metadata

    return {}


def _page(result: dict[str, Any]) -> int | None:
    """Extract page number from either top-level or nested metadata."""

    value = result.get("page_number")

    if value is None:
        value = result.get("page")

    if value is None:
        metadata = _metadata(result)
        value = metadata.get("page_number")

        if value is None:
            value = metadata.get("page")

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk(result: dict[str, Any]) -> int | None:
    """Extract chunk index from either top-level or nested metadata."""

    value = result.get("chunk_index")

    if value is None:
        value = result.get("chunk")

    if value is None:
        metadata = _metadata(result)
        value = metadata.get("chunk_index")

        if value is None:
            value = metadata.get("chunk")

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def hit_rate(
    results: list[dict[str, Any]],
    relevant_pages: set[int],
) -> float:
    """Return 1 if at least one relevant page was retrieved."""

    retrieved_pages = {
        page
        for result in results
        if (page := _page(result)) is not None
    }

    return float(bool(retrieved_pages & relevant_pages))


def recall_at_k(
    results: list[dict[str, Any]],
    relevant_pages: set[int],
) -> float:
    """Calculate page-level Recall@K."""

    if not relevant_pages:
        return 0.0

    retrieved_pages = {
        page
        for result in results
        if (page := _page(result)) is not None
    }

    return len(retrieved_pages & relevant_pages) / len(relevant_pages)


def precision_at_k(
    results: list[dict[str, Any]],
    relevant_pages: set[int],
) -> float:
    """Calculate page-level Precision@K."""

    if not results:
        return 0.0

    relevant_count = sum(
        1
        for result in results
        if (page := _page(result)) is not None
        and page in relevant_pages
    )

    return relevant_count / len(results)


def reciprocal_rank(
    results: list[dict[str, Any]],
    relevant_pages: set[int],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""

    for index, result in enumerate(results, start=1):
        page = _page(result)

        if page is not None and page in relevant_pages:
            return 1.0 / index

    return 0.0


def evaluate_retriever(
    name: str,
    retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
    cases: list[EvaluationCase],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate one retrieval strategy."""

    hits: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []

    for case in cases:
        results = retrieve_fn(case.query, top_k)

        hits.append(
            hit_rate(
                results,
                case.relevant_pages,
            )
        )

        recalls.append(
            recall_at_k(
                results,
                case.relevant_pages,
            )
        )

        precisions.append(
            precision_at_k(
                results,
                case.relevant_pages,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                results,
                case.relevant_pages,
            )
        )

    total = len(cases)

    return {
        "name": name,
        "cases": total,
        "hit_rate": (
            sum(hits) / total
            if total
            else 0.0
        ),
        "recall_at_k": (
            sum(recalls) / total
            if total
            else 0.0
        ),
        "precision_at_k": (
            sum(precisions) / total
            if total
            else 0.0
        ),
        "mrr": (
            sum(reciprocal_ranks) / total
            if total
            else 0.0
        ),
    }


def print_result(
    result: dict[str, Any],
    top_k: int,
) -> None:
    """Print evaluation result."""

    print(f"\n{result['name']}")
    print("-" * len(result["name"]))

    print(
        f"Cases        : "
        f"{result['cases']}"
    )

    print(
        f"Hit Rate@{top_k}   : "
        f"{result['hit_rate']:.3f}"
    )

    print(
        f"Recall@{top_k}     : "
        f"{result['recall_at_k']:.3f}"
    )

    print(
        f"Precision@{top_k}  : "
        f"{result['precision_at_k']:.3f}"
    )

    print(
        f"MRR          : "
        f"{result['mrr']:.3f}"
    )


def main() -> None:
    top_k = 5

    print("=" * 60)
    print("OmniMind Retrieval Evaluation")
    print("=" * 60)

    kb = get_knowledge_base()

    def semantic(
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        return kb.semantic_search(query, k)

    def bm25(
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        results = kb.bm25.search(query, k)

        return [
            {
                "text": result.get("text", ""),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
            }
            for result in results
        ]

    def hybrid_rrf(
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        return kb.hybrid_search(query, k)

    def hybrid_reranked(
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        return kb.search(query, k)

    evaluation_results = [
        evaluate_retriever(
            "Semantic Retrieval",
            semantic,
            EVALUATION_CASES,
            top_k,
        ),
        evaluate_retriever(
            "BM25 Retrieval",
            bm25,
            EVALUATION_CASES,
            top_k,
        ),
        evaluate_retriever(
            "Hybrid RRF",
            hybrid_rrf,
            EVALUATION_CASES,
            top_k,
        ),
        evaluate_retriever(
            "Hybrid RRF + Cross-Encoder",
            hybrid_reranked,
            EVALUATION_CASES,
            top_k,
        ),
    ]

    for result in evaluation_results:
        print_result(
            result,
            top_k,
        )

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()