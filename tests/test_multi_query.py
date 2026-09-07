"""
Deterministic tests for multi-query retrieval.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.retrieval.multi_query import MultiQueryRetriever


class FakeHybridRetriever:
    """
    Deterministic fake hybrid retriever used for testing.
    """

    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append(query)

        # Return overlapping results deliberately.
        #
        # This lets us verify that MultiQueryRetriever:
        #   1. executes multiple queries
        #   2. deduplicates results
        #   3. combines their RRF scores
        #   4. ranks the strongest shared result first

        return [
            {
                "result_id": "chunk-A",
                "text": "Natural Questions dataset",
                "metadata": {"page": 2},
            },
            {
                "result_id": "chunk-B",
                "text": "MS MARCO dataset",
                "metadata": {"page": 3},
            },
            {
                "result_id": "chunk-C",
                "text": "TriviaQA dataset",
                "metadata": {"page": 4},
            },
        ]


def main():
    hybrid = FakeHybridRetriever()

    retriever = MultiQueryRetriever(
        hybrid_retriever=hybrid,
        max_queries=3,
        rrf_k=60,
    )

    results = retriever.retrieve(
        "What datasets were used to evaluate the RAG models?",
        top_k=3,
    )

    # ---------------------------------------------------------
    # Multiple queries were actually executed.
    # ---------------------------------------------------------

    assert len(hybrid.calls) == 3

    # ---------------------------------------------------------
    # top_k is an upper bound.
    # The fake retriever provides 3 unique results.
    # ---------------------------------------------------------

    assert 0 < len(results) <= 3

    # ---------------------------------------------------------
    # Results must be deduplicated.
    # ---------------------------------------------------------

    result_ids = [
        result["result_id"]
        for result in results
    ]

    assert len(result_ids) == len(set(result_ids))

    # ---------------------------------------------------------
    # chunk-A appears at rank 1 for every query.
    # Therefore it should have the strongest RRF score.
    # ---------------------------------------------------------

    chunk_a = next(
        result
        for result in results
        if result["result_id"] == "chunk-A"
    )

    assert chunk_a["multi_query_matches"] == 3
    assert chunk_a["multi_query_score"] > 0
    assert chunk_a["multi_query_rank"] == 1

    # ---------------------------------------------------------
    # Every result contains the expected multi-query metadata.
    # ---------------------------------------------------------

    for result in results:
        assert "multi_query_score" in result
        assert "multi_query_rank" in result
        assert "multi_query_sources" in result
        assert "multi_query_matches" in result

        assert result["multi_query_score"] > 0
        assert result["multi_query_rank"] > 0
        assert result["multi_query_sources"]

        assert isinstance(
            result["multi_query_sources"],
            list,
        )

    # ---------------------------------------------------------
    # Final ranks must be sequential.
    # ---------------------------------------------------------

    ranks = [
        result["multi_query_rank"]
        for result in results
    ]

    assert ranks == list(range(1, len(results) + 1))

    # ---------------------------------------------------------
    # RRF score should decrease or remain equal with rank.
    # ---------------------------------------------------------

    scores = [
        result["multi_query_score"]
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    print("Multi-query retrieval: PASSED")
    print(f"Queries executed: {len(hybrid.calls)}")
    print(f"Results returned: {len(results)}")
    print(f"Result ranking: {result_ids}")

    print("\nQuery variations executed:")
    for index, query in enumerate(
        hybrid.calls,
        start=1,
    ):
        print(f"{index}. {query}")


if __name__ == "__main__":
    main()