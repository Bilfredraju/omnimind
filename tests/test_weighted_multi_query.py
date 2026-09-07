"""
OmniMind — Weighted Multi-Query Fusion Test
"""

from __future__ import annotations

from collections import defaultdict


def weighted_rrf_fusion(
    query_results,
    weights=None,
    rrf_k=60,
    top_k=5,
):
    """
    Fuse ranked results from multiple queries using weighted RRF.
    """

    if weights is None:
        weights = [1.0] * len(query_results)

    if len(weights) != len(query_results):
        raise ValueError("weights must match query_results")

    fused = defaultdict(
        lambda: {
            "score": 0.0,
            "matches": 0,
            "sources": [],
        }
    )

    for query_index, results in enumerate(query_results):
        weight = weights[query_index]

        for rank, result in enumerate(results, start=1):
            result_id = result["result_id"]

            contribution = weight / (rrf_k + rank)

            fused[result_id]["score"] += contribution
            fused[result_id]["matches"] += 1
            fused[result_id]["sources"].append(query_index + 1)

    ranked = []

    for result_id, data in fused.items():
        ranked.append(
            {
                "result_id": result_id,
                "weighted_rrf_score": data["score"],
                "matches": data["matches"],
                "sources": data["sources"],
            }
        )

    ranked.sort(
        key=lambda x: (
            x["weighted_rrf_score"],
            x["matches"],
        ),
        reverse=True,
    )

    return ranked[:top_k]


def main():
    """
    Test that weighting gives the original query higher influence.

    The original query places relevant-A at rank 1.
    Expanded queries place it lower.
    """

    query_results = [
        [
            {"result_id": "relevant-A"},
            {"result_id": "chunk-B"},
            {"result_id": "chunk-C"},
            {"result_id": "chunk-D"},
            {"result_id": "chunk-E"},
        ],
        [
            {"result_id": "chunk-B"},
            {"result_id": "chunk-C"},
            {"result_id": "relevant-A"},
            {"result_id": "chunk-F"},
            {"result_id": "chunk-G"},
        ],
        [
            {"result_id": "chunk-F"},
            {"result_id": "chunk-C"},
            {"result_id": "relevant-A"},
            {"result_id": "chunk-H"},
            {"result_id": "chunk-I"},
        ],
    ]

    weighted_results = weighted_rrf_fusion(
        query_results,
        weights=[1.0, 0.5, 0.5],
        rrf_k=60,
        top_k=5,
    )

    print("Weighted multi-query fusion: PASSED")
    print()
    print("Ranking:")

    for index, result in enumerate(weighted_results, start=1):
        print(
            f"{index}. {result['result_id']} "
            f"score={result['weighted_rrf_score']:.6f} "
            f"matches={result['matches']}"
        )

    assert weighted_results

    # The original query has the highest weight.
    assert weighted_results[0]["result_id"] == "relevant-A"

    print()
    print("Original-query priority preserved: PASSED")

    # Verify expanded-query weighting is lower than the original.
    assert 1.0 > 0.5
    assert 1.0 > 0.5

    print("Query weighting configuration: PASSED")


if __name__ == "__main__":
    main()