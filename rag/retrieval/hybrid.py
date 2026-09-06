from __future__ import annotations

from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.search import SemanticRetriever


class HybridRetriever:
    """
    Hybrid document retriever combining:

    1. Semantic vector retrieval
    2. BM25 keyword retrieval
    3. Reciprocal Rank Fusion (RRF)

    RRF avoids directly comparing incompatible BM25 and
    embedding score scales.
    """

    def __init__(
        self,
        chunks: list[dict],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError(
                "semantic_weight must be between 0 and 1."
            )

        if not 0 <= keyword_weight <= 1:
            raise ValueError(
                "keyword_weight must be between 0 and 1."
            )

        if abs(
            semantic_weight
            + keyword_weight
            - 1.0
        ) > 1e-6:
            raise ValueError(
                "Weights must add up to 1."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than 0."
            )

        self.chunks = chunks or []

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        self.semantic_retriever = SemanticRetriever()

        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

    @staticmethod
    def _result_key(
        result: dict,
        fallback_index: int,
    ) -> str:
        """
        Return a stable identity for a retrieved chunk.
        """
        if result.get("result_id"):
            return str(result["result_id"])

        metadata = result.get(
            "metadata",
            {},
        )

        chunk_id = metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

        return (
            f"{metadata.get('source', 'unknown')}:"
            f"{metadata.get('page', 0)}:"
            f"{metadata.get('chunk_index', fallback_index)}"
        )

    def _rrf_score(
        self,
        rank: int,
        weight: float,
    ) -> float:
        """
        Calculate weighted Reciprocal Rank Fusion score.
        """
        return weight / (
            self.rrf_k + rank
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve documents using semantic + BM25 search.

        Returns results containing:

            result_id
            text
            metadata
            semantic_score
            keyword_score
            semantic_rank
            keyword_rank
            hybrid_score
            retrieval_sources
        """
        if top_k <= 0:
            return []

        if not query or not query.strip():
            return []

        if not self.chunks:
            return []

        # Retrieve a sufficiently large candidate pool.
        candidate_k = max(
            top_k * 5,
            20,
        )

        candidate_k = min(
            candidate_k,
            len(self.chunks),
        )

        semantic_results = (
            self.semantic_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        keyword_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        combined: dict[str, dict] = {}

        # ---------------------------------------------------------
        # Semantic retrieval
        # ---------------------------------------------------------

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            key = self._result_key(
                result,
                rank,
            )

            entry = combined.setdefault(
                key,
                {
                    "result_id": key,
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "retrieval_sources": [],
                },
            )

            entry["semantic_score"] = float(
                result.get("score", 0.0)
            )

            entry["semantic_rank"] = rank

            if "semantic" not in entry[
                "retrieval_sources"
            ]:
                entry["retrieval_sources"].append(
                    "semantic"
                )

        # ---------------------------------------------------------
        # BM25 retrieval
        # ---------------------------------------------------------

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):
            key = self._result_key(
                result,
                rank,
            )

            entry = combined.setdefault(
                key,
                {
                    "result_id": key,
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "retrieval_sources": [],
                },
            )

            entry["keyword_score"] = float(
                result.get(
                    "normalized_score",
                    result.get(
                        "score",
                        0.0,
                    ),
                )
            )

            entry["keyword_rank"] = rank

            if "bm25" not in entry[
                "retrieval_sources"
            ]:
                entry["retrieval_sources"].append(
                    "bm25"
                )

        # ---------------------------------------------------------
        # RRF fusion
        # ---------------------------------------------------------

        results: list[dict] = []

        for entry in combined.values():

            semantic_rank = entry[
                "semantic_rank"
            ]

            keyword_rank = entry[
                "keyword_rank"
            ]

            semantic_rrf = 0.0
            keyword_rrf = 0.0

            if semantic_rank is not None:
                semantic_rrf = self._rrf_score(
                    semantic_rank,
                    self.semantic_weight,
                )

            if keyword_rank is not None:
                keyword_rrf = self._rrf_score(
                    keyword_rank,
                    self.keyword_weight,
                )

            hybrid_score = (
                semantic_rrf
                + keyword_rrf
            )

            entry["semantic_rrf_score"] = (
                semantic_rrf
            )

            entry["keyword_rrf_score"] = (
                keyword_rrf
            )

            entry["hybrid_score"] = (
                hybrid_score
            )

            entry["retrieval_source_count"] = (
                len(
                    entry[
                        "retrieval_sources"
                    ]
                )
            )

            results.append(entry)

        # Prefer:
        # 1. hybrid RRF score
        # 2. results appearing in both systems
        # 3. semantic rank
        results.sort(
            key=lambda item: (
                item["hybrid_score"],
                item["retrieval_source_count"],
                -(
                    item["semantic_rank"]
                    or 10**9
                ),
            ),
            reverse=True,
        )

        return results[:top_k]

    def close(self) -> None:
        """
        Close resources owned by the semantic retriever.
        """
        self.semantic_retriever.close()