from rag.embeddings.embedder import EmbeddingModel
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.search import SemanticRetriever


class HybridRetriever:
    """
    Combine semantic retrieval and BM25 keyword retrieval.
    """

    def __init__(
        self,
        chunks: list[dict],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
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
            semantic_weight + keyword_weight - 1.0
        ) > 1e-6:
            raise ValueError(
                "Weights must add up to 1."
            )

        self.chunks = chunks

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        self.semantic_retriever = SemanticRetriever()

        self.bm25_retriever = BM25Retriever(
            chunks
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        semantic_results = (
            self.semantic_retriever.search(
                query,
                top_k=len(self.chunks),
            )
        )

        keyword_results = (
            self.bm25_retriever.search(
                query,
                top_k=len(self.chunks),
            )
        )

        combined = {}

        # Normalize semantic scores using rank
        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            key = (
                result["metadata"]["source"],
                result["metadata"]["page"],
                result["metadata"]["chunk_index"],
            )

            semantic_score = 1 / rank

            combined.setdefault(
                key,
                {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                },
            )

            combined[key][
                "semantic_score"
            ] = semantic_score

        # Normalize keyword scores using rank
        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):
            key = (
                result["metadata"]["source"],
                result["metadata"]["page"],
                result["metadata"]["chunk_index"],
            )

            keyword_score = 1 / rank

            combined.setdefault(
                key,
                {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                },
            )

            combined[key][
                "keyword_score"
            ] = keyword_score

        # Calculate final hybrid score
        results = []

        for result in combined.values():

            final_score = (
                self.semantic_weight
                * result["semantic_score"]
                +
                self.keyword_weight
                * result["keyword_score"]
            )

            result["hybrid_score"] = final_score

            results.append(result)

        results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        return results[:top_k]