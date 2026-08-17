from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved documents using a cross-encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
    ):
        self.model_name = model_name

        print(
            f"Loading reranker model: {model_name}"
        )

        self.model = CrossEncoder(model_name)

        print(
            "Reranker model loaded successfully."
        )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank documents based on query-document relevance.
        """

        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=True,
        )

        reranked = []

        for document, score in zip(
            documents,
            scores,
        ):
            result = document.copy()

            result["rerank_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]