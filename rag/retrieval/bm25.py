from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based BM25 retriever.

    Designed for technical/document retrieval where exact terms,
    acronyms, identifiers, and product names can be highly relevant.
    """

    TOKEN_PATTERN = re.compile(
        r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*"
    )

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks or []

        self.tokenized_documents = [
            self._tokenize(
                chunk.get("text", "")
            )
            for chunk in self.chunks
        ]

        if self.tokenized_documents:
            self.bm25 = BM25Okapi(
                self.tokenized_documents
            )
        else:
            self.bm25 = None

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """
        Tokenize text while preserving useful technical identifiers.

        Examples:
            GPT-OSS-20B -> ["gpt-oss-20b"]
            PostgreSQL  -> ["postgresql"]
            RAG          -> ["rag"]
            Qdrant       -> ["qdrant"]
        """
        if not text:
            return []

        return [
            token.lower()
            for token in cls.TOKEN_PATTERN.findall(text)
        ]

    @staticmethod
    def _normalize_scores(
        scores: list[float],
    ) -> list[float]:
        """
        Min-max normalize BM25 scores into [0, 1].

        If all scores are identical, return 0 for all scores.
        """
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum <= minimum:
            return [0.0 for _ in scores]

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]

    @staticmethod
    def _result_key(
        chunk: dict,
        index: int,
    ) -> str:
        """
        Build a stable result identity.

        Prefer the deterministic chunk_id introduced during
        document ingestion. Fall back to index for compatibility
        with older chunks.
        """
        metadata = chunk.get("metadata", {})

        chunk_id = metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

        return f"chunk-index-{index}"

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Return the top BM25 results.

        Results contain:
            - score
            - bm25_score
            - normalized_score
            - text
            - metadata
            - result_id
        """
        if top_k <= 0:
            return []

        if not query or not query.strip():
            return []

        if not self.chunks or self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        score_list = [
            float(score)
            for score in scores
        ]

        normalized_scores = self._normalize_scores(
            score_list
        )

        ranked_indices = sorted(
            range(len(score_list)),
            key=lambda index: (
                score_list[index],
                -index,
            ),
            reverse=True,
        )[:top_k]

        results: list[dict] = []

        for index in ranked_indices:
            chunk = self.chunks[index]
            metadata = chunk.get(
                "metadata",
                {},
            )

            bm25_score = score_list[index]
            normalized_score = normalized_scores[index]

            results.append(
                {
                    "result_id": self._result_key(
                        chunk,
                        index,
                    ),
                    "score": bm25_score,
                    "bm25_score": bm25_score,
                    "normalized_score": normalized_score,
                    "text": chunk.get(
                        "text",
                        "",
                    ),
                    "metadata": metadata,
                }
            )

        return results

    def close(self) -> None:
        """
        BM25 does not own external resources.
        Kept for retriever interface compatibility.
        """
        return None