import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Keyword-based BM25 retriever."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        self.tokenized_documents = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple lowercase word tokenizer."""

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return the top BM25 results."""

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            results.append(
                {
                    "score": float(scores[index]),
                    "text": self.chunks[index]["text"],
                    "metadata": self.chunks[index]["metadata"],
                }
            )

        return results