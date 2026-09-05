from __future__ import annotations

from typing import Any

from memory.consolidated_store import ConsolidatedMemoryStore


class ConsolidatedMemoryRetrievalEngine:
    """
    Retrieves persistent consolidated knowledge.

    This layer does not modify memories.
    It focuses only on finding relevant consolidated records.
    """

    def __init__(
        self,
        store: ConsolidatedMemoryStore | None = None,
    ):
        self.store = (
            store
            if store is not None
            else ConsolidatedMemoryStore()
        )

    # ------------------------------------------------------------------
    # Retrieve by topic
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not query or not query.strip():
            return []

        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        candidates = []

        for memory in self.store.all():

            topic = str(
                memory.get("topic", "")
            )

            summary = str(
                memory.get("summary", "")
            )

            searchable_text = (
                f"{topic} {summary}"
            ).lower()

            memory_terms = self._tokenize(
                searchable_text
            )

            overlap = query_terms.intersection(
                memory_terms
            )

            if not overlap:
                continue

            score = len(overlap) / len(
                query_terms
            )

            # Topic matches receive additional weight.
            topic_terms = self._tokenize(
                topic
            )

            topic_overlap = (
                query_terms.intersection(
                    topic_terms
                )
            )

            if topic_overlap:
                score += (
                    0.25
                    * len(topic_overlap)
                    / len(query_terms)
                )

            candidates.append(
                {
                    "consolidation": memory,
                    "score": round(
                        min(score, 1.0),
                        4,
                    ),
                }
            )

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Retrieve best result
    # ------------------------------------------------------------------

    def retrieve_best(
        self,
        query: str,
    ) -> dict[str, Any] | None:

        results = self.retrieve(
            query,
            top_k=1,
        )

        if not results:
            return None

        return results[0]

    # ------------------------------------------------------------------
    # Convert result into useful context
    # ------------------------------------------------------------------

    @staticmethod
    def build_context(
        results: list[dict[str, Any]],
    ) -> str:

        if not results:
            return ""

        lines = [
            "Consolidated Memory Context:"
        ]

        for result in results:

            memory = result[
                "consolidation"
            ]

            topic = memory.get(
                "topic",
                "Unknown topic",
            )

            summary = memory.get(
                "summary",
                "",
            )

            current_id = memory.get(
                "current_memory_id"
            )

            lines.append(
                f"- Topic: {topic}"
            )

            lines.append(
                f"  Summary: {summary}"
            )

            if current_id:
                lines.append(
                    f"  Current memory: {current_id}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:

        return {
            token
            for token in (
                text.lower()
                .replace(",", " ")
                .replace(".", " ")
                .replace("?", " ")
                .replace("!", " ")
                .replace(":", " ")
                .replace(";", " ")
                .split()
            )
            if len(token) > 2
        }