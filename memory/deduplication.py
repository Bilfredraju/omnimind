from __future__ import annotations

from typing import Any


class MemoryDeduplicationEngine:
    """
    Determines whether a newly created memory is semantically
    similar enough to an existing memory to be considered a duplicate.
    """

    def __init__(self, threshold: float = 0.88):
        self.threshold = threshold

    def find_duplicate(
        self,
        embedding: list[float],
        memories: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Find the most similar existing memory.

        Returns the existing memory plus similarity information
        when the similarity reaches the configured threshold.
        """

        best_memory = None
        best_similarity = -1.0

        for memory in memories:
            existing_embedding = memory.get("embedding")

            if not existing_embedding:
                continue

            similarity = self._cosine_similarity(
                embedding,
                existing_embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_memory = memory

        if (
            best_memory is not None
            and best_similarity >= self.threshold
        ):
            return {
                "memory": best_memory,
                "similarity": round(best_similarity, 4),
            }

        return None

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        if not vector_a or not vector_b:
            return 0.0

        if len(vector_a) != len(vector_b):
            return 0.0

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = sum(
            a * a
            for a in vector_a
        ) ** 0.5

        magnitude_b = sum(
            b * b
            for b in vector_b
        ) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )