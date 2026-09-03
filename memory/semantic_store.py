import json
import math
from pathlib import Path

from rag.embeddings.embedder import EmbeddingModel


class SemanticMemoryStore:
    """
    Local semantic memory store.

    Stores memories together with their embedding vectors
    and retrieves relevant memories using semantic similarity.
    """

    def __init__(
        self,
        path: str = "data/memory/semantic_memories.json",
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedder = EmbeddingModel()

        self.memories = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if isinstance(data, list):
                return data

        except (json.JSONDecodeError, OSError):
            pass

        return []

    def _save(self) -> None:
        with self.path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.memories,
                file,
                indent=2,
                ensure_ascii=False,
            )

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

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )

    def add(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> dict:
        text = text.strip()

        if not text:
            raise ValueError(
                "Memory text cannot be empty."
            )

        embedding = self.embedder.encode_single(
            text
        )

        memory = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }

        self.memories.append(memory)

        self._save()

        return {
            "text": text,
            "metadata": metadata or {},
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        query = query.strip()

        if not query:
            return []

        if not self.memories:
            return []

        query_embedding = self.embedder.encode_single(
            query
        )

        scored = []

        for memory in self.memories:
            score = self._cosine_similarity(
                query_embedding,
                memory["embedding"],
            )

            if score >= min_score:
                scored.append(
                    {
                        "text": memory["text"],
                        "metadata": memory.get(
                            "metadata",
                            {},
                        ),
                        "score": score,
                    }
                )

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored[:max(1, top_k)]

    def count(self) -> int:
        return len(self.memories)

    def clear(self) -> None:
        self.memories = []
        self._save()