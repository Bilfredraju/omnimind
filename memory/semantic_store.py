import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dateutil.parser import isoparse

from rag.embeddings.embedder import EmbeddingModel
from memory.time_parser import MemoryTimeParser


class SemanticMemoryStore:
    """
    Persistent semantic memory store for OmniMind.

    Retrieval supports:

        1. Semantic similarity
        2. Memory importance
        3. Temporal filtering

    This allows OmniMind to answer historical queries such as:

        "What did I decide 3 months ago?"
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
        self.time_parser = MemoryTimeParser()

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

        except (
            json.JSONDecodeError,
            OSError,
        ):
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
            for a, b in zip(
                vector_a,
                vector_b,
            )
        )

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if (
            magnitude_a == 0.0
            or magnitude_b == 0.0
        ):
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _parse_timestamp(
        timestamp: str | None,
    ) -> datetime | None:

        if not timestamp:
            return None

        try:
            parsed = isoparse(timestamp)

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=timezone.utc
                )

            return parsed

        except (
            ValueError,
            TypeError,
        ):
            return None

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

        metadata = dict(metadata or {})

        memory_id = metadata.get(
            "memory_id",
            str(uuid.uuid4()),
        )

        created_at = metadata.get(
            "created_at",
            self._utc_now(),
        )

        metadata["memory_id"] = memory_id
        metadata["created_at"] = created_at

        if "importance" not in metadata:
            metadata["importance"] = 0.5

        if "type" not in metadata:
            metadata["type"] = "general"

        if "source" not in metadata:
            metadata["source"] = "conversation"

        embedding = self.embedder.encode_single(
            text
        )

        memory = {
            "memory_id": memory_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }

        self.memories.append(memory)
        self._save()

        return {
            "memory_id": memory_id,
            "text": text,
            "metadata": metadata,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        now: datetime | None = None,
    ) -> list[dict]:

        query = query.strip()

        if not query:
            return []

        if now is None:
            now = datetime.now(
                timezone.utc
            )

        if now.tzinfo is None:
            now = now.replace(
                tzinfo=timezone.utc
            )

        query_embedding = self.embedder.encode_single(
            query
        )

        time_filter = self.time_parser.parse(
            query=query,
            now=now,
        )

        results = []

        for memory in self.memories:

            metadata = memory.get(
                "metadata",
                {},
            )

            created_at = self._parse_timestamp(
                metadata.get("created_at")
            )

            # ------------------------------------------
            # Temporal filtering
            # ------------------------------------------
            if time_filter["has_time_filter"]:

                if created_at is None:
                    continue

                if created_at < time_filter["start"]:
                    continue

                if created_at > time_filter["end"]:
                    continue

            embedding = memory.get(
                "embedding",
                [],
            )

            similarity = self._cosine_similarity(
                query_embedding,
                embedding,
            )

            if similarity < min_score:
                continue

            importance = float(
                metadata.get(
                    "importance",
                    0.5,
                )
            )

            ranking_score = (
                similarity * 0.85
                + importance * 0.15
            )

            results.append(
                {
                    "memory_id": memory.get(
                        "memory_id",
                        metadata.get(
                            "memory_id",
                            "",
                        ),
                    ),
                    "text": memory.get(
                        "text",
                        "",
                    ),
                    "metadata": metadata,
                    "score": similarity,
                    "ranking_score": ranking_score,
                    "temporal_filter": (
                        time_filter["expression"]
                    ),
                }
            )

        results.sort(
            key=lambda item: item[
                "ranking_score"
            ],
            reverse=True,
        )

        return results[:top_k]

    def count(self) -> int:
        return len(self.memories)

    def clear(self) -> None:
        self.memories = []
        self._save()