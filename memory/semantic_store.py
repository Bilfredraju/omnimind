from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dateutil.parser import isoparse

from memory.importance import MemoryImportanceEngine
from memory.time_parser import MemoryTimeParser
from rag.embeddings.embedder import EmbeddingModel


class SemanticMemoryStore:
    """
    Persistent semantic memory store for OmniMind.

    Features:
    - Persistent JSON storage
    - Semantic embeddings
    - Importance-aware ranking
    - Temporal filtering
    - Stable memory IDs
    """

    def __init__(
        self,
        path: str = "data/memory/semantic_memories.json",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.embedder = EmbeddingModel()
        self.time_parser = MemoryTimeParser()
        self.importance_engine = MemoryImportanceEngine()

        self.memories: list[dict[str, Any]] = []

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if not self.path.exists():
            self.memories = []
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.memories = json.load(f)

        except (json.JSONDecodeError, OSError):
            self.memories = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                self.memories,
                f,
                indent=2,
                ensure_ascii=False,
            )

    # ------------------------------------------------------------------
    # Add memory
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        metadata = dict(metadata or {})

        # Stable memory metadata.
        metadata.setdefault(
            "memory_id",
            str(uuid.uuid4()),
        )

        metadata.setdefault(
            "created_at",
            datetime.now(timezone.utc).isoformat(),
        )

        metadata.setdefault(
            "source",
            "conversation",
        )

        metadata.setdefault(
            "type",
            "general",
        )

        # Calculate intelligent importance.
        importance = self.importance_engine.calculate(
            text=text,
            metadata=metadata,
        )

        metadata["importance"] = importance

        embedding = self.embedder.encode_single(text)

        memory = {
            "memory_id": metadata["memory_id"],
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }

        self.memories.append(memory)
        self._save()

        return memory

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:

        temporal = self.time_parser.parse(query, now=now)

        query_embedding = self.embedder.encode_single(query)

        candidates = []

        for memory in self.memories:

            metadata = memory.get("metadata", {})

            # ----------------------------------------------------------
            # Temporal filtering
            # ----------------------------------------------------------

            if temporal["has_time_filter"]:

                created_at = metadata.get("created_at")

                if not created_at:
                    continue

                try:
                    memory_time = isoparse(created_at)

                    if memory_time.tzinfo is None:
                        memory_time = memory_time.replace(
                            tzinfo=timezone.utc
                        )

                except (ValueError, TypeError):
                    continue

                start = temporal["start"]
                end = temporal["end"]

                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)

                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)

                if not (start <= memory_time <= end):
                    continue

            # ----------------------------------------------------------
            # Semantic similarity
            # ----------------------------------------------------------

            embedding = memory.get("embedding")

            if not embedding:
                continue

            similarity = sum(
                a * b
                for a, b in zip(
                    query_embedding,
                    embedding,
                )
            )

            if similarity < min_score:
                continue

            # ----------------------------------------------------------
            # Importance-aware ranking
            # ----------------------------------------------------------

            importance = float(
                metadata.get("importance", 0.0)
            )

            ranking_score = (
                similarity * 0.75
                + importance * 0.25
            )

            candidates.append(
                {
                    "memory_id": memory.get("memory_id"),
                    "text": memory.get("text", ""),
                    "metadata": metadata,
                    "score": round(similarity, 4),
                    "ranking_score": round(
                        ranking_score,
                        4,
                    ),
                    "temporal_filter": temporal.get(
                        "expression"
                    ),
                }
            )

        # Highest combined score first.
        candidates.sort(
            key=lambda x: x["ranking_score"],
            reverse=True,
        )

        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self.memories)

    def clear(self):
        self.memories = []
        self._save()