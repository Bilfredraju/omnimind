from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dateutil.parser import isoparse

from memory.deduplication import MemoryDeduplicationEngine
from memory.importance import MemoryImportanceEngine
from memory.time_parser import MemoryTimeParser
from memory.updater import MemoryUpdateEngine
from rag.embeddings.embedder import EmbeddingModel


class SemanticMemoryStore:
    """
    Persistent semantic memory store for OmniMind.

    Features:
    - Persistent JSON storage
    - Semantic embeddings
    - Importance-aware ranking
    - Temporal filtering
    - Semantic deduplication
    - Memory versioning / updates
    - Stable memory IDs
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
        self.importance_engine = MemoryImportanceEngine()

        self.deduplication_engine = MemoryDeduplicationEngine(
            threshold=0.88
        )

        self.update_engine = MemoryUpdateEngine()

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
            with open(
                self.path,
                "r",
                encoding="utf-8",
            ) as f:
                self.memories = json.load(f)

        except (json.JSONDecodeError, OSError):
            self.memories = []

    def _save(self):
        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.memories,
                f,
                indent=2,
                ensure_ascii=False,
            )

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(
        text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build safe default metadata for a memory.
        """

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

        metadata.setdefault(
            "status",
            "current",
        )

        return metadata

    # ------------------------------------------------------------------
    # Add memory
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add a memory while handling:

        1. Embedding generation
        2. Importance scoring
        3. Semantic duplicates
        4. Explicit memory updates
        5. Historical versioning
        """

        metadata = dict(metadata or {})

        # --------------------------------------------------------------
        # Generate embedding
        # --------------------------------------------------------------

        embedding = self.embedder.encode_single(text)

        # --------------------------------------------------------------
        # Prepare metadata
        # --------------------------------------------------------------

        metadata = self._build_metadata(
            text=text,
            metadata=metadata,
        )

        # --------------------------------------------------------------
        # Calculate importance
        # --------------------------------------------------------------

        importance = self.importance_engine.calculate(
            text=text,
            metadata=metadata,
        )

        metadata["importance"] = importance

        # --------------------------------------------------------------
        # Find semantically similar existing memory
        # --------------------------------------------------------------

        duplicate = self.deduplication_engine.find_duplicate(
            embedding=embedding,
            memories=self.memories,
        )

        # --------------------------------------------------------------
        # Case 1: Semantically similar memory exists
        # --------------------------------------------------------------

        if duplicate is not None:

            existing = duplicate["memory"]

            # ----------------------------------------------------------
            # Check whether the new statement explicitly updates it.
            # ----------------------------------------------------------

            if self.update_engine.is_update(
                new_text=text,
                existing_memory=existing,
            ):

                new_memory = {
                    "memory_id": metadata["memory_id"],
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata,
                }

                update_result = self.update_engine.apply_update(
                    new_memory=new_memory,
                    existing_memory=existing,
                )

                self.memories.append(new_memory)
                self._save()

                return {
                    **new_memory,
                    "duplicate": False,
                    "updated": True,
                    "duplicate_of": None,
                    "old_memory_id": update_result[
                        "old_memory_id"
                    ],
                    "duplicate_similarity": duplicate[
                        "similarity"
                    ],
                }

            # ----------------------------------------------------------
            # Same semantic meaning = duplicate
            # ----------------------------------------------------------

            return {
                "memory_id": existing.get(
                    "memory_id"
                ),
                "text": existing.get(
                    "text",
                    "",
                ),
                "embedding": existing.get(
                    "embedding",
                    [],
                ),
                "metadata": existing.get(
                    "metadata",
                    {},
                ),
                "duplicate": True,
                "updated": False,
                "duplicate_of": existing.get(
                    "memory_id"
                ),
                "old_memory_id": None,
                "duplicate_similarity": duplicate[
                    "similarity"
                ],
            }

        # --------------------------------------------------------------
        # Case 2: Semantically different memory
        #
        # An explicit update can still occur here.
        #
        # Example:
        #
        #   Existing:
        #       I decided to use Qdrant.
        #
        #   New:
        #       I changed my decision. I'll use PostgreSQL instead.
        #
        # These statements may not have enough semantic similarity
        # to trigger the duplicate detector.
        # --------------------------------------------------------------

        if self.update_engine.is_update(
            new_text=text,
            existing_memory={
                "memory_id": None,
                "text": "",
                "metadata": metadata,
            },
        ):

            # ----------------------------------------------------------
            # Find current memories with the same memory type.
            #
            # This prevents an update to a decision from accidentally
            # superseding an unrelated preference, goal, etc.
            # ----------------------------------------------------------

            current_memories = [
                memory
                for memory in self.memories
                if (
                    memory.get(
                        "metadata",
                        {},
                    ).get(
                        "status",
                        "current",
                    )
                    == "current"
                )
                and (
                    memory.get(
                        "metadata",
                        {},
                    ).get(
                        "type"
                    )
                    == metadata.get(
                        "type"
                    )
                )
            ]

            # ----------------------------------------------------------
            # Select the most recent current memory.
            # ----------------------------------------------------------

            if current_memories:

                current_memories.sort(
                    key=lambda memory: memory.get(
                        "metadata",
                        {},
                    ).get(
                        "created_at",
                        "",
                    ),
                    reverse=True,
                )

                existing = current_memories[0]

                new_memory = {
                    "memory_id": metadata["memory_id"],
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata,
                }

                update_result = self.update_engine.apply_update(
                    new_memory=new_memory,
                    existing_memory=existing,
                )

                self.memories.append(new_memory)
                self._save()

                return {
                    **new_memory,
                    "duplicate": False,
                    "updated": True,
                    "duplicate_of": None,
                    "old_memory_id": update_result[
                        "old_memory_id"
                    ],
                    "duplicate_similarity": None,
                }

        # --------------------------------------------------------------
        # Case 3: Completely new memory
        # --------------------------------------------------------------

        memory = {
            "memory_id": metadata["memory_id"],
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }

        self.memories.append(memory)
        self._save()

        return {
            **memory,
            "duplicate": False,
            "updated": False,
            "duplicate_of": None,
            "old_memory_id": None,
            "duplicate_similarity": None,
        }

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

        temporal = self.time_parser.parse(
            query,
            now=now,
        )

        query_embedding = self.embedder.encode_single(query)

        candidates = []

        for memory in self.memories:

            metadata = memory.get(
                "metadata",
                {},
            )

            # ----------------------------------------------------------
            # Temporal filtering
            # ----------------------------------------------------------

            if temporal["has_time_filter"]:

                created_at = metadata.get(
                    "created_at"
                )

                if not created_at:
                    continue

                try:
                    memory_time = isoparse(
                        created_at
                    )

                    if memory_time.tzinfo is None:
                        memory_time = memory_time.replace(
                            tzinfo=timezone.utc
                        )

                except (
                    ValueError,
                    TypeError,
                ):
                    continue

                start = temporal["start"]
                end = temporal["end"]

                if start.tzinfo is None:
                    start = start.replace(
                        tzinfo=timezone.utc
                    )

                if end.tzinfo is None:
                    end = end.replace(
                        tzinfo=timezone.utc
                    )

                if not (
                    start
                    <= memory_time
                    <= end
                ):
                    continue

            # ----------------------------------------------------------
            # Semantic similarity
            # ----------------------------------------------------------

            embedding = memory.get(
                "embedding"
            )

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
                metadata.get(
                    "importance",
                    0.0,
                )
            )

            ranking_score = (
                similarity * 0.75
                + importance * 0.25
            )

            candidates.append(
                {
                    "memory_id": memory.get(
                        "memory_id"
                    ),
                    "text": memory.get(
                        "text",
                        "",
                    ),
                    "metadata": metadata,
                    "score": round(
                        similarity,
                        4,
                    ),
                    "ranking_score": round(
                        ranking_score,
                        4,
                    ),
                    "temporal_filter": temporal.get(
                        "expression"
                    ),
                }
            )

        # --------------------------------------------------------------
        # Highest combined score first
        # --------------------------------------------------------------

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