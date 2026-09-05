from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dateutil.parser import isoparse

from memory.contradiction import MemoryContradictionEngine
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
    - Contradiction detection
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

        self.contradiction_engine = MemoryContradictionEngine()

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

        except (
            json.JSONDecodeError,
            OSError,
        ):
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
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(
        text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:

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
    # Contradiction detection
    # ------------------------------------------------------------------

    def _detect_contradiction(
        self,
        text: str,
        metadata: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:

        best_result = {
            "contradiction": False,
            "contradiction_with": None,
            "contradiction_confidence": 0.0,
            "contradiction_reason": None,
        }

        for existing in candidates:

            result = self.contradiction_engine.detect(
                new_text=text,
                existing_memory=existing,
            )

            if not result.get("contradiction"):
                continue

            confidence = float(
                result.get(
                    "confidence",
                    0.0,
                )
            )

            if confidence > best_result[
                "contradiction_confidence"
            ]:

                best_result = {
                    "contradiction": True,
                    "contradiction_with": existing.get(
                        "memory_id"
                    ),
                    "contradiction_confidence": confidence,
                    "contradiction_reason": result.get(
                        "reason"
                    ),
                }

        return best_result

    # ------------------------------------------------------------------
    # Add memory
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        if not text or not text.strip():
            raise ValueError(
                "Memory text cannot be empty."
            )

        text = text.strip()

        metadata = dict(metadata or {})

        # --------------------------------------------------------------
        # Generate embedding
        # --------------------------------------------------------------

        embedding = self.embedder.encode_single(
            text
        )

        # --------------------------------------------------------------
        # Build metadata
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
        # Find semantic duplicate candidate
        # --------------------------------------------------------------

        duplicate = self.deduplication_engine.find_duplicate(
            embedding=embedding,
            memories=self.memories,
        )

        # ==============================================================
        # CASE 1: SEMANTICALLY SIMILAR MEMORY EXISTS
        # ==============================================================

        if duplicate is not None:

            existing = duplicate["memory"]

            # ----------------------------------------------------------
            # 1A. Explicit update
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

                self.memories.append(
                    new_memory
                )

                self._save()

                return {
                    **new_memory,
                    "duplicate": False,
                    "updated": True,
                    "contradiction": False,
                    "duplicate_of": None,
                    "old_memory_id": update_result[
                        "old_memory_id"
                    ],
                    "duplicate_similarity": duplicate[
                        "similarity"
                    ],
                    "contradiction_with": None,
                    "contradiction_confidence": 0.0,
                    "contradiction_reason": None,
                }

            # ----------------------------------------------------------
            # 1B. CONTRADICTION CHECK
            #
            # This MUST happen before returning duplicate.
            #
            # Example:
            #
            # Existing:
            #   I decided to use Qdrant.
            #
            # New:
            #   I decided to use PostgreSQL.
            #
            # These may have high semantic similarity, but they
            # represent conflicting decisions.
            # ----------------------------------------------------------

            contradiction_result = (
                self._detect_contradiction(
                    text=text,
                    metadata=metadata,
                    candidates=[existing],
                )
            )

            if contradiction_result["contradiction"]:

                metadata["contradiction"] = True

                metadata["contradicts_memory_id"] = (
                    contradiction_result[
                        "contradiction_with"
                    ]
                )

                metadata["contradiction_confidence"] = (
                    contradiction_result[
                        "contradiction_confidence"
                    ]
                )

                metadata["contradiction_reason"] = (
                    contradiction_result[
                        "contradiction_reason"
                    ]
                )

                new_memory = {
                    "memory_id": metadata["memory_id"],
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata,
                }

                # IMPORTANT:
                # Contradictions preserve both memories.
                # The old memory remains current.

                self.memories.append(
                    new_memory
                )

                self._save()

                return {
                    **new_memory,
                    "duplicate": False,
                    "updated": False,
                    "contradiction": True,
                    "duplicate_of": None,
                    "old_memory_id": None,
                    "duplicate_similarity": duplicate[
                        "similarity"
                    ],
                    "contradiction_with": (
                        contradiction_result[
                            "contradiction_with"
                        ]
                    ),
                    "contradiction_confidence": (
                        contradiction_result[
                            "contradiction_confidence"
                        ]
                    ),
                    "contradiction_reason": (
                        contradiction_result[
                            "contradiction_reason"
                        ]
                    ),
                }

            # ----------------------------------------------------------
            # 1C. Genuine duplicate
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
                "contradiction": False,
                "duplicate_of": existing.get(
                    "memory_id"
                ),
                "old_memory_id": None,
                "duplicate_similarity": duplicate[
                    "similarity"
                ],
                "contradiction_with": None,
                "contradiction_confidence": 0.0,
                "contradiction_reason": None,
            }

        # ==============================================================
        # CASE 2: NO SEMANTIC DUPLICATE
        # ==============================================================

        # --------------------------------------------------------------
        # 2A. Explicit update without semantic duplicate
        # --------------------------------------------------------------

        if self.update_engine.is_update(
            new_text=text,
            existing_memory={
                "memory_id": None,
                "text": "",
                "metadata": metadata,
            },
        ):

            current_memories = [
                memory
                for memory in self.memories
                if (
                    memory.get(
                        "metadata",
                        {},
                    ).get("status")
                    == "current"
                    and memory.get(
                        "metadata",
                        {},
                    ).get("type")
                    == metadata.get("type")
                )
            ]

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

            if current_memories:

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

                self.memories.append(
                    new_memory
                )

                self._save()

                return {
                    **new_memory,
                    "duplicate": False,
                    "updated": True,
                    "contradiction": False,
                    "duplicate_of": None,
                    "old_memory_id": update_result[
                        "old_memory_id"
                    ],
                    "duplicate_similarity": None,
                    "contradiction_with": None,
                    "contradiction_confidence": 0.0,
                    "contradiction_reason": None,
                }

        # ==============================================================
        # CASE 3: CONTRADICTION CHECK
        # ==============================================================

        current_memories = [
            memory
            for memory in self.memories
            if memory.get(
                "metadata",
                {},
            ).get("status")
            == "current"
        ]

        # Prefer memories with the same type.
        same_type_memories = [
            memory
            for memory in current_memories
            if memory.get(
                "metadata",
                {},
            ).get("type")
            == metadata.get("type")
        ]

        candidates = (
            same_type_memories
            if same_type_memories
            else current_memories
        )

        contradiction_result = (
            self._detect_contradiction(
                text=text,
                metadata=metadata,
                candidates=candidates,
            )
        )

        # ==============================================================
        # CASE 4: STORE MEMORY
        # ==============================================================

        if contradiction_result["contradiction"]:

            metadata["contradiction"] = True

            metadata["contradicts_memory_id"] = (
                contradiction_result[
                    "contradiction_with"
                ]
            )

            metadata["contradiction_confidence"] = (
                contradiction_result[
                    "contradiction_confidence"
                ]
            )

            metadata["contradiction_reason"] = (
                contradiction_result[
                    "contradiction_reason"
                ]
            )

        else:

            metadata["contradiction"] = False

            metadata["contradicts_memory_id"] = None

            metadata["contradiction_confidence"] = 0.0

            metadata["contradiction_reason"] = None

        memory = {
            "memory_id": metadata["memory_id"],
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }

        self.memories.append(
            memory
        )

        self._save()

        return {
            **memory,
            "duplicate": False,
            "updated": False,
            "contradiction": contradiction_result[
                "contradiction"
            ],
            "duplicate_of": None,
            "old_memory_id": None,
            "duplicate_similarity": None,
            "contradiction_with": contradiction_result[
                "contradiction_with"
            ],
            "contradiction_confidence": contradiction_result[
                "contradiction_confidence"
            ],
            "contradiction_reason": contradiction_result[
                "contradiction_reason"
            ],
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

        query_embedding = self.embedder.encode_single(
            query
        )

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