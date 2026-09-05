from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConsolidatedMemoryStore:
    """
    Persistent JSON store for consolidated memory records.

    Consolidated memories are derived knowledge.
    Original semantic memories remain untouched.
    """

    def __init__(
        self,
        path: str | Path = "data/memory/consolidated_memories.json",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.memories: list[dict[str, Any]] = []

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:

        if not self.path.exists():
            self.memories = []
            return

        try:

            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if isinstance(data, list):
                self.memories = data
            else:
                self.memories = []

        except (
            json.JSONDecodeError,
            OSError,
        ):
            self.memories = []

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

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        consolidation: dict[str, Any],
    ) -> dict[str, Any]:

        if not consolidation:
            raise ValueError(
                "Consolidation cannot be empty."
            )

        consolidation_id = consolidation.get(
            "consolidation_id"
        )

        if not consolidation_id:
            raise ValueError(
                "Consolidation must contain consolidation_id."
            )

        # Replace an existing record with the same ID.
        existing_index = self._find_index(
            consolidation_id
        )

        if existing_index is not None:
            self.memories[existing_index] = consolidation
        else:
            self.memories.append(
                consolidation
            )

        self._save()

        return consolidation

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(
        self,
        consolidation_id: str,
    ) -> dict[str, Any] | None:

        for memory in self.memories:

            if memory.get(
                "consolidation_id"
            ) == consolidation_id:

                return memory

        return None

    # ------------------------------------------------------------------
    # Search by topic
    # ------------------------------------------------------------------

    def search_topic(
        self,
        topic: str,
    ) -> list[dict[str, Any]]:

        query = topic.lower().strip()

        if not query:
            return []

        return [
            memory
            for memory in self.memories
            if query in str(
                memory.get("topic", "")
            ).lower()
        ]

    # ------------------------------------------------------------------
    # Get all
    # ------------------------------------------------------------------

    def all(
        self,
    ) -> list[dict[str, Any]]:

        return list(self.memories)

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def count(self) -> int:

        return len(self.memories)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        consolidation_id: str,
    ) -> bool:

        index = self._find_index(
            consolidation_id
        )

        if index is None:
            return False

        self.memories.pop(index)

        self._save()

        return True

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:

        self.memories = []

        self._save()

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _find_index(
        self,
        consolidation_id: str,
    ) -> int | None:

        for index, memory in enumerate(
            self.memories
        ):

            if memory.get(
                "consolidation_id"
            ) == consolidation_id:

                return index

        return None