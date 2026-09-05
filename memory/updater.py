from __future__ import annotations

from typing import Any


class MemoryUpdateEngine:
    """
    Determines whether a new memory explicitly supersedes
    an existing memory.

    Important:
    A decision statement by itself is NOT considered an update.
    Explicit change/reversal language is required.
    """

    UPDATE_SIGNALS = [
        "changed my decision",
        "change my decision",
        "changed the decision",
        "change the decision",
        "changed my mind",
        "change my mind",
        "we changed our mind",
        "we changed our decision",
        "we changed the decision",
        "instead",
        "rather than",
        "rather,",
        "no longer",
        "not using",
        "stop using",
        "stopped using",
        "i'll use instead",
        "i will use instead",
        "i am going to use instead",
        "i'm going to use instead",
        "we'll use instead",
        "we will use instead",
        "we are going to use instead",
        "we're going to use instead",
        "i've decided to switch",
        "i decided to switch",
        "we decided to switch",
        "switch from",
        "switch to",
        "replace qdrant",
        "replace postgres",
        "replace postgresql",
    ]

    def is_update(
        self,
        new_text: str,
        existing_memory: dict[str, Any],
    ) -> bool:
        """
        Return True only when the new memory contains
        explicit language indicating that an existing decision
        or preference has been changed/superseded.
        """

        text = new_text.lower().strip()

        return any(
            signal in text
            for signal in self.UPDATE_SIGNALS
        )

    def apply_update(
        self,
        new_memory: dict[str, Any],
        existing_memory: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Mark the old memory as superseded while preserving it
        for historical recall.
        """

        existing_id = existing_memory.get(
            "memory_id"
        )

        new_id = new_memory.get(
            "memory_id"
        )

        existing_metadata = existing_memory.setdefault(
            "metadata",
            {},
        )

        new_metadata = new_memory.setdefault(
            "metadata",
            {},
        )

        existing_metadata["status"] = "superseded"
        existing_metadata["superseded_by"] = new_id
        existing_metadata["superseded_at"] = (
            new_metadata.get("created_at")
        )

        new_metadata["status"] = "current"
        new_metadata["updates_memory_id"] = existing_id

        return {
            "updated": True,
            "old_memory_id": existing_id,
            "new_memory_id": new_id,
        }