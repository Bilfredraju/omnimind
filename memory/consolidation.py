from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class MemoryConsolidationEngine:
    """
    Converts a group of related memories into a higher-level
    consolidated knowledge record.

    Important design principle:
    Consolidation NEVER deletes or modifies the original memories.

    Original memories remain the source of truth.
    The consolidated record is a derived summary.
    """

    # ------------------------------------------------------------------
    # Main consolidation
    # ------------------------------------------------------------------

    def consolidate(
        self,
        memories: list[dict[str, Any]],
        topic: str | None = None,
    ) -> dict[str, Any]:

        if not memories:
            raise ValueError(
                "Cannot consolidate an empty memory list."
            )

        ordered_memories = self._sort_by_time(
            memories
        )

        memory_ids = [
            memory.get("memory_id")
            for memory in ordered_memories
            if memory.get("memory_id")
        ]

        current_memories = [
            memory
            for memory in ordered_memories
            if self._is_current(memory)
        ]

        historical_memories = [
            memory
            for memory in ordered_memories
            if not self._is_current(memory)
        ]

        current_memory = self._select_current_memory(
            current_memories,
            ordered_memories,
        )

        resolved_topic = (
            topic.strip()
            if topic
            else self._infer_topic(
                ordered_memories
            )
        )

        summary = self._build_summary(
            resolved_topic,
            ordered_memories,
            current_memory,
        )

        return {
            "consolidation_id": self._generate_id(
                memory_ids
            ),
            "topic": resolved_topic,
            "memory_ids": memory_ids,
            "summary": summary,
            "current_memory_id": (
                current_memory.get("memory_id")
                if current_memory
                else None
            ),
            "historical_memory_ids": [
                memory.get("memory_id")
                for memory in historical_memories
                if memory.get("memory_id")
            ],
            "memory_count": len(
                ordered_memories
            ),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    # ------------------------------------------------------------------
    # Sort memories
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_by_time(
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        def timestamp(memory: dict[str, Any]) -> str:
            return str(
                memory.get(
                    "metadata",
                    {},
                ).get(
                    "created_at",
                    "",
                )
            )

        return sorted(
            memories,
            key=timestamp,
        )

    # ------------------------------------------------------------------
    # Current status
    # ------------------------------------------------------------------

    @staticmethod
    def _is_current(
        memory: dict[str, Any],
    ) -> bool:

        metadata = memory.get(
            "metadata",
            {},
        )

        return (
            metadata.get(
                "status",
                "current",
            )
            == "current"
        )

    # ------------------------------------------------------------------
    # Select current memory
    # ------------------------------------------------------------------

    @staticmethod
    def _select_current_memory(
        current_memories: list[dict[str, Any]],
        ordered_memories: list[dict[str, Any]],
    ) -> dict[str, Any] | None:

        if current_memories:

            return current_memories[-1]

        if ordered_memories:

            return ordered_memories[-1]

        return None

    # ------------------------------------------------------------------
    # Topic inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_topic(
        memories: list[dict[str, Any]],
    ) -> str:

        if not memories:
            return "Unknown topic"

        # Prefer an explicitly supplied type/topic-like metadata field.
        for memory in memories:

            metadata = memory.get(
                "metadata",
                {},
            )

            topic = metadata.get(
                "topic"
            )

            if topic:
                return str(topic)

        # Fall back to the most important memory.
        best_memory = max(
            memories,
            key=lambda memory: float(
                memory.get(
                    "metadata",
                    {},
                ).get(
                    "importance",
                    0.0,
                )
            ),
        )

        text = str(
            best_memory.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            return "Unknown topic"

        # Keep the inferred topic compact.
        words = text.split()

        if len(words) <= 8:
            return text

        return " ".join(
            words[:8]
        ) + "..."

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        topic: str,
        memories: list[dict[str, Any]],
        current_memory: dict[str, Any] | None,
    ) -> str:

        if not memories:
            return f"No memories available for {topic}."

        statements = []

        for memory in memories:

            text = str(
                memory.get(
                    "text",
                    "",
                )
            ).strip()

            if text:
                statements.append(
                    text
                )

        if not statements:
            return f"No textual memories available for {topic}."

        current_text = ""

        if current_memory:

            current_text = str(
                current_memory.get(
                    "text",
                    "",
                )
            ).strip()

        if len(statements) == 1:

            return (
                f"{topic}: "
                f"{statements[0]}"
            )

        historical_count = max(
            0,
            len(statements) - 1,
        )

        if current_text:

            return (
                f"{topic}: "
                f"{historical_count} historical "
                f"memory/memories were recorded. "
                f"Current knowledge: "
                f"{current_text}"
            )

        return (
            f"{topic}: "
            f"{len(statements)} related memories "
            f"were recorded over time."
        )

    # ------------------------------------------------------------------
    # Deterministic consolidation ID
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_id(
        memory_ids: list[str],
    ) -> str:

        if not memory_ids:
            return "consolidation-empty"

        joined = "-".join(
            sorted(memory_ids)
        )

        # Stable lightweight identifier.
        checksum = sum(
            ord(character)
            for character in joined
        )

        return (
            f"consolidation-"
            f"{checksum}"
        )