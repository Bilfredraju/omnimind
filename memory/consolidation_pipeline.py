from __future__ import annotations

from typing import Any

from memory.timeline import MemoryTimelineEngine
from memory.consolidation import MemoryConsolidationEngine
from memory.consolidated_store import ConsolidatedMemoryStore


class MemoryConsolidationPipeline:
    """
    End-to-end pipeline for converting related semantic memories
    into persistent consolidated knowledge.

    Flow:

        Memories
            ↓
        Timeline
            ↓
        Consolidation
            ↓
        Persistent Store

    Original memories are never modified or deleted.
    """

    def __init__(
        self,
        store: ConsolidatedMemoryStore | None = None,
        timeline_engine: MemoryTimelineEngine | None = None,
        consolidation_engine: MemoryConsolidationEngine | None = None,
    ):
        self.store = (
            store
            if store is not None
            else ConsolidatedMemoryStore()
        )

        self.timeline_engine = (
            timeline_engine
            if timeline_engine is not None
            else MemoryTimelineEngine()
        )

        self.consolidation_engine = (
            consolidation_engine
            if consolidation_engine is not None
            else MemoryConsolidationEngine()
        )

    # ------------------------------------------------------------------
    # Consolidate memories
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

        # --------------------------------------------------------------
        # Step 1 — Build chronological timeline
        # --------------------------------------------------------------

        timeline = self.timeline_engine.build_timeline(
            memories
        )

        # --------------------------------------------------------------
        # Step 2 — Consolidate knowledge
        # --------------------------------------------------------------

        consolidation = (
            self.consolidation_engine.consolidate(
                memories,
                topic=topic,
            )
        )

        # --------------------------------------------------------------
        # Step 3 — Attach timeline
        # --------------------------------------------------------------

        consolidation["timeline"] = timeline

        # --------------------------------------------------------------
        # Step 4 — Persist consolidated knowledge
        # --------------------------------------------------------------

        persisted = self.store.add(
            consolidation
        )

        return persisted

    # ------------------------------------------------------------------
    # Retrieve consolidated knowledge
    # ------------------------------------------------------------------

    def get(
        self,
        consolidation_id: str,
    ) -> dict[str, Any] | None:

        return self.store.get(
            consolidation_id
        )

    # ------------------------------------------------------------------
    # Search consolidated knowledge
    # ------------------------------------------------------------------

    def search_topic(
        self,
        topic: str,
    ) -> list[dict[str, Any]]:

        return self.store.search_topic(
            topic
        )

    # ------------------------------------------------------------------
    # Format result
    # ------------------------------------------------------------------

    def format_result(
        self,
        consolidation: dict[str, Any],
    ) -> str:

        topic = consolidation.get(
            "topic",
            "Unknown topic",
        )

        summary = consolidation.get(
            "summary",
            "",
        )

        timeline = consolidation.get(
            "timeline",
            [],
        )

        timeline_text = (
            self.timeline_engine.format_timeline(
                timeline
            )
        )

        return (
            f"Topic: {topic}\n"
            f"Summary: {summary}\n\n"
            f"{timeline_text}"
        )