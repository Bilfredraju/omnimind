from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memory.consolidated_store import ConsolidatedMemoryStore
from memory.consolidated_time_parser import (
    ConsolidatedTimeQueryParser,
)
from memory.temporal_retrieval import (
    TemporalConsolidatedRetrievalEngine,
)


class TemporalMemoryPipeline:
    """
    End-to-end temporal memory retrieval pipeline.

    Flow:

        Natural-language query
                ↓
        Temporal query parser
                ↓
        Temporal boundaries
                ↓
        Consolidated retrieval
                ↓
        Memory context

    This pipeline is read-only.
    It never modifies or deletes memories.
    """

    def __init__(
        self,
        store: ConsolidatedMemoryStore | None = None,
        time_parser: ConsolidatedTimeQueryParser | None = None,
        retrieval_engine: (
            TemporalConsolidatedRetrievalEngine | None
        ) = None,
    ):
        self.store = (
            store
            if store is not None
            else ConsolidatedMemoryStore()
        )

        self.time_parser = (
            time_parser
            if time_parser is not None
            else ConsolidatedTimeQueryParser()
        )

        self.retrieval_engine = (
            retrieval_engine
            if retrieval_engine is not None
            else TemporalConsolidatedRetrievalEngine(
                self.store
            )
        )

    # ------------------------------------------------------------------
    # Main query
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        now: datetime | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:

        if not query or not query.strip():
            return {
                "query": query,
                "temporal_intent": {},
                "results": [],
                "context": "",
            }

        # --------------------------------------------------------------
        # Step 1 — Parse temporal intent
        # --------------------------------------------------------------

        temporal_intent = self.time_parser.parse(
            query,
            now=now,
        )

        # --------------------------------------------------------------
        # Step 2 — Choose retrieval mode
        # --------------------------------------------------------------

        if temporal_intent.get(
            "is_current"
        ):

            results = (
                self.retrieval_engine.retrieve_current(
                    query,
                    top_k=top_k,
                )
            )

        elif temporal_intent.get(
            "has_time_filter"
        ):

            results = (
                self.retrieval_engine.retrieve_historical(
                    query,
                    start=temporal_intent.get(
                        "start"
                    ),
                    end=temporal_intent.get(
                        "end"
                    ),
                    top_k=top_k,
                )
            )

        else:

            results = (
                self.retrieval_engine.retrieve(
                    query,
                    top_k=top_k,
                )
            )

        # --------------------------------------------------------------
        # Step 3 — Build context
        # --------------------------------------------------------------

        context = (
            self.retrieval_engine.build_context(
                results
            )
        )

        return {
            "query": query,
            "temporal_intent": temporal_intent,
            "results": results,
            "context": context,
        }

    # ------------------------------------------------------------------
    # Human-readable response
    # ------------------------------------------------------------------

    @staticmethod
    def format_result(
        result: dict[str, Any],
    ) -> str:

        if not result.get("results"):
            return (
                "No relevant consolidated memories found."
            )

        lines = []

        temporal_intent = result.get(
            "temporal_intent",
            {},
        )

        expression = temporal_intent.get(
            "expression"
        )

        if expression:

            lines.append(
                f"Temporal intent: {expression}"
            )

        lines.append(
            ""
        )

        lines.append(
            result.get(
                "context",
                "",
            )
        )

        return "\n".join(lines)