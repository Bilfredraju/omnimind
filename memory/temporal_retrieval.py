from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dateutil.parser import isoparse

from memory.consolidated_store import ConsolidatedMemoryStore


class TemporalConsolidatedRetrievalEngine:
    """
    Timeline-aware retrieval for consolidated memories.

    Supports:
    - general consolidated retrieval
    - current knowledge retrieval
    - historical knowledge retrieval
    - explicit datetime ranges
    - timeline-aware temporal filtering
    - context generation

    Important:
    A consolidation's created_at is NOT assumed to be the
    timestamp of every memory inside it.

    When a timeline is available, temporal filtering is performed
    against individual timeline events.
    """

    CURRENT_MIN_SCORE = 0.70

    STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "about",
        "from",
        "into",
        "that",
        "this",
        "these",
        "those",
        "my",
        "our",
        "your",
        "their",
        "i",
        "we",
        "you",
        "me",
        "it",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "as",
        "be",
        "been",
        "being",

        # Temporal terms are handled separately.
        "ago",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "year",
        "years",

        "current",
        "currently",
        "latest",
        "recent",
        "recently",
        "now",
        "today",

        # Generic memory terms.
        "knowledge",
        "memory",
        "memories",

        # IMPORTANT:
        # project, omnimind, decide, decided, decision
        # remain searchable.
    }

    def __init__(
        self,
        store: ConsolidatedMemoryStore | None = None,
    ):
        self.store = (
            store
            if store is not None
            else ConsolidatedMemoryStore()
        )

    # ==================================================================
    # MAIN RETRIEVAL
    # ==================================================================

    def retrieve(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        include_historical: bool = True,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve consolidated memories.

        If a temporal range is supplied and a consolidation contains
        a timeline, individual timeline events are checked against
        the requested range.

        This prevents a later consolidation date from hiding an older
        historical event.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        if start is not None:
            start = self._normalize_datetime(start)

        if end is not None:
            end = self._normalize_datetime(end)

        if (
            start is not None
            and end is not None
            and start > end
        ):
            raise ValueError(
                "Start datetime cannot be after end datetime."
            )

        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        temporal_filter_requested = (
            start is not None
            or end is not None
        )

        results: list[dict[str, Any]] = []

        for consolidation in self.store.all():

            # ----------------------------------------------------------
            # Current-only mode
            # ----------------------------------------------------------

            if not include_historical:

                current_memory_id = (
                    consolidation.get(
                        "current_memory_id"
                    )
                )

                if not current_memory_id:
                    continue

            # ----------------------------------------------------------
            # Timeline-aware temporal filtering
            # ----------------------------------------------------------

            matched_timeline_events = []

            if temporal_filter_requested:

                timeline = consolidation.get(
                    "timeline",
                    []
                )

                if timeline:

                    matched_timeline_events = (
                        self._events_in_range(
                            timeline,
                            start,
                            end,
                        )
                    )

                    # If the consolidation has a timeline,
                    # the timeline is authoritative for historical
                    # temporal retrieval.
                    if not matched_timeline_events:
                        continue

                else:
                    # Backward compatibility:
                    # older consolidations may not have a timeline.
                    created_at = (
                        self._parse_timestamp(
                            consolidation.get(
                                "created_at"
                            )
                        )
                    )

                    if created_at is None:
                        continue

                    if (
                        start is not None
                        and created_at < start
                    ):
                        continue

                    if (
                        end is not None
                        and created_at > end
                    ):
                        continue

            # ----------------------------------------------------------
            # Extract topic and summary
            # ----------------------------------------------------------

            topic = str(
                consolidation.get(
                    "topic",
                    "",
                )
            )

            summary = str(
                consolidation.get(
                    "summary",
                    "",
                )
            )

            topic_terms = self._tokenize(
                topic
            )

            summary_terms = self._tokenize(
                summary
            )

            # ----------------------------------------------------------
            # Include matching timeline text
            # ----------------------------------------------------------

            timeline_terms: set[str] = set()

            for event in matched_timeline_events:

                event_text = str(
                    event.get(
                        "text",
                        "",
                    )
                )

                timeline_terms.update(
                    self._tokenize(event_text)
                )

            # ----------------------------------------------------------
            # Matching
            # ----------------------------------------------------------

            topic_overlap = (
                query_terms.intersection(
                    topic_terms
                )
            )

            summary_overlap = (
                query_terms.intersection(
                    summary_terms
                )
            )

            timeline_overlap = (
                query_terms.intersection(
                    timeline_terms
                )
            )

            overlap = (
                topic_overlap
                | summary_overlap
                | timeline_overlap
            )

            if not overlap:
                continue

            # ----------------------------------------------------------
            # Relevance score
            # ----------------------------------------------------------

            score = (
                len(overlap)
                / len(query_terms)
            )

            # Topic relevance gets the strongest boost.
            if topic_overlap:

                score += (
                    0.50
                    * len(topic_overlap)
                    / len(query_terms)
                )

            # Timeline event relevance is highly useful for
            # historical questions.
            if timeline_overlap:

                score += (
                    0.35
                    * len(timeline_overlap)
                    / len(query_terms)
                )

            # Current knowledge boost.
            if not include_historical:

                current_memory_id = (
                    consolidation.get(
                        "current_memory_id"
                    )
                )

                if current_memory_id:
                    score += 0.15

            # Historical knowledge boost.
            historical_ids = (
                consolidation.get(
                    "historical_memory_ids",
                    [],
                )
            )

            if (
                include_historical
                and historical_ids
            ):
                score += 0.05

            score = round(
                min(score, 1.0),
                4,
            )

            result = {
                "consolidation": consolidation,
                "score": score,
                "temporal_match":
                    temporal_filter_requested,
            }

            if temporal_filter_requested:

                result[
                    "matched_timeline_events"
                ] = matched_timeline_events

            results.append(result)

        # --------------------------------------------------------------
        # Sort by relevance
        # --------------------------------------------------------------

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]

    # ==================================================================
    # CURRENT RETRIEVAL
    # ==================================================================

    def retrieve_current(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve only current consolidated knowledge.
        """

        results = self.retrieve(
            query=query,
            include_historical=False,
            top_k=top_k,
        )

        return [
            result
            for result in results
            if result["score"]
            >= self.CURRENT_MIN_SCORE
        ][:top_k]

    # ==================================================================
    # HISTORICAL RETRIEVAL
    # ==================================================================

    def retrieve_historical(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve historical consolidated knowledge.

        Temporal filtering is performed against individual timeline
        events when a timeline exists.
        """

        results = self.retrieve(
            query=query,
            start=start,
            end=end,
            include_historical=True,
            top_k=top_k,
        )

        historical_results = []

        for result in results:

            consolidation = result[
                "consolidation"
            ]

            historical_ids = (
                consolidation.get(
                    "historical_memory_ids",
                    [],
                )
            )

            matched_events = result.get(
                "matched_timeline_events",
                [],
            )

            # If we have timeline matches, keep the result.
            if matched_events:

                historical_results.append(
                    result
                )

                continue

            # Backward compatibility for consolidations
            # without timelines.
            if historical_ids:

                historical_results.append(
                    result
                )

        return historical_results[:top_k]

    # ==================================================================
    # BUILD CONTEXT
    # ==================================================================

    @staticmethod
    def build_context(
        results: list[dict[str, Any]],
    ) -> str:
        """
        Build LLM-friendly temporal memory context.
        """

        if not results:
            return ""

        lines = [
            "Temporal Consolidated Memory Context:"
        ]

        for result in results:

            memory = result[
                "consolidation"
            ]

            lines.append(
                f"- Topic: "
                f"{memory.get('topic', 'Unknown topic')}"
            )

            lines.append(
                f"  Summary: "
                f"{memory.get('summary', '')}"
            )

            current_id = memory.get(
                "current_memory_id"
            )

            if current_id:

                lines.append(
                    f"  Current memory: "
                    f"{current_id}"
                )

            historical_ids = (
                memory.get(
                    "historical_memory_ids",
                    [],
                )
            )

            if historical_ids:

                lines.append(
                    f"  Historical memories: "
                    f"{', '.join(historical_ids)}"
                )

            matched_events = result.get(
                "matched_timeline_events",
                [],
            )

            if matched_events:

                lines.append(
                    "  Timeline events matching "
                    "the requested period:"
                )

                for event in matched_events:

                    lines.append(
                        "    - "
                        f"[{event.get('memory_id', 'unknown')}] "
                        f"{event.get('timestamp', '')} | "
                        f"{event.get('text', '')}"
                    )

            lines.append(
                f"  Retrieval score: "
                f"{result.get('score', 0.0)}"
            )

            if result.get(
                "temporal_match"
            ):

                lines.append(
                    "  Temporal filter: applied"
                )

        return "\n".join(lines)

    # ==================================================================
    # TIMELINE FILTERING
    # ==================================================================

    @classmethod
    def _events_in_range(
        cls,
        timeline: list[dict[str, Any]],
        start: datetime | None,
        end: datetime | None,
    ) -> list[dict[str, Any]]:
        """
        Return timeline events whose own timestamps fall inside
        the requested temporal range.
        """

        matched = []

        for event in timeline:

            timestamp = cls._parse_timestamp(
                event.get("timestamp")
            )

            if timestamp is None:
                continue

            if (
                start is not None
                and timestamp < start
            ):
                continue

            if (
                end is not None
                and timestamp > end
            ):
                continue

            matched.append(event)

        matched.sort(
            key=lambda event: str(
                event.get(
                    "timestamp",
                    "",
                )
            )
        )

        return matched

    # ==================================================================
    # TOKENIZATION
    # ==================================================================

    @classmethod
    def _tokenize(
        cls,
        text: str,
    ) -> set[str]:
        """
        Normalize text into searchable terms.
        """

        cleaned = (
            text.lower()
            .replace(",", " ")
            .replace(".", " ")
            .replace("?", " ")
            .replace("!", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace("-", " ")
            .replace("/", " ")
            .replace("'", " ")
        )

        return {
            token
            for token in cleaned.split()
            if len(token) > 2
            and token not in cls.STOPWORDS
        }

    # ==================================================================
    # TIMESTAMP PARSING
    # ==================================================================

    @staticmethod
    def _parse_timestamp(
        value: Any,
    ) -> datetime | None:
        """
        Safely parse an ISO timestamp.
        """

        if not value:
            return None

        try:

            timestamp = isoparse(
                str(value)
            )

        except (
            ValueError,
            TypeError,
        ):

            return None

        return (
            TemporalConsolidatedRetrievalEngine
            ._normalize_datetime(
                timestamp
            )
        )

    # ==================================================================
    # DATETIME NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize_datetime(
        value: datetime,
    ) -> datetime:
        """
        Normalize datetime to UTC.
        """

        if value.tzinfo is None:

            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )