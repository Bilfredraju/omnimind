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

    The engine is read-only.
    It never modifies or deletes stored memories.
    """

    CURRENT_MIN_SCORE = 0.60

    STOPWORDS = {
        # General language
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

        # Temporal expressions
        "ago",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "year",
        "years",

        # Current / temporal intent
        "current",
        "currently",
        "latest",
        "recent",
        "recently",
        "now",
        "today",

        # Generic memory terminology
        "knowledge",
        "memory",
        "memories",
    }

    # Terms that are too broad to independently identify a topic.
    # They remain searchable but don't independently establish
    # strong topic relevance.
    BROAD_TERMS = {
        "omnimind",
        "project",
        "decision",
        "decide",
        "decided",
        "choice",
        "choose",
        "chose",
        "use",
        "used",
        "using",
        "thing",
        "things",
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
        Retrieve consolidated memories relevant to a query.

        Temporal filtering uses individual timeline events when
        timeline information is available.

        If no timeline exists, consolidation.created_at is used
        as a backward-compatible temporal fallback.
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

            current_memory_id = (
                consolidation.get(
                    "current_memory_id"
                )
            )

            # ----------------------------------------------------------
            # Current-only mode
            # ----------------------------------------------------------

            if not include_historical:

                if not current_memory_id:
                    continue

            # ----------------------------------------------------------
            # Temporal filtering
            # ----------------------------------------------------------

            matched_timeline_events: list[
                dict[str, Any]
            ] = []

            if temporal_filter_requested:

                timeline = consolidation.get(
                    "timeline",
                    [],
                )

                if timeline:

                    matched_timeline_events = (
                        self._events_in_range(
                            timeline,
                            start,
                            end,
                        )
                    )

                    if not matched_timeline_events:
                        continue

                else:

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
            # Searchable consolidation fields
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
            # Timeline terms
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
                    self._tokenize(
                        event_text
                    )
                )

            # ----------------------------------------------------------
            # Current memory text
            # ----------------------------------------------------------

            current_memory_text = ""

            timeline = consolidation.get(
                "timeline",
                [],
            )

            for event in timeline:

                if (
                    event.get(
                        "memory_id"
                    )
                    == current_memory_id
                ):

                    current_memory_text = str(
                        event.get(
                            "text",
                            "",
                        )
                    )

                    break

            current_memory_terms = (
                self._tokenize(
                    current_memory_text
                )
            )

            # ----------------------------------------------------------
            # Calculate overlaps
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

            current_memory_overlap = (
                query_terms.intersection(
                    current_memory_terms
                )
            )

            overlap = (
                topic_overlap
                | summary_overlap
                | timeline_overlap
                | current_memory_overlap
            )

            if not overlap:
                continue

            # ----------------------------------------------------------
            # Meaningful relevance guard
            # ----------------------------------------------------------
            #
            # We don't simply require two matching terms.
            #
            # Example:
            #
            # "OmniMind authentication"
            #
            # should NOT retrieve:
            #
            # "OmniMind Vector Database"
            #
            # because only the broad term "OmniMind" matches.
            #
            # But:
            #
            # "What did I decide about my project 3 months ago?"
            #
            # should retrieve:
            #
            # "OmniMind Vector Database"
            #
            # because "project" and "decision" become normalized
            # searchable terms.
            #
            # And:
            #
            # "OmniMind database"
            #
            # strongly matches the database topic.

            if len(query_terms) >= 2:

                specific_query_terms = (
                    query_terms
                    - self.BROAD_TERMS
                )

                specific_overlap = (
                    overlap
                    - self.BROAD_TERMS
                )

                # If the query contains specific terms,
                # at least one specific term must match.
                if (
                    specific_query_terms
                    and not specific_overlap
                ):
                    continue

                # If every query term is broad,
                # require at least two matching terms.
                if (
                    not specific_query_terms
                    and len(overlap) < 2
                ):
                    continue

            # ----------------------------------------------------------
            # Field-specific ratios
            # ----------------------------------------------------------

            query_size = len(query_terms)

            overlap_ratio = (
                len(overlap)
                / query_size
            )

            topic_ratio = (
                len(topic_overlap)
                / query_size
            )

            summary_ratio = (
                len(summary_overlap)
                / query_size
            )

            timeline_ratio = (
                len(timeline_overlap)
                / query_size
            )

            current_ratio = (
                len(current_memory_overlap)
                / query_size
            )

            # ----------------------------------------------------------
            # Specific-term relevance
            # ----------------------------------------------------------

            specific_query_terms = (
                query_terms
                - self.BROAD_TERMS
            )

            specific_overlap = (
                overlap
                - self.BROAD_TERMS
            )

            if specific_query_terms:

                specific_ratio = (
                    len(specific_overlap)
                    / len(specific_query_terms)
                )

            else:

                specific_ratio = 0.0

            # ----------------------------------------------------------
            # Weighted relevance score
            # ----------------------------------------------------------
            #
            # General overlap       = 35%
            # Topic relevance       = 30%
            # Summary relevance     = 15%
            # Timeline relevance    = 10%
            # Current memory        = 5%
            # Specific terms        = 5%

            score = (
                0.35 * overlap_ratio
                + 0.30 * topic_ratio
                + 0.15 * summary_ratio
                + 0.10 * timeline_ratio
                + 0.05 * current_ratio
                + 0.05 * specific_ratio
            )

            # Current knowledge boost.
            if (
                not include_historical
                and current_memory_id
            ):
                score += 0.05

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
        Retrieve current consolidated knowledge.

        Only consolidations with a current_memory_id are eligible.
        """

        results = self.retrieve(
            query=query,
            include_historical=False,
            top_k=max(top_k, 10),
        )

        current_results = []

        for result in results:

            consolidation = result[
                "consolidation"
            ]

            if not consolidation.get(
                "current_memory_id"
            ):
                continue

            if (
                result["score"]
                < self.CURRENT_MIN_SCORE
            ):
                continue

            current_results.append(
                result
            )

        return current_results[:top_k]

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

        If timeline data exists, historical events are selected
        according to their individual timestamps.
        """

        results = self.retrieve(
            query=query,
            start=start,
            end=end,
            include_historical=True,
            top_k=max(top_k, 10),
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

            # ----------------------------------------------------------
            # Timeline-aware historical retrieval
            # ----------------------------------------------------------

            if matched_events:

                historical_events = [
                    event
                    for event in matched_events
                    if (
                        event.get(
                            "memory_id"
                        )
                        in historical_ids
                        or event.get(
                            "status",
                            "current",
                        )
                        != "current"
                    )
                ]

                if historical_events:

                    result[
                        "matched_timeline_events"
                    ] = historical_events

                    historical_results.append(
                        result
                    )

                    continue

            # ----------------------------------------------------------
            # Backward compatibility for consolidations without
            # timelines.
            # ----------------------------------------------------------

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
        Return timeline events inside the requested date range.
        """

        matched = []

        for event in timeline:

            timestamp = cls._parse_timestamp(
                event.get(
                    "timestamp"
                )
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

        Includes lightweight semantic normalization for common
        memory/query variations such as:

            decide
            decided
            deciding
            decisions

        -> decision

        And:

            database
            databases

        -> database

        This allows natural-language queries to match stored
        memory statements without requiring a full NLP model.
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

        terms: set[str] = set()

        aliases = {
            # Decision normalization
            "decide": "decision",
            "decided": "decision",
            "deciding": "decision",
            "decisions": "decision",

            # Choice normalization
            "choose": "choice",
            "chose": "choice",
            "choosing": "choice",
            "choices": "choice",

            # Database normalization
            "databases": "database",

            # Project normalization
            "projects": "project",

            # Authentication normalization
            "auth": "authentication",
            "authenticate": "authentication",
            "authenticated": "authentication",
            "authenticating": "authentication",
        }

        for token in cleaned.split():

            if len(token) <= 2:
                continue

            if token in cls.STOPWORDS:
                continue

            normalized = aliases.get(
                token,
                token,
            )

            if (
                len(normalized) > 2
                and normalized not in cls.STOPWORDS
            ):
                terms.add(normalized)

        return terms

    # ==================================================================
    # TIMESTAMP PARSING
    # ==================================================================

    @staticmethod
    def _parse_timestamp(
        value: Any,
    ) -> datetime | None:

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

        if value.tzinfo is None:

            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )