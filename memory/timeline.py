from __future__ import annotations

from datetime import timezone
from typing import Any

from dateutil.parser import isoparse


class MemoryTimelineEngine:
    """
    Builds a chronological timeline from related memories.

    The engine does not modify, delete, or supersede memories.
    It only converts existing memories into chronological events.
    """

    def build_timeline(
        self,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        events = []

        for memory in memories:
            event = self._build_event(memory)

            if event is not None:
                events.append(event)

        events.sort(
            key=lambda event: event["timestamp"]
        )

        for index, event in enumerate(events, start=1):
            event["position"] = index

        return events

    @staticmethod
    def _build_event(
        memory: dict[str, Any],
    ) -> dict[str, Any] | None:

        metadata = memory.get("metadata", {})

        created_at = metadata.get("created_at")

        if not created_at:
            return None

        try:
            timestamp = isoparse(str(created_at))
        except (ValueError, TypeError):
            return None

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        try:
            importance = float(
                metadata.get("importance", 0.0)
            )
        except (TypeError, ValueError):
            importance = 0.0

        return {
            "memory_id": memory.get("memory_id"),
            "timestamp": timestamp.isoformat(),
            "text": memory.get("text", ""),
            "type": metadata.get("type", "general"),
            "status": metadata.get("status", "current"),
            "importance": importance,
            "position": 0,
        }

    @staticmethod
    def get_current_event(
        timeline: list[dict[str, Any]],
    ) -> dict[str, Any] | None:

        current_events = [
            event
            for event in timeline
            if event.get("status") == "current"
        ]

        if not current_events:
            return None

        return current_events[-1]

    @staticmethod
    def get_historical_events(
        timeline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        return [
            event
            for event in timeline
            if event.get("status") != "current"
        ]

    @staticmethod
    def format_timeline(
        timeline: list[dict[str, Any]],
    ) -> str:

        if not timeline:
            return "No timeline events available."

        lines = [
            "Memory Timeline:"
        ]

        for event in timeline:

            memory_id = event.get(
                "memory_id",
                "unknown",
            )

            timestamp = event.get(
                "timestamp",
                "unknown",
            )

            memory_type = event.get(
                "type",
                "general",
            )

            status = event.get(
                "status",
                "current",
            )

            importance = event.get(
                "importance",
                0.0,
            )

            text = event.get(
                "text",
                "",
            )

            lines.append(
                f"{event['position']}. "
                f"[{memory_id}] "
                f"{timestamp} | "
                f"type={memory_type} | "
                f"status={status} | "
                f"importance={importance} | "
                f"{text}"
            )

        return "\n".join(lines)