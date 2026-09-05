from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from dateutil.relativedelta import relativedelta


class ConsolidatedTimeQueryParser:
    """
    Extracts temporal intent from natural-language queries.

    Examples:

        "What did I decide 3 months ago?"
        "What was my decision last month?"
        "What did I choose last week?"
        "What happened 10 days ago?"
        "What is my current decision?"

    The parser returns temporal boundaries but does not perform
    memory retrieval.
    """

    def parse(
        self,
        query: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:

        if not query or not query.strip():
            return {
                "has_time_filter": False,
                "start": None,
                "end": None,
                "expression": None,
                "is_current": False,
            }

        if now is None:
            now = datetime.now(timezone.utc)

        now = self._normalize_datetime(now)

        text = query.lower().strip()

        # --------------------------------------------------------------
        # Current intent
        # --------------------------------------------------------------

        current_signals = [
            "current",
            "currently",
            "now",
            "latest",
            "today",
        ]

        if any(
            signal in text
            for signal in current_signals
        ):

            return {
                "has_time_filter": False,
                "start": None,
                "end": None,
                "expression": "current",
                "is_current": True,
            }

        # --------------------------------------------------------------
        # "N months ago"
        # --------------------------------------------------------------

        import re

        match = re.search(
            r"\b(\d+)\s+months?\s+ago\b",
            text,
        )

        if match:

            amount = int(
                match.group(1)
            )

            center = (
                now
                - relativedelta(
                    months=amount
                )
            )

            start = center - timedelta(
                days=7
            )

            end = center + timedelta(
                days=7
            )

            return {
                "has_time_filter": True,
                "start": start,
                "end": end,
                "expression":
                    f"{amount} months ago",
                "is_current": False,
            }

        # --------------------------------------------------------------
        # "N weeks ago"
        # --------------------------------------------------------------

        match = re.search(
            r"\b(\d+)\s+weeks?\s+ago\b",
            text,
        )

        if match:

            amount = int(
                match.group(1)
            )

            center = (
                now
                - timedelta(
                    weeks=amount
                )
            )

            start = center - timedelta(
                days=3
            )

            end = center + timedelta(
                days=3
            )

            return {
                "has_time_filter": True,
                "start": start,
                "end": end,
                "expression":
                    f"{amount} weeks ago",
                "is_current": False,
            }

        # --------------------------------------------------------------
        # "N days ago"
        # --------------------------------------------------------------

        match = re.search(
            r"\b(\d+)\s+days?\s+ago\b",
            text,
        )

        if match:

            amount = int(
                match.group(1)
            )

            center = (
                now
                - timedelta(
                    days=amount
                )
            )

            start = center - timedelta(
                days=1
            )

            end = center + timedelta(
                days=1
            )

            return {
                "has_time_filter": True,
                "start": start,
                "end": end,
                "expression":
                    f"{amount} days ago",
                "is_current": False,
            }

        # --------------------------------------------------------------
        # "N years ago"
        # --------------------------------------------------------------

        match = re.search(
            r"\b(\d+)\s+years?\s+ago\b",
            text,
        )

        if match:

            amount = int(
                match.group(1)
            )

            center = (
                now
                - relativedelta(
                    years=amount
                )
            )

            start = center - timedelta(
                days=14
            )

            end = center + timedelta(
                days=14
            )

            return {
                "has_time_filter": True,
                "start": start,
                "end": end,
                "expression":
                    f"{amount} years ago",
                "is_current": False,
            }

        # --------------------------------------------------------------
        # Last week
        # --------------------------------------------------------------

        if "last week" in text:

            start = now - timedelta(
                days=7
            )

            return {
                "has_time_filter": True,
                "start": start,
                "end": now,
                "expression": "last week",
                "is_current": False,
            }

        # --------------------------------------------------------------
        # Last month
        # --------------------------------------------------------------

        if "last month" in text:

            start = now - relativedelta(
                months=1
            )

            return {
                "has_time_filter": True,
                "start": start,
                "end": now,
                "expression": "last month",
                "is_current": False,
            }

        # --------------------------------------------------------------
        # Last year
        # --------------------------------------------------------------

        if "last year" in text:

            start = now - relativedelta(
                years=1
            )

            return {
                "has_time_filter": True,
                "start": start,
                "end": now,
                "expression": "last year",
                "is_current": False,
            }

        # --------------------------------------------------------------
        # Recently
        # --------------------------------------------------------------

        if "recently" in text:

            start = now - timedelta(
                days=7
            )

            return {
                "has_time_filter": True,
                "start": start,
                "end": now,
                "expression": "recently",
                "is_current": False,
            }

        # --------------------------------------------------------------
        # No temporal expression
        # --------------------------------------------------------------

        return {
            "has_time_filter": False,
            "start": None,
            "end": None,
            "expression": None,
            "is_current": False,
        }

    # ------------------------------------------------------------------
    # Normalize datetime
    # ------------------------------------------------------------------

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