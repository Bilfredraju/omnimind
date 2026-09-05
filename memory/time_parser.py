import re
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta


class MemoryTimeParser:
    """
    Detect temporal expressions in natural-language memory queries.

    The parser returns a temporal window:

        {
            "has_time_filter": bool,
            "start": datetime | None,
            "end": datetime | None,
            "expression": str | None,
        }

    This first version focuses on common conversational
    expressions used when asking about past memories.
    """

    RECENT_DAYS = 7

    def parse(
        self,
        query: str,
        now: datetime | None = None,
    ) -> dict:

        query = query.strip().lower()

        if now is None:
            now = datetime.now(timezone.utc)

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # --------------------------------------------------
        # Yesterday
        # --------------------------------------------------
        if re.search(r"\byesterday\b", query):
            start = now - timedelta(days=1)
            end = now

            return self._result(
                start=start,
                end=end,
                expression="yesterday",
            )

        # --------------------------------------------------
        # Last week
        # --------------------------------------------------
        if re.search(r"\blast week\b", query):
            start = now - timedelta(days=7)
            end = now

            return self._result(
                start=start,
                end=end,
                expression="last week",
            )

        # --------------------------------------------------
        # Last month
        # --------------------------------------------------
        if re.search(r"\blast month\b", query):
            start = now - relativedelta(months=1)
            end = now

            return self._result(
                start=start,
                end=end,
                expression="last month",
            )

        # --------------------------------------------------
        # Last year
        # --------------------------------------------------
        if re.search(r"\blast year\b", query):
            start = now - relativedelta(years=1)
            end = now

            return self._result(
                start=start,
                end=end,
                expression="last year",
            )

        # --------------------------------------------------
        # Recently / recent
        # --------------------------------------------------
        if re.search(r"\brecently\b|\brecent\b", query):
            start = now - timedelta(
                days=self.RECENT_DAYS
            )
            end = now

            return self._result(
                start=start,
                end=end,
                expression="recently",
            )

        # --------------------------------------------------
        # N days ago
        # --------------------------------------------------
        match = re.search(
            r"\b(\d+)\s+days?\s+ago\b",
            query,
        )

        if match:
            days = int(match.group(1))

            start = now - timedelta(days=days + 1)
            end = now - timedelta(days=days - 1)

            return self._result(
                start=start,
                end=end,
                expression=match.group(0),
            )

        # --------------------------------------------------
        # N weeks ago
        # --------------------------------------------------
        match = re.search(
            r"\b(\d+)\s+weeks?\s+ago\b",
            query,
        )

        if match:
            weeks = int(match.group(1))

            target = now - timedelta(
                weeks=weeks
            )

            start = target - timedelta(days=3)
            end = target + timedelta(days=3)

            return self._result(
                start=start,
                end=end,
                expression=match.group(0),
            )

        # --------------------------------------------------
        # N months ago
        # --------------------------------------------------
        match = re.search(
            r"\b(\d+)\s+months?\s+ago\b",
            query,
        )

        if match:
            months = int(match.group(1))

            target = now - relativedelta(
                months=months
            )

            start = target - timedelta(days=7)
            end = target + timedelta(days=7)

            return self._result(
                start=start,
                end=end,
                expression=match.group(0),
            )

        # --------------------------------------------------
        # N years ago
        # --------------------------------------------------
        match = re.search(
            r"\b(\d+)\s+years?\s+ago\b",
            query,
        )

        if match:
            years = int(match.group(1))

            target = now - relativedelta(
                years=years
            )

            start = target - timedelta(days=14)
            end = target + timedelta(days=14)

            return self._result(
                start=start,
                end=end,
                expression=match.group(0),
            )

        # --------------------------------------------------
        # No temporal expression
        # --------------------------------------------------
        return {
            "has_time_filter": False,
            "start": None,
            "end": None,
            "expression": None,
        }

    @staticmethod
    def _result(
        start: datetime,
        end: datetime,
        expression: str,
    ) -> dict:

        return {
            "has_time_filter": True,
            "start": start,
            "end": end,
            "expression": expression,
        }