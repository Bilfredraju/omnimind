from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.consolidated_store import ConsolidatedMemoryStore
from memory.temporal_retrieval import (
    TemporalConsolidatedRetrievalEngine,
)


def main():
    print("=" * 60)
    print("OMNIMIND TIMELINE-AWARE RETRIEVAL TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:

        path = (
            Path(temp_dir)
            / "consolidated_memories.json"
        )

        store = ConsolidatedMemoryStore(path)

        # ----------------------------------------------------------
        # One consolidation created in August.
        #
        # But the actual memories happened in:
        #
        # June  → Qdrant
        # July  → comparison
        # August → PostgreSQL
        #
        # This is the important scenario.
        # ----------------------------------------------------------

        store.add(
            {
                "consolidation_id":
                    "vector-db-history",

                "topic":
                    "OmniMind Vector Database",

                "memory_ids": [
                    "qdrant-decision",
                    "comparison-decision",
                    "postgres-decision",
                ],

                "summary":
                    (
                        "The project initially used Qdrant. "
                        "The current decision is PostgreSQL."
                    ),

                "current_memory_id":
                    "postgres-decision",

                "historical_memory_ids": [
                    "qdrant-decision",
                    "comparison-decision",
                ],

                # Consolidation created much later.
                "created_at":
                    "2026-08-15T10:00:00+00:00",

                # Actual memory timeline.
                "timeline": [
                    {
                        "memory_id":
                            "qdrant-decision",

                        "timestamp":
                            "2026-06-01T10:00:00+00:00",

                        "text":
                            (
                                "I decided to use "
                                "Qdrant as the vector "
                                "database."
                            ),

                        "type":
                            "decision",

                        "status":
                            "superseded",

                        "importance":
                            1.0,

                        "position":
                            1,
                    },
                    {
                        "memory_id":
                            "comparison-decision",

                        "timestamp":
                            "2026-07-01T10:00:00+00:00",

                        "text":
                            (
                                "I compared Qdrant "
                                "with PostgreSQL."
                            ),

                        "type":
                            "decision",

                        "status":
                            "superseded",

                        "importance":
                            0.9,

                        "position":
                            2,
                    },
                    {
                        "memory_id":
                            "postgres-decision",

                        "timestamp":
                            "2026-08-01T10:00:00+00:00",

                        "text":
                            (
                                "I changed my decision "
                                "and will use PostgreSQL."
                            ),

                        "type":
                            "decision",

                        "status":
                            "current",

                        "importance":
                            1.0,

                        "position":
                            3,
                    },
                ],
            }
        )

        engine = (
            TemporalConsolidatedRetrievalEngine(
                store
            )
        )

        # Fixed current date.
        now = datetime(
            2026,
            9,
            5,
            12,
            0,
            tzinfo=timezone.utc,
        )

        # ----------------------------------------------------------
        # Test 1
        # ----------------------------------------------------------

        print(
            "\nTEST 1: Historical event inside date range"
        )

        results = engine.retrieve_historical(
            query=(
                "What database decision "
                "did I make?"
            ),
            start=datetime(
                2026,
                5,
                29,
                12,
                0,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2026,
                6,
                12,
                12,
                0,
                tzinfo=timezone.utc,
            ),
        )

        assert len(results) == 1

        result = results[0]

        assert (
            result["consolidation"][
                "consolidation_id"
            ]
            == "vector-db-history"
        )

        assert (
            "matched_timeline_events"
            in result
        )

        matched = result[
            "matched_timeline_events"
        ]

        assert len(matched) == 1

        assert (
            matched[0]["memory_id"]
            == "qdrant-decision"
        )

        print(
            "Matched historical memory:",
            matched[0]["memory_id"],
        )

        print(
            "Event timestamp:",
            matched[0]["timestamp"],
        )

        # ----------------------------------------------------------
        # Test 2
        # ----------------------------------------------------------

        print(
            "\nTEST 2: Consolidation created later"
        )

        consolidation_created = (
            result["consolidation"][
                "created_at"
            ]
        )

        assert (
            consolidation_created
            == "2026-08-15T10:00:00+00:00"
        )

        print(
            "Consolidation created:",
            consolidation_created,
        )

        print(
            "Historical event still found correctly."
        )

        # ----------------------------------------------------------
        # Test 3
        # ----------------------------------------------------------

        print(
            "\nTEST 3: August event excluded "
            "from June range"
        )

        for event in matched:
            event_time = datetime.fromisoformat(
                event["timestamp"]
            )

            assert (
                event_time.month
                == 6
            )

        print(
            "Out-of-range events excluded correctly."
        )

        # ----------------------------------------------------------
        # Test 4
        # ----------------------------------------------------------

        print(
            "\nTEST 4: Current knowledge"
        )

        current_results = (
            engine.retrieve_current(
                "What is my current database decision?"
            )
        )

        assert len(
            current_results
        ) == 1

        current = current_results[0]

        assert (
            current["consolidation"][
                "current_memory_id"
            ]
            == "postgres-decision"
        )

        print(
            "Current memory:",
            current["consolidation"][
                "current_memory_id"
            ],
        )

        # ----------------------------------------------------------
        # Test 5
        # ----------------------------------------------------------

        print(
            "\nTEST 5: Timeline-aware context"
        )

        context = (
            engine.build_context(
                results
            )
        )

        assert (
            "qdrant-decision"
            in context
        )

        print(context)

        # ----------------------------------------------------------
        # Test 6
        # ----------------------------------------------------------

        print(
            "\nTEST 6: Future/current event excluded "
            "from historical result"
        )

        for event in matched:

            event_time = datetime.fromisoformat(
                event["timestamp"]
            )

            assert (
                event_time
                <= datetime(
                    2026,
                    6,
                    12,
                    12,
                    0,
                    tzinfo=timezone.utc,
                )
            )

        print(
            "Future events correctly excluded."
        )

    print(
        "\nTIMELINE-AWARE RETRIEVAL TEST PASSED"
    )


if __name__ == "__main__":
    main()