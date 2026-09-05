import sys
from datetime import datetime, timezone
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from memory.consolidated_store import (
    ConsolidatedMemoryStore,
)

from memory.temporal_retrieval import (
    TemporalConsolidatedRetrievalEngine,
)


def main():

    print("=" * 60)
    print(
        "OMNIMIND TEMPORAL CONSOLIDATED RETRIEVAL TEST"
    )
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:

        path = (
            Path(temp_dir)
            / "consolidated_memories.json"
        )

        store = ConsolidatedMemoryStore(
            path
        )

        # ----------------------------------------------------------
        # Historical database decision
        # ----------------------------------------------------------

        store.add(
            {
                "consolidation_id":
                    "database-history",

                "topic":
                    "OmniMind Vector Database",

                "memory_ids": [
                    "qdrant-decision",
                    "postgres-decision",
                ],

                "summary":
                    (
                        "The project initially used "
                        "Qdrant. The current decision "
                        "is PostgreSQL."
                    ),

                "current_memory_id":
                    "postgres-decision",

                "historical_memory_ids": [
                    "qdrant-decision",
                ],

                "memory_count": 2,

                "created_at":
                    "2026-08-01T10:00:00+00:00",
            }
        )

        # ----------------------------------------------------------
        # Another topic
        # ----------------------------------------------------------

        store.add(
            {
                "consolidation_id":
                    "authentication-history",

                "topic":
                    "OmniMind Authentication",

                "memory_ids": [
                    "auth-001",
                ],

                "summary":
                    (
                        "User authentication was "
                        "implemented for OmniMind."
                    ),

                "current_memory_id":
                    "auth-001",

                "historical_memory_ids": [],

                "memory_count": 1,

                "created_at":
                    "2026-08-20T10:00:00+00:00",
            }
        )

        engine = (
            TemporalConsolidatedRetrievalEngine(
                store
            )
        )

        # ----------------------------------------------------------
        # Test 1 — Historical query
        # ----------------------------------------------------------

        print(
            "\nTEST 1: Historical retrieval"
        )

        start = datetime(
            2026,
            7,
            20,
            tzinfo=timezone.utc,
        )

        end = datetime(
            2026,
            8,
            10,
            tzinfo=timezone.utc,
        )

        results = engine.retrieve_historical(
            "What database did I decide for OmniMind?",
            start=start,
            end=end,
        )

        assert len(results) == 1

        result = results[0][
            "consolidation"
        ]

        assert (
            result["topic"]
            == "OmniMind Vector Database"
        )

        assert (
            "qdrant-decision"
            in result["historical_memory_ids"]
        )

        print(
            "Historical consolidated memory found."
        )

        print(
            "Historical memories:",
            result[
                "historical_memory_ids"
            ],
        )

        # ----------------------------------------------------------
        # Test 2 — Current retrieval
        # ----------------------------------------------------------

        print(
            "\nTEST 2: Current knowledge retrieval"
        )

        results = engine.retrieve_current(
            "What is the current OmniMind database?"
        )

        assert len(results) == 1

        current = results[0][
            "consolidation"
        ]

        assert (
            current["current_memory_id"]
            == "postgres-decision"
        )

        print(
            "Current memory:",
            current[
                "current_memory_id"
            ],
        )

        # ----------------------------------------------------------
        # Test 3 — Temporal exclusion
        # ----------------------------------------------------------

        print(
            "\nTEST 3: Temporal exclusion"
        )

        results = engine.retrieve(
            "OmniMind authentication",
            start=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2026,
                8,
                10,
                tzinfo=timezone.utc,
            ),
        )

        assert results == []

        print(
            "Out-of-range consolidation excluded correctly."
        )

        # ----------------------------------------------------------
        # Test 4 — Context generation
        # ----------------------------------------------------------

        print(
            "\nTEST 4: Temporal context"
        )

        results = engine.retrieve_historical(
            "OmniMind database",
            start=start,
            end=end,
        )

        context = engine.build_context(
            results
        )

        assert (
            "Temporal Consolidated Memory Context:"
            in context
        )

        assert (
            "qdrant-decision"
            in context
        )

        assert (
            "postgres-decision"
            in context
        )

        assert (
            "Temporal filter: applied"
            in context
        )

        print(context)

    print(
        "\nTEMPORAL CONSOLIDATED RETRIEVAL TEST PASSED"
    )


if __name__ == "__main__":
    main()