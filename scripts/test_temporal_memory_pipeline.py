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

from memory.temporal_memory_pipeline import (
    TemporalMemoryPipeline,
)


def main():

    print("=" * 60)
    print(
        "OMNIMIND END-TO-END TEMPORAL MEMORY PIPELINE TEST"
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
        #
        # Current date:
        # 2026-09-05
        #
        # Three months ago:
        # approximately June 2026
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
                    "2026-06-05T10:00:00+00:00",
            }
        )

        # ----------------------------------------------------------
        # Unrelated later memory
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

        pipeline = TemporalMemoryPipeline(
            store=store
        )

        fixed_now = datetime(
            2026,
            9,
            5,
            12,
            0,
            tzinfo=timezone.utc,
        )

        # ----------------------------------------------------------
        # Test 1 — Natural language historical query
        # ----------------------------------------------------------

        print(
            "\nTEST 1: Natural-language historical query"
        )

        query = (
            "What did I decide about my project "
            "3 months ago?"
        )

        result = pipeline.query(
            query,
            now=fixed_now,
        )

        assert result["temporal_intent"][
            "has_time_filter"
        ] is True

        assert (
            result["temporal_intent"][
                "expression"
            ]
            == "3 months ago"
        )

        assert len(
            result["results"]
        ) == 1

        retrieved = result[
            "results"
        ][0]["consolidation"]

        assert (
            retrieved["consolidation_id"]
            == "database-history"
        )

        assert (
            "qdrant-decision"
            in retrieved[
                "historical_memory_ids"
            ]
        )

        print(
            "Query:",
            query,
        )

        print(
            "Temporal expression:",
            result["temporal_intent"][
                "expression"
            ],
        )

        print(
            "Retrieved topic:",
            retrieved["topic"],
        )

        print(
            "Historical memory:",
            retrieved[
                "historical_memory_ids"
            ],
        )

        # ----------------------------------------------------------
        # Test 2 — Current query
        # ----------------------------------------------------------

        print(
            "\nTEST 2: Current knowledge query"
        )

        current_query = (
            "What is my current database decision?"
        )

        current_result = pipeline.query(
            current_query,
            now=fixed_now,
        )

        assert current_result[
            "temporal_intent"
        ]["is_current"] is True

        assert len(
            current_result["results"]
        ) == 1

        current_memory = (
            current_result[
                "results"
            ][0]["consolidation"]
        )

        assert (
            current_memory[
                "current_memory_id"
            ]
            == "postgres-decision"
        )

        print(
            "Current memory:",
            current_memory[
                "current_memory_id"
            ],
        )

        # ----------------------------------------------------------
        # Test 3 — Context
        # ----------------------------------------------------------

        print(
            "\nTEST 3: Generated memory context"
        )

        context = result[
            "context"
        ]

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

        print(context)

        # ----------------------------------------------------------
        # Test 4 — Formatted result
        # ----------------------------------------------------------

        print(
            "\nTEST 4: Human-readable result"
        )

        formatted = (
            pipeline.format_result(
                result
            )
        )

        assert (
            "3 months ago"
            in formatted
        )

        assert (
            "OmniMind Vector Database"
            in formatted
        )

        print(formatted)

        # ----------------------------------------------------------
        # Test 5 — No-result query
        # ----------------------------------------------------------

        print(
            "\nTEST 5: No matching memory"
        )

        empty_result = pipeline.query(
            "What did I decide about "
            "quantum computing 3 months ago?",
            now=fixed_now,
        )

        assert (
            empty_result["results"]
            == []
        )

        print(
            "No-result query handled correctly."
        )

    print(
        "\nEND-TO-END TEMPORAL MEMORY PIPELINE TEST PASSED"
    )


if __name__ == "__main__":
    main()