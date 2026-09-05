import sys
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

from memory.consolidated_retrieval import (
    ConsolidatedMemoryRetrievalEngine,
)


def main():

    print("=" * 60)
    print(
        "OMNIMIND CONSOLIDATED MEMORY RETRIEVAL TEST"
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
        # Test data
        # ----------------------------------------------------------

        store.add(
            {
                "consolidation_id":
                    "vector-db-001",

                "topic":
                    "OmniMind Vector Database",

                "memory_ids": [
                    "qdrant-001",
                    "postgres-001",
                ],

                "summary":
                    (
                        "The project initially used "
                        "Qdrant but the current decision "
                        "is PostgreSQL."
                    ),

                "current_memory_id":
                    "postgres-001",

                "historical_memory_ids": [
                    "qdrant-001",
                ],

                "memory_count": 2,

                "created_at":
                    "2026-08-01T10:00:00+00:00",
            }
        )

        store.add(
            {
                "consolidation_id":
                    "authentication-001",

                "topic":
                    "OmniMind Authentication",

                "memory_ids": [
                    "auth-001",
                ],

                "summary":
                    (
                        "Implemented user authentication "
                        "for the OmniMind application."
                    ),

                "current_memory_id":
                    "auth-001",

                "historical_memory_ids": [],

                "memory_count": 1,

                "created_at":
                    "2026-08-02T10:00:00+00:00",
            }
        )

        engine = (
            ConsolidatedMemoryRetrievalEngine(
                store
            )
        )

        # ----------------------------------------------------------
        # Test 1 — Relevant retrieval
        # ----------------------------------------------------------

        print(
            "\nTEST 1: Retrieve vector database memory"
        )

        results = engine.retrieve(
            "What database did I choose for OmniMind?"
        )

        assert len(results) >= 1

        best = results[0][
            "consolidation"
        ]

        assert (
            best["consolidation_id"]
            == "vector-db-001"
        )

        assert results[0][
            "score"
        ] > 0

        print(
            "Best result:",
            best["topic"],
        )

        print(
            "Score:",
            results[0]["score"],
        )

        # ----------------------------------------------------------
        # Test 2 — Current memory
        # ----------------------------------------------------------

        print(
            "\nTEST 2: Current knowledge retrieval"
        )

        assert (
            best["current_memory_id"]
            == "postgres-001"
        )

        print(
            "Current memory:",
            best["current_memory_id"],
        )

        # ----------------------------------------------------------
        # Test 3 — Irrelevant query
        # ----------------------------------------------------------

        print(
            "\nTEST 3: Irrelevant query"
        )

        results = engine.retrieve(
            "weather forecast tomorrow"
        )

        assert results == []

        print(
            "Irrelevant query correctly returned no results."
        )

        # ----------------------------------------------------------
        # Test 4 — Top K
        # ----------------------------------------------------------

        print(
            "\nTEST 4: Top-K retrieval"
        )

        results = engine.retrieve(
            "OmniMind",
            top_k=1,
        )

        assert len(results) <= 1

        print(
            "Top-K constraint respected."
        )

        # ----------------------------------------------------------
        # Test 5 — Context generation
        # ----------------------------------------------------------

        print(
            "\nTEST 5: Build retrieval context"
        )

        results = engine.retrieve(
            "OmniMind database"
        )

        context = engine.build_context(
            results
        )

        assert (
            "Consolidated Memory Context:"
            in context
        )

        assert (
            "OmniMind Vector Database"
            in context
        )

        assert (
            "postgres-001"
            in context
        )

        print(context)

    print(
        "\nCONSOLIDATED MEMORY RETRIEVAL TEST PASSED"
    )


if __name__ == "__main__":
    main()