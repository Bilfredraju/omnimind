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


def main():

    print("=" * 60)
    print(
        "OMNIMIND PERSISTENT CONSOLIDATED MEMORY TEST"
    )
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:

        path = (
            Path(temp_dir)
            / "consolidated_memories.json"
        )

        # ----------------------------------------------------------
        # Test 1 — Add
        # ----------------------------------------------------------

        print(
            "\nTEST 1: Add consolidation"
        )

        store = ConsolidatedMemoryStore(
            path
        )

        consolidation = {
            "consolidation_id":
                "consolidation-001",

            "topic":
                "OmniMind Vector Database",

            "memory_ids": [
                "memory-1",
                "memory-2",
                "memory-3",
            ],

            "summary":
                (
                    "OmniMind Vector Database: "
                    "Current knowledge is PostgreSQL."
                ),

            "current_memory_id":
                "memory-3",

            "historical_memory_ids": [
                "memory-1",
            ],

            "memory_count": 3,

            "created_at":
                "2026-08-01T10:00:00+00:00",
        }

        result = store.add(
            consolidation
        )

        assert result[
            "consolidation_id"
        ] == "consolidation-001"

        assert store.count() == 1

        print(
            "Consolidation stored successfully."
        )

        # ----------------------------------------------------------
        # Test 2 — Get
        # ----------------------------------------------------------

        print(
            "\nTEST 2: Retrieve consolidation"
        )

        retrieved = store.get(
            "consolidation-001"
        )

        assert retrieved is not None

        assert retrieved[
            "topic"
        ] == "OmniMind Vector Database"

        assert retrieved[
            "current_memory_id"
        ] == "memory-3"

        print(
            "Consolidation retrieved successfully."
        )

        # ----------------------------------------------------------
        # Test 3 — Topic search
        # ----------------------------------------------------------

        print(
            "\nTEST 3: Topic search"
        )

        results = store.search_topic(
            "Vector Database"
        )

        assert len(results) == 1

        assert results[0][
            "consolidation_id"
        ] == "consolidation-001"

        print(
            "Topic search successful."
        )

        # ----------------------------------------------------------
        # Test 4 — Persistence
        # ----------------------------------------------------------

        print(
            "\nTEST 4: Persistence reload"
        )

        reloaded_store = (
            ConsolidatedMemoryStore(path)
        )

        assert reloaded_store.count() == 1

        persisted = reloaded_store.get(
            "consolidation-001"
        )

        assert persisted is not None

        assert persisted[
            "current_memory_id"
        ] == "memory-3"

        assert persisted[
            "historical_memory_ids"
        ] == ["memory-1"]

        print(
            "Persistence verified successfully."
        )

        # ----------------------------------------------------------
        # Test 5 — Update same consolidation
        # ----------------------------------------------------------

        print(
            "\nTEST 5: Update existing consolidation"
        )

        updated = dict(consolidation)

        updated[
            "summary"
        ] = (
            "Updated consolidated knowledge: "
            "PostgreSQL remains the current choice."
        )

        store.add(updated)

        assert store.count() == 1

        updated_result = store.get(
            "consolidation-001"
        )

        assert updated_result[
            "summary"
        ].startswith(
            "Updated consolidated knowledge"
        )

        print(
            "Existing consolidation updated successfully."
        )

        # ----------------------------------------------------------
        # Test 6 — Delete
        # ----------------------------------------------------------

        print(
            "\nTEST 6: Delete consolidation"
        )

        deleted = store.delete(
            "consolidation-001"
        )

        assert deleted is True

        assert store.count() == 0

        print(
            "Deletion successful."
        )

        # ----------------------------------------------------------
        # Test 7 — Empty input
        # ----------------------------------------------------------

        print(
            "\nTEST 7: Empty consolidation"
        )

        try:

            store.add({})

            raise AssertionError(
                "Empty consolidation should fail."
            )

        except ValueError:

            print(
                "Empty consolidation handled correctly."
            )

    print(
        "\nPERSISTENT CONSOLIDATED MEMORY TEST PASSED"
    )


if __name__ == "__main__":
    main()