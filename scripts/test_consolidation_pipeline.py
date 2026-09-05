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

from memory.consolidation_pipeline import (
    MemoryConsolidationPipeline,
)


def main():

    print("=" * 60)
    print(
        "OMNIMIND MEMORY CONSOLIDATION PIPELINE TEST"
    )
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:

        store_path = (
            Path(temp_dir)
            / "consolidated_memories.json"
        )

        store = ConsolidatedMemoryStore(
            store_path
        )

        pipeline = MemoryConsolidationPipeline(
            store=store
        )

        # ----------------------------------------------------------
        # Historical → current memory evolution
        # ----------------------------------------------------------

        memories = [
            {
                "memory_id": "memory-qdrant",
                "text": (
                    "I decided to use Qdrant "
                    "as the vector database."
                ),
                "metadata": {
                    "type": "decision",
                    "importance": 1.0,
                    "status": "superseded",
                    "created_at":
                        "2026-06-01T10:00:00+00:00",
                },
            },
            {
                "memory_id": "memory-comparison",
                "text": (
                    "I compared Qdrant "
                    "with PostgreSQL."
                ),
                "metadata": {
                    "type": "project",
                    "importance": 0.85,
                    "status": "current",
                    "created_at":
                        "2026-07-01T10:00:00+00:00",
                },
            },
            {
                "memory_id": "memory-postgres",
                "text": (
                    "I changed my decision "
                    "and will use PostgreSQL."
                ),
                "metadata": {
                    "type": "decision",
                    "importance": 1.0,
                    "status": "current",
                    "created_at":
                        "2026-08-01T10:00:00+00:00",
                },
            },
        ]

        # ----------------------------------------------------------
        # Test 1 — Full pipeline
        # ----------------------------------------------------------

        print(
            "\nTEST 1: Run consolidation pipeline"
        )

        result = pipeline.consolidate(
            memories,
            topic="OmniMind Vector Database",
        )

        assert result is not None

        assert (
            result["topic"]
            == "OmniMind Vector Database"
        )

        assert (
            result["memory_count"]
            == 3
        )

        assert (
            result["current_memory_id"]
            == "memory-postgres"
        )

        print(
            "Pipeline executed successfully."
        )

        # ----------------------------------------------------------
        # Test 2 — Timeline
        # ----------------------------------------------------------

        print(
            "\nTEST 2: Timeline attached"
        )

        timeline = result[
            "timeline"
        ]

        assert len(timeline) == 3

        assert (
            timeline[0]["memory_id"]
            == "memory-qdrant"
        )

        assert (
            timeline[2]["memory_id"]
            == "memory-postgres"
        )

        print(
            "Timeline attached successfully."
        )

        # ----------------------------------------------------------
        # Test 3 — Persistent storage
        # ----------------------------------------------------------

        print(
            "\nTEST 3: Persistent storage"
        )

        assert store.count() == 1

        reloaded_store = (
            ConsolidatedMemoryStore(
                store_path
            )
        )

        persisted = reloaded_store.get(
            result["consolidation_id"]
        )

        assert persisted is not None

        assert (
            persisted["current_memory_id"]
            == "memory-postgres"
        )

        assert len(
            persisted["timeline"]
        ) == 3

        print(
            "Persistent consolidation verified."
        )

        # ----------------------------------------------------------
        # Test 4 — Original memories unchanged
        # ----------------------------------------------------------

        print(
            "\nTEST 4: Original memories preserved"
        )

        assert (
            memories[0]["text"]
            == (
                "I decided to use Qdrant "
                "as the vector database."
            )
        )

        assert (
            memories[0]["metadata"]["status"]
            == "superseded"
        )

        assert (
            memories[2]["metadata"]["status"]
            == "current"
        )

        assert (
            memories[2]["text"]
            == (
                "I changed my decision "
                "and will use PostgreSQL."
            )
        )

        print(
            "Original memories remain unchanged."
        )

        # ----------------------------------------------------------
        # Test 5 — Search
        # ----------------------------------------------------------

        print(
            "\nTEST 5: Search consolidated knowledge"
        )

        results = pipeline.search_topic(
            "Vector Database"
        )

        assert len(results) == 1

        assert (
            results[0]["consolidation_id"]
            == result["consolidation_id"]
        )

        print(
            "Consolidated knowledge search successful."
        )

        # ----------------------------------------------------------
        # Test 6 — Format
        # ----------------------------------------------------------

        print(
            "\nTEST 6: Format consolidated result"
        )

        formatted = pipeline.format_result(
            result
        )

        assert (
            "OmniMind Vector Database"
            in formatted
        )

        assert "memory-qdrant" in formatted

        assert "memory-postgres" in formatted

        assert "PostgreSQL" in formatted

        print(formatted)

    print(
        "\nMEMORY CONSOLIDATION PIPELINE TEST PASSED"
    )


if __name__ == "__main__":
    main()