import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from memory.consolidation import (
    MemoryConsolidationEngine,
)


def main():

    print("=" * 60)
    print(
        "OMNIMIND MEMORY CONSOLIDATION TEST"
    )
    print("=" * 60)

    engine = MemoryConsolidationEngine()

    # --------------------------------------------------------------
    # Historical + current memories
    # --------------------------------------------------------------

    memories = [
        {
            "memory_id": "memory-1",
            "text": (
                "I decided to use Qdrant "
                "for the OmniMind project."
            ),
            "metadata": {
                "type": "decision",
                "importance": 1.0,
                "status": "superseded",
                "created_at": (
                    "2026-06-01T10:00:00+00:00"
                ),
            },
        },
        {
            "memory_id": "memory-2",
            "text": (
                "I compared Qdrant "
                "with PostgreSQL."
            ),
            "metadata": {
                "type": "project",
                "importance": 0.85,
                "status": "current",
                "created_at": (
                    "2026-07-01T10:00:00+00:00"
                ),
            },
        },
        {
            "memory_id": "memory-3",
            "text": (
                "I changed my decision "
                "and will use PostgreSQL."
            ),
            "metadata": {
                "type": "decision",
                "importance": 1.0,
                "status": "current",
                "created_at": (
                    "2026-08-01T10:00:00+00:00"
                ),
            },
        },
    ]

    # --------------------------------------------------------------
    # Test 1 — Consolidation
    # --------------------------------------------------------------

    print(
        "\nTEST 1: Consolidate related memories"
    )

    result = engine.consolidate(
        memories,
        topic="OmniMind Vector Database",
    )

    print(
        "Topic:",
        result["topic"],
    )

    print(
        "Memory count:",
        result["memory_count"],
    )

    print(
        "Current memory:",
        result["current_memory_id"],
    )

    print(
        "Historical memories:",
        result["historical_memory_ids"],
    )

    print(
        "Summary:",
        result["summary"],
    )

    assert result["memory_count"] == 3

    assert result["current_memory_id"] == (
        "memory-3"
    )

    assert "memory-1" in (
        result["historical_memory_ids"]
    )

    assert "memory-2" not in (
        result["historical_memory_ids"]
    )

    # --------------------------------------------------------------
    # Test 2 — Original memories preserved
    # --------------------------------------------------------------

    print(
        "\nTEST 2: Original memories preserved"
    )

    assert len(memories) == 3

    assert memories[0]["memory_id"] == (
        "memory-1"
    )

    assert memories[2]["memory_id"] == (
        "memory-3"
    )

    print(
        "Original memories preserved: True"
    )

    # --------------------------------------------------------------
    # Test 3 — Empty input
    # --------------------------------------------------------------

    print(
        "\nTEST 3: Empty input"
    )

    try:

        engine.consolidate([])

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:

        print(
            "Empty input handled correctly."
        )

    # --------------------------------------------------------------
    # Test 4 — Single memory
    # --------------------------------------------------------------

    print(
        "\nTEST 4: Single memory"
    )

    single_memory = [
        {
            "memory_id": "single-1",
            "text": (
                "I prefer Python for ML projects."
            ),
            "metadata": {
                "type": "preference",
                "importance": 0.8,
                "status": "current",
                "created_at": (
                    "2026-09-01T10:00:00+00:00"
                ),
            },
        }
    ]

    single_result = engine.consolidate(
        single_memory,
        topic="Python preference",
    )

    assert single_result[
        "memory_count"
    ] == 1

    assert single_result[
        "current_memory_id"
    ] == "single-1"

    print(
        "Single memory handled correctly."
    )

    print(
        "\nMEMORY CONSOLIDATION TEST PASSED"
    )


if __name__ == "__main__":
    main()