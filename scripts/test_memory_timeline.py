import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from memory.timeline import (
    MemoryTimelineEngine,
)


def main():

    print("=" * 60)
    print(
        "OMNIMIND MEMORY TIMELINE TEST"
    )
    print("=" * 60)

    engine = MemoryTimelineEngine()

    # --------------------------------------------------------------
    # Test memories
    # --------------------------------------------------------------

    memories = [
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
        {
            "memory_id": "memory-1",
            "text": (
                "I decided to use Qdrant."
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
    ]

    # --------------------------------------------------------------
    # Test 1 — Chronological ordering
    # --------------------------------------------------------------

    print(
        "\nTEST 1: Chronological ordering"
    )

    timeline = engine.build_timeline(
        memories
    )

    assert len(timeline) == 3

    assert timeline[0][
        "memory_id"
    ] == "memory-1"

    assert timeline[1][
        "memory_id"
    ] == "memory-2"

    assert timeline[2][
        "memory_id"
    ] == "memory-3"

    print(
        "Timeline order:"
    )

    for event in timeline:

        print(
            f"{event['position']}. "
            f"{event['memory_id']} → "
            f"{event['timestamp']}"
        )

    # --------------------------------------------------------------
    # Test 2 — Current event
    # --------------------------------------------------------------

    print(
        "\nTEST 2: Current event"
    )

    current = engine.get_current_event(
        timeline
    )

    assert current is not None

    assert current[
        "memory_id"
    ] == "memory-3"

    print(
        "Current event:",
        current["memory_id"],
    )

    # --------------------------------------------------------------
    # Test 3 — Historical events
    # --------------------------------------------------------------

    print(
        "\nTEST 3: Historical events"
    )

    historical = engine.get_historical_events(
        timeline
    )

    assert len(historical) == 1

    assert historical[0][
        "memory_id"
    ] == "memory-1"

    print(
        "Historical events:",
        [
            event["memory_id"]
            for event in historical
        ],
    )

    # --------------------------------------------------------------
    # Test 4 — Invalid timestamp
    # --------------------------------------------------------------

    print(
        "\nTEST 4: Invalid timestamp"
    )

    invalid_memory = {
        "memory_id": "invalid-1",
        "text": "Invalid timestamp memory.",
        "metadata": {
            "type": "general",
            "created_at": "not-a-date",
        },
    }

    invalid_result = engine.build_timeline(
        [invalid_memory]
    )

    assert invalid_result == []

    print(
        "Invalid timestamp handled correctly."
    )

    # --------------------------------------------------------------
    # Test 5 — Timeline formatting
    # --------------------------------------------------------------

    print(
        "\nTEST 5: Timeline formatting"
    )

    formatted = engine.format_timeline(
        timeline
    )

    assert "Memory Timeline:" in formatted

    assert "memory-1" in formatted

    assert "memory-3" in formatted

    print(formatted)

    print(
        "\nMEMORY TIMELINE TEST PASSED"
    )


if __name__ == "__main__":
    main()