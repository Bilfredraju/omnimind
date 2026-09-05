import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.updater import MemoryUpdateEngine


def main():
    print("=" * 60)
    print("OMNIMIND MEMORY UPDATE TEST")
    print("=" * 60)

    engine = MemoryUpdateEngine()

    old_memory = {
        "memory_id": "memory-old",
        "text": "I decided to use Qdrant as the vector database.",
        "metadata": {
            "type": "decision",
            "created_at": "2026-06-01T12:00:00+00:00",
            "status": "current",
        },
    }

    # --------------------------------------------------------------
    # Test 1: Explicit update
    # --------------------------------------------------------------

    update_text = (
        "I changed my decision. "
        "I'll use PostgreSQL instead."
    )

    print()
    print("Test 1 - Explicit decision change")
    print("Text:", update_text)

    result = engine.is_update(
        update_text,
        old_memory,
    )

    print("Update detected:", result)

    assert result is True

    # --------------------------------------------------------------
    # Test 2: Same decision wording
    # --------------------------------------------------------------

    duplicate_text = (
        "For my OmniMind project, "
        "I chose Qdrant for the vector database."
    )

    print()
    print("Test 2 - Same decision")
    print("Text:", duplicate_text)

    result = engine.is_update(
        duplicate_text,
        old_memory,
    )

    print("Update detected:", result)

    # Choosing the same thing is NOT an update.
    assert result is False

    # --------------------------------------------------------------
    # Test 3: Ordinary statement
    # --------------------------------------------------------------

    ordinary_text = (
        "I am learning about PostgreSQL."
    )

    print()
    print("Test 3 - Ordinary statement")
    print("Text:", ordinary_text)

    result = engine.is_update(
        ordinary_text,
        old_memory,
    )

    print("Update detected:", result)

    assert result is False

    # --------------------------------------------------------------
    # Test 4: Apply update
    # --------------------------------------------------------------

    new_memory = {
        "memory_id": "memory-new",
        "text": update_text,
        "metadata": {
            "type": "decision",
            "created_at": "2026-09-05T12:00:00+00:00",
        },
    }

    result = engine.apply_update(
        new_memory,
        old_memory,
    )

    print()
    print("Test 4 - Apply update")
    print("Result:", result)

    assert result["updated"] is True

    assert (
        old_memory["metadata"]["status"]
        == "superseded"
    )

    assert (
        old_memory["metadata"]["superseded_by"]
        == "memory-new"
    )

    assert (
        new_memory["metadata"]["status"]
        == "current"
    )

    assert (
        new_memory["metadata"]["updates_memory_id"]
        == "memory-old"
    )

    print()
    print("-" * 60)
    print("All memory update checks passed.")
    print("=" * 60)
    print("MEMORY UPDATE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()