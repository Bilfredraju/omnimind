import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.semantic_store import SemanticMemoryStore


def main():
    print("=" * 60)
    print("OMNIMIND MEMORY UPDATE INTEGRATION TEST")
    print("=" * 60)

    store = SemanticMemoryStore()
    store.clear()

    # --------------------------------------------------------------
    # Original decision
    # --------------------------------------------------------------

    old = store.add(
        "I decided to use Qdrant as the vector database for my OmniMind project.",
        {
            "type": "decision",
            "source": "conversation",
        },
    )

    print()
    print("ORIGINAL MEMORY")
    print("Text:", old["text"])
    print("Memory ID:", old["memory_id"])
    print("Status:", old["metadata"]["status"])
    print("Importance:", old["metadata"]["importance"])

    assert old["updated"] is False
    assert old["duplicate"] is False
    assert old["metadata"]["status"] == "current"
    assert store.count() == 1

    # --------------------------------------------------------------
    # Changed decision
    # --------------------------------------------------------------

    new = store.add(
        "I changed my decision. I'll use PostgreSQL instead of Qdrant for my OmniMind project.",
        {
            "type": "decision",
            "source": "conversation",
        },
    )

    print()
    print("UPDATED MEMORY")
    print("Text:", new["text"])
    print("Memory ID:", new["memory_id"])
    print("Updated:", new["updated"])
    print("Old memory ID:", new["old_memory_id"])
    print("Similarity:", new["duplicate_similarity"])
    print("Status:", new["metadata"]["status"])

    # --------------------------------------------------------------
    # Verify update
    # --------------------------------------------------------------

    assert new["updated"] is True
    assert new["duplicate"] is False
    assert new["old_memory_id"] == old["memory_id"]
    assert new["metadata"]["status"] == "current"
    assert (
        new["metadata"]["updates_memory_id"]
        == old["memory_id"]
    )

    # Both historical and current memories must exist.
    assert store.count() == 2

    # --------------------------------------------------------------
    # Verify old memory was preserved
    # --------------------------------------------------------------

    stored_old = next(
        memory
        for memory in store.memories
        if memory["memory_id"] == old["memory_id"]
    )

    print()
    print("OLD MEMORY AFTER UPDATE")
    print("Status:", stored_old["metadata"]["status"])
    print(
        "Superseded by:",
        stored_old["metadata"]["superseded_by"],
    )

    assert (
        stored_old["metadata"]["status"]
        == "superseded"
    )

    assert (
        stored_old["metadata"]["superseded_by"]
        == new["memory_id"]
    )

    # --------------------------------------------------------------
    # Verify new memory
    # --------------------------------------------------------------

    stored_new = next(
        memory
        for memory in store.memories
        if memory["memory_id"] == new["memory_id"]
    )

    assert (
        stored_new["metadata"]["status"]
        == "current"
    )

    # --------------------------------------------------------------
    # Persistence test
    # --------------------------------------------------------------

    reloaded = SemanticMemoryStore()

    assert reloaded.count() == 2

    reloaded_old = next(
        memory
        for memory in reloaded.memories
        if memory["memory_id"] == old["memory_id"]
    )

    reloaded_new = next(
        memory
        for memory in reloaded.memories
        if memory["memory_id"] == new["memory_id"]
    )

    assert (
        reloaded_old["metadata"]["status"]
        == "superseded"
    )

    assert (
        reloaded_new["metadata"]["status"]
        == "current"
    )

    print()
    print("Persistence verification:")
    print("Historical memory preserved: True")
    print("Current memory preserved: True")

    print()
    print("-" * 60)
    print("All memory update integration checks passed.")
    print("=" * 60)
    print("MEMORY UPDATE INTEGRATION TEST PASSED")
    print("=" * 60)

    reloaded.clear()


if __name__ == "__main__":
    main()