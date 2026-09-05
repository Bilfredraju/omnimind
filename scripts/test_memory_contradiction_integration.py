import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.semantic_store import SemanticMemoryStore


def main():
    print("=" * 60)
    print("OMNIMIND CONTRADICTION INTEGRATION TEST")
    print("=" * 60)

    store = SemanticMemoryStore()

    # Start with clean memory for this test.
    store.clear()

    # ---------------------------------------------------------
    # 1. Store original decision
    # ---------------------------------------------------------
    original = store.add(
        "I decided to use Qdrant as the vector database for my OmniMind project.",
        metadata={
            "type": "decision",
            "source": "contradiction_test",
        },
    )

    print("\nORIGINAL MEMORY")
    print("Text:", original["text"])
    print("Memory ID:", original["memory_id"])
    print("Status:", original["metadata"]["status"])

    assert original["duplicate"] is False
    assert original["updated"] is False

    # ---------------------------------------------------------
    # 2. Add conflicting decision
    # ---------------------------------------------------------
    conflicting = store.add(
        "I decided to use PostgreSQL as the vector database for my OmniMind project.",
        metadata={
            "type": "decision",
            "source": "contradiction_test",
        },
    )

    print("\nCONFLICTING MEMORY")
    print("Text:", conflicting["text"])
    print("Memory ID:", conflicting["memory_id"])
    print("Contradiction:", conflicting["contradiction"])
    print("Contradicts:", conflicting["contradiction_with"])
    print("Confidence:", conflicting["contradiction_confidence"])
    print("Reason:", conflicting["contradiction_reason"])

    assert conflicting["duplicate"] is False
    assert conflicting["updated"] is False
    assert conflicting["contradiction"] is True
    assert conflicting["contradiction_with"] == original["memory_id"]
    assert conflicting["contradiction_confidence"] >= 0.90

    # ---------------------------------------------------------
    # 3. Verify both memories remain
    # ---------------------------------------------------------
    assert store.count() == 2

    memories = store.memories

    original_after = next(
        memory
        for memory in memories
        if memory["memory_id"] == original["memory_id"]
    )

    conflicting_after = next(
        memory
        for memory in memories
        if memory["memory_id"] == conflicting["memory_id"]
    )

    print("\nMEMORY PRESERVATION")
    print(
        "Original preserved:",
        original_after["text"]
        == "I decided to use Qdrant as the vector database for my OmniMind project.",
    )

    print(
        "Conflicting preserved:",
        conflicting_after["text"]
        == "I decided to use PostgreSQL as the vector database for my OmniMind project.",
    )

    assert original_after["metadata"]["status"] == "current"
    assert conflicting_after["metadata"]["status"] == "current"

    # Contradiction must NOT supersede either memory.
    assert original_after["metadata"]["status"] != "superseded"
    assert conflicting_after["metadata"]["status"] != "superseded"

    print("\n" + "-" * 60)
    print("All contradiction integration checks passed.")
    print("-" * 60)

    store.clear()

    print("\n" + "=" * 60)
    print("MEMORY CONTRADICTION INTEGRATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()