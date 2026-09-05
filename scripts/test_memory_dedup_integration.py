import sys
from pathlib import Path

# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.semantic_store import SemanticMemoryStore


def main():
    print("=" * 60)
    print("OMNIMIND SEMANTIC MEMORY DEDUPLICATION INTEGRATION TEST")
    print("=" * 60)

    store = SemanticMemoryStore()

    # Start with a clean store.
    store.clear()

    print()
    print("Initial memory count:", store.count())

    # --------------------------------------------------------------
    # Memory 1
    # --------------------------------------------------------------

    first = store.add(
        "I decided to use Qdrant as the vector database for my OmniMind project.",
        {
            "type": "decision",
            "source": "conversation",
        },
    )

    print()
    print("Memory 1 stored:")
    print("Text:", first["text"])
    print("Duplicate:", first["duplicate"])
    print("Memory count:", store.count())

    assert first["duplicate"] is False
    assert store.count() == 1

    # --------------------------------------------------------------
    # Memory 2 - semantically similar
    # --------------------------------------------------------------

    second = store.add(
        "For my OmniMind project, I chose Qdrant for the vector database.",
        {
            "type": "decision",
            "source": "conversation",
        },
    )

    print()
    print("Memory 2:")
    print("Text:", second["text"])
    print("Duplicate:", second["duplicate"])
    print("Duplicate similarity:", second["duplicate_similarity"])
    print("Duplicate of:", second["duplicate_of"])
    print("Memory count:", store.count())

    # The important assertion:
    # the second memory should not create another stored record.
    assert second["duplicate"] is True
    assert second["duplicate_of"] == first["memory_id"]
    assert store.count() == 1

    # --------------------------------------------------------------
    # Memory 3 - clearly different
    # --------------------------------------------------------------

    third = store.add(
        "I decided to use PostgreSQL for application metadata.",
        {
            "type": "decision",
            "source": "conversation",
        },
    )

    print()
    print("Memory 3:")
    print("Text:", third["text"])
    print("Duplicate:", third["duplicate"])
    print("Memory count:", store.count())

    assert third["duplicate"] is False
    assert store.count() == 2

    print()
    print("-" * 60)
    print("All integration checks passed.")
    print("=" * 60)
    print("MEMORY DEDUPLICATION INTEGRATION TEST PASSED")
    print("=" * 60)

    store.clear()


if __name__ == "__main__":
    main()