import sys
from pathlib import Path

# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.deduplication import MemoryDeduplicationEngine


def main():
    print("=" * 60)
    print("OMNIMIND MEMORY DEDUPLICATION TEST")
    print("=" * 60)

    engine = MemoryDeduplicationEngine(
        threshold=0.88
    )

    # Simple normalized vectors make the test deterministic.
    existing_memory = {
        "memory_id": "memory-001",
        "text": "I decided to use Qdrant as the vector database.",
        "embedding": [1.0, 0.0, 0.0],
        "metadata": {
            "type": "decision"
        },
    }

    memories = [existing_memory]

    print()
    print("Existing memory:")
    print(existing_memory["text"])

    # --------------------------------------------------------------
    # Test 1: Identical embedding
    # --------------------------------------------------------------

    identical_embedding = [1.0, 0.0, 0.0]

    result = engine.find_duplicate(
        identical_embedding,
        memories,
    )

    assert result is not None
    assert result["memory"]["memory_id"] == "memory-001"
    assert result["similarity"] == 1.0

    print()
    print("Test 1 - Identical memory")
    print("Duplicate:", True)
    print("Similarity:", result["similarity"])

    # --------------------------------------------------------------
    # Test 2: Clearly different memory
    # --------------------------------------------------------------

    different_embedding = [0.0, 1.0, 0.0]

    result = engine.find_duplicate(
        different_embedding,
        memories,
    )

    assert result is None

    print()
    print("Test 2 - Different memory")
    print("Duplicate:", False)

    # --------------------------------------------------------------
    # Test 3: Similarity below threshold
    # --------------------------------------------------------------

    below_threshold = [0.7, 0.71414284, 0.0]

    result = engine.find_duplicate(
        below_threshold,
        memories,
    )

    assert result is None

    print()
    print("Test 3 - Below duplicate threshold")
    print("Duplicate:", False)

    print()
    print("-" * 60)
    print("All deduplication checks passed.")
    print("=" * 60)
    print("MEMORY DEDUPLICATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()