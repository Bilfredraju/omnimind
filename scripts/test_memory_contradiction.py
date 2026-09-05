import sys
from pathlib import Path

# Add OmniMind project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.contradiction import MemoryContradictionEngine


def main():
    print("=" * 60)
    print("OMNIMIND MEMORY CONTRADICTION TEST")
    print("=" * 60)

    engine = MemoryContradictionEngine()

    existing_memory = {
        "memory_id": "memory-1",
        "text": (
            "I decided to use Qdrant as the vector database "
            "for my OmniMind project."
        ),
        "metadata": {
            "type": "decision",
            "status": "current",
        },
    }

    # ---------------------------------------------------------
    # Test 1: Contradictory technology decision
    # ---------------------------------------------------------
    result = engine.detect(
        "I decided to use PostgreSQL as the vector database "
        "for my OmniMind project.",
        existing_memory,
    )

    print("\nTEST 1: Conflicting technology")
    print("Contradiction:", result["contradiction"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

    assert result["contradiction"] is True
    assert result["confidence"] >= 0.90

    # ---------------------------------------------------------
    # Test 2: Same technology = no contradiction
    # ---------------------------------------------------------
    result = engine.detect(
        "I decided to use Qdrant as the vector database "
        "for my OmniMind project.",
        existing_memory,
    )

    print("\nTEST 2: Same decision")
    print("Contradiction:", result["contradiction"])
    print("Confidence:", result["confidence"])

    assert result["contradiction"] is False

    # ---------------------------------------------------------
    # Test 3: Unrelated statement
    # ---------------------------------------------------------
    result = engine.detect(
        "I created a Python script for testing the RAG pipeline.",
        existing_memory,
    )

    print("\nTEST 3: Unrelated memory")
    print("Contradiction:", result["contradiction"])

    assert result["contradiction"] is False

    # ---------------------------------------------------------
    # Test 4: Explicit positive/negative conflict
    # ---------------------------------------------------------
    negative_memory = {
        "memory_id": "memory-2",
        "text": "I will not use Qdrant for the project.",
        "metadata": {
            "type": "decision",
            "status": "current",
        },
    }

    result = engine.detect(
        "I will use Qdrant for the project.",
        negative_memory,
    )

    print("\nTEST 4: Positive vs negative")
    print("Contradiction:", result["contradiction"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

    assert result["contradiction"] is True

    print("\n" + "=" * 60)
    print("MEMORY CONTRADICTION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()