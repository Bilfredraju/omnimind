import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.importance import MemoryImportanceEngine


def main():
    print("=" * 60)
    print("OMNIMIND MEMORY IMPORTANCE TEST")
    print("=" * 60)

    engine = MemoryImportanceEngine()

    memories = [
        {
            "text": "I decided to use Qdrant as the vector database for my OmniMind project.",
            "metadata": {"type": "decision"},
        },
        {
            "text": "I prefer concise answers.",
            "metadata": {"type": "preference"},
        },
        {
            "text": "I want to build an AI assistant.",
            "metadata": {"type": "goal"},
        },
        {
            "text": "RAG combines retrieval with language generation.",
            "metadata": {"type": "technical"},
        },
        {
            "text": "Hello, how are you?",
            "metadata": {"type": "general"},
        },
    ]

    scores = {}

    for item in memories:
        score = engine.calculate(
            item["text"],
            item["metadata"],
        )

        memory_type = item["metadata"]["type"]
        scores[memory_type] = score

        print()
        print("Type:", memory_type)
        print("Text:", item["text"])
        print("Importance:", score)

    print()
    print("-" * 60)

    # Importance ordering
    assert scores["decision"] > scores["preference"]
    assert scores["decision"] > scores["technical"]
    assert scores["goal"] > scores["general"]
    assert scores["preference"] > scores["general"]

    # Decision should receive very high importance.
    assert scores["decision"] >= 0.9

    # Every score must stay between 0 and 1.
    for score in scores.values():
        assert 0.0 <= score <= 1.0

    print("All importance checks passed.")
    print("=" * 60)
    print("MEMORY IMPORTANCE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()