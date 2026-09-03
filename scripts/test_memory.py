import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.manager import MemoryManager
from memory.store import ConversationStore


def main():
    print("=" * 60)
    print("OMNIMIND MEMORY SYSTEM TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        memory_path = Path(temp_dir) / "conversations.json"

        store = ConversationStore(
            path=str(memory_path)
        )

        manager = MemoryManager(store=store)

        print("\nAdding conversation turns...")

        manager.remember(
            "What is RAG?",
            "RAG combines retrieval with language generation.",
        )

        manager.remember(
            "What is hybrid retrieval?",
            "Hybrid retrieval combines semantic and keyword search.",
        )

        manager.remember(
            "What does reranking do?",
            "Reranking improves the ordering of retrieved candidates.",
        )

        print("Turns stored:", manager.count())

        print("\nRecent memory:")

        for index, turn in enumerate(
            manager.recall(limit=2),
            start=1,
        ):
            print(f"\nTurn {index}")
            print("User:", turn["user"])
            print("Assistant:", turn["assistant"])
            print("Timestamp:", turn["timestamp"])

        assert manager.count() == 3

        recent = manager.recall(limit=2)

        assert len(recent) == 2
        assert recent[0]["user"] == "What is hybrid retrieval?"
        assert recent[1]["user"] == "What does reranking do?"

        print("\nTesting persistence...")

        new_store = ConversationStore(
            path=str(memory_path)
        )

        new_manager = MemoryManager(
            store=new_store
        )

        assert new_manager.count() == 3

        print(
            "Reloaded turns:",
            new_manager.count(),
        )

        print("\nTesting clear...")

        new_manager.clear()

        assert new_manager.count() == 0

        print("Turns after clear:", new_manager.count())

        print("\n" + "=" * 60)
        print("MEMORY TEST PASSED")
        print("=" * 60)


if __name__ == "__main__":
    main()