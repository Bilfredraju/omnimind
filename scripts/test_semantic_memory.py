import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.semantic_store import SemanticMemoryStore


def main():
    print("=" * 60)
    print("OMNIMIND SEMANTIC MEMORY TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        memory_path = (
            Path(temp_dir)
            / "semantic_memories.json"
        )

        store = SemanticMemoryStore(
            path=str(memory_path)
        )

        print("\nAdding semantic memories...")

        store.add(
            "RAG combines retrieval with language generation.",
            metadata={
                "topic": "RAG",
                "type": "technical",
            },
        )

        store.add(
            "Hybrid retrieval combines semantic search and keyword search.",
            metadata={
                "topic": "retrieval",
                "type": "technical",
            },
        )

        store.add(
            "Cross-encoder reranking improves the ordering of retrieved documents.",
            metadata={
                "topic": "reranking",
                "type": "technical",
            },
        )

        store.add(
            "LangGraph can orchestrate multiple agents through a shared state.",
            metadata={
                "topic": "agents",
                "type": "technical",
            },
        )

        print("Memories stored:", store.count())

        print("\nSearching semantic memory...")

        results = store.search(
            "How does RAG retrieve information?",
            top_k=3,
        )

        print(
            "\nQuery:",
            "How does RAG retrieve information?",
        )

        for index, result in enumerate(
            results,
            start=1,
        ):
            print(f"\nResult {index}")
            print("Score:", round(result["score"], 4))
            print("Text:", result["text"])
            print("Metadata:", result["metadata"])

        assert store.count() == 4
        assert len(results) > 0

        print("\nTesting persistence...")

        new_store = SemanticMemoryStore(
            path=str(memory_path)
        )

        assert new_store.count() == 4

        print(
            "Reloaded memories:",
            new_store.count(),
        )

        print("\nTesting clear...")

        new_store.clear()

        assert new_store.count() == 0

        print(
            "Memories after clear:",
            new_store.count(),
        )

        print("\n" + "=" * 60)
        print("SEMANTIC MEMORY TEST PASSED")
        print("=" * 60)


if __name__ == "__main__":
    main()