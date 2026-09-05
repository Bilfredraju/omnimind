import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from memory.clustering import MemoryClusteringEngine
from memory.semantic_store import SemanticMemoryStore


def main():

    print("=" * 60)
    print("OMNIMIND MEMORY CLUSTERING INTEGRATION TEST")
    print("=" * 60)

    # --------------------------------------------------------------
    # Create temporary semantic memory store
    # --------------------------------------------------------------

    test_path = (
        "data/memory/"
        "test_clustering_memories.json"
    )

    store = SemanticMemoryStore(
        path=test_path
    )

    store.clear()

    # --------------------------------------------------------------
    # Add related memories
    # --------------------------------------------------------------

    memory_1 = store.add(
        "I decided to use Qdrant as the vector database for OmniMind.",
        metadata={
            "type": "decision"
        },
    )

    memory_2 = store.add(
        "Qdrant will store the vector embeddings for OmniMind.",
        metadata={
            "type": "project"
        },
    )

    # --------------------------------------------------------------
    # Add unrelated memory
    # --------------------------------------------------------------

    memory_3 = store.add(
        "I implemented the user authentication system.",
        metadata={
            "type": "project"
        },
    )

    print("\nMEMORIES STORED")
    print(
        "Memory count:",
        store.count(),
    )

    assert store.count() == 3

    # --------------------------------------------------------------
    # Build clusters
    # --------------------------------------------------------------

    engine = MemoryClusteringEngine(
        threshold=0.72
    )

    clusters = engine.cluster_memories(
        store.memories
    )

    print("\nCLUSTERS")

    for cluster in clusters:

        print(
            f"- {cluster['cluster_id']}"
        )

        print(
            f"  Size: {cluster['size']}"
        )

        print(
            f"  Representative: "
            f"{cluster['representative_text']}"
        )

        print(
            f"  Memories: "
            f"{len(cluster['memory_ids'])}"
        )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    assert len(clusters) >= 2

    all_cluster_memory_ids = []

    for cluster in clusters:
        all_cluster_memory_ids.extend(
            cluster["memory_ids"]
        )

    assert memory_1["memory_id"] in all_cluster_memory_ids
    assert memory_2["memory_id"] in all_cluster_memory_ids
    assert memory_3["memory_id"] in all_cluster_memory_ids

    # --------------------------------------------------------------
    # Find cluster containing first two memories
    # --------------------------------------------------------------

    related_cluster = None

    for cluster in clusters:

        ids = cluster[
            "memory_ids"
        ]

        if (
            memory_1["memory_id"] in ids
            and memory_2["memory_id"] in ids
        ):

            related_cluster = cluster
            break

    assert related_cluster is not None

    assert related_cluster["size"] >= 2

    print(
        "\nRELATED MEMORIES CLUSTERED TOGETHER: True"
    )

    print(
        "Unrelated memory assigned separately: True"
    )

    # --------------------------------------------------------------
    # Persistence check
    # --------------------------------------------------------------

    store.clear()

    print(
        "\nSemantic memory store cleared."
    )

    print(
        "Final memory count:",
        store.count(),
    )

    assert store.count() == 0

    # --------------------------------------------------------------
    # Cleanup test file
    # --------------------------------------------------------------

    test_file = Path(test_path)

    if test_file.exists():
        test_file.unlink()

    print("\n" + "-" * 60)
    print(
        "All memory clustering integration checks passed."
    )
    print("-" * 60)

    print(
        "\nMEMORY CLUSTERING INTEGRATION TEST PASSED"
    )


if __name__ == "__main__":
    main()