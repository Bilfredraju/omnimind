import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from memory.clustering import MemoryClusteringEngine


def main():

    engine = MemoryClusteringEngine(
        threshold=0.80
    )

    # --------------------------------------------------------------
    # Synthetic embeddings
    # --------------------------------------------------------------

    memory_1 = {
        "memory_id": "memory-1",
        "text": "I decided to use Qdrant for OmniMind.",
        "embedding": [1.0, 0.0, 0.0],
    }

    memory_2 = {
        "memory_id": "memory-2",
        "text": "Qdrant will store the vector embeddings.",
        "embedding": [0.98, 0.05, 0.0],
    }

    memory_3 = {
        "memory_id": "memory-3",
        "text": "I implemented the user login system.",
        "embedding": [0.0, 1.0, 0.0],
    }

    # --------------------------------------------------------------
    # Test 1 — Similar memories
    # --------------------------------------------------------------

    print("\nTEST 1: Similar memories")

    clusters = engine.cluster_memories(
        [
            memory_1,
            memory_2,
        ]
    )

    assert len(clusters) == 1

    assert clusters[0]["size"] == 2

    print(
        "Clusters:",
        len(clusters),
    )

    print(
        "Cluster size:",
        clusters[0]["size"],
    )

    # --------------------------------------------------------------
    # Test 2 — Different memories
    # --------------------------------------------------------------

    print("\nTEST 2: Different memories")

    clusters = engine.cluster_memories(
        [
            memory_1,
            memory_3,
        ]
    )

    assert len(clusters) == 2

    print(
        "Clusters:",
        len(clusters),
    )

    # --------------------------------------------------------------
    # Test 3 — Cluster assignment
    # --------------------------------------------------------------

    print("\nTEST 3: Cluster assignment")

    clusters = engine.cluster_memories(
        [
            memory_1,
            memory_2,
            memory_3,
        ]
    )

    assert len(clusters) == 2

    sizes = sorted(
        cluster["size"]
        for cluster in clusters
    )

    assert sizes == [1, 2]

    print(
        "Cluster sizes:",
        sizes,
    )

    # --------------------------------------------------------------
    # Test 4 — Empty embedding
    # --------------------------------------------------------------

    print("\nTEST 4: Empty embedding")

    memory_4 = {
        "memory_id": "memory-4",
        "text": "Memory without embedding.",
        "embedding": [],
    }

    result = engine.find_cluster(
        memory_4,
        clusters,
    )

    assert result is None

    print(
        "Empty embedding handled correctly."
    )

    print(
        "\nMEMORY CLUSTERING TEST PASSED"
    )


if __name__ == "__main__":
    main()