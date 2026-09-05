from __future__ import annotations

from typing import Any


class MemoryClusteringEngine:
    """
    Groups semantically related memories into clusters.

    Clustering is based on cosine similarity between memory
    embeddings.

    The engine does not delete, modify, supersede, or contradict
    memories. It only determines logical groups of related memories.

    Each cluster maintains a representative embedding. The
    representative is updated whenever a new memory joins the cluster.
    """

    def __init__(
        self,
        threshold: float = 0.72,
    ):
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        if not vector_a or not vector_b:
            return 0.0

        if len(vector_a) != len(vector_b):
            return 0.0

        dot_product = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b,
            )
        )

        magnitude_a = sum(
            a * a
            for a in vector_a
        ) ** 0.5

        magnitude_b = sum(
            b * b
            for b in vector_b
        ) ** 0.5

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )

    # ------------------------------------------------------------------
    # Find cluster
    # ------------------------------------------------------------------

    def find_cluster(
        self,
        memory: dict[str, Any],
        clusters: list[dict[str, Any]],
    ) -> dict[str, Any] | None:

        embedding = memory.get(
            "embedding"
        )

        if not embedding:
            return None

        best_cluster = None
        best_similarity = -1.0

        for cluster in clusters:

            representative = cluster.get(
                "representative_embedding"
            )

            if not representative:
                continue

            similarity = self.cosine_similarity(
                embedding,
                representative,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if (
            best_cluster is not None
            and best_similarity >= self.threshold
        ):

            return {
                "cluster": best_cluster,
                "similarity": round(
                    best_similarity,
                    4,
                ),
            }

        return None

    # ------------------------------------------------------------------
    # Create cluster
    # ------------------------------------------------------------------

    def create_cluster(
        self,
        memory: dict[str, Any],
    ) -> dict[str, Any]:

        memory_id = memory.get(
            "memory_id"
        )

        text = memory.get(
            "text",
            "",
        )

        embedding = memory.get(
            "embedding",
            [],
        )

        return {
            "cluster_id": self._generate_cluster_id(
                memory_id
            ),
            "representative_memory_id": memory_id,
            "representative_text": text,
            "representative_embedding": list(
                embedding
            ),
            "memory_ids": (
                [memory_id]
                if memory_id
                else []
            ),
            "size": 1,
        }

    # ------------------------------------------------------------------
    # Update representative embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _update_representative(
        cluster: dict[str, Any],
        new_embedding: list[float],
    ):

        old_embedding = cluster.get(
            "representative_embedding"
        )

        if not old_embedding:
            cluster["representative_embedding"] = list(
                new_embedding
            )
            return

        if len(old_embedding) != len(new_embedding):
            return

        size = cluster.get(
            "size",
            1,
        )

        # Weighted running average.
        #
        # Existing representative represents `size` memories.
        # New memory contributes one additional vector.

        updated_embedding = [
            (
                old_value * size
                + new_value
            )
            / (size + 1)
            for old_value, new_value
            in zip(
                old_embedding,
                new_embedding,
            )
        ]

        cluster[
            "representative_embedding"
        ] = updated_embedding

    # ------------------------------------------------------------------
    # Add memory to cluster
    # ------------------------------------------------------------------

    def add_to_cluster(
        self,
        memory: dict[str, Any],
        cluster: dict[str, Any],
    ) -> dict[str, Any]:

        memory_id = memory.get(
            "memory_id"
        )

        embedding = memory.get(
            "embedding"
        )

        if memory_id is not None:

            if memory_id not in cluster[
                "memory_ids"
            ]:

                previous_size = cluster.get(
                    "size",
                    len(
                        cluster[
                            "memory_ids"
                        ]
                    ),
                )

                cluster[
                    "memory_ids"
                ].append(
                    memory_id
                )

                # Update representative before changing size.
                if embedding:
                    self._update_representative(
                        cluster=cluster,
                        new_embedding=embedding,
                    )

                cluster["size"] = previous_size + 1

        return cluster

    # ------------------------------------------------------------------
    # Cluster memories
    # ------------------------------------------------------------------

    def cluster_memories(
        self,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        clusters: list[
            dict[str, Any]
        ] = []

        for memory in memories:

            if not memory.get(
                "embedding"
            ):
                continue

            result = self.find_cluster(
                memory=memory,
                clusters=clusters,
            )

            if result is None:

                cluster = self.create_cluster(
                    memory
                )

                clusters.append(
                    cluster
                )

            else:

                self.add_to_cluster(
                    memory=memory,
                    cluster=result["cluster"],
                )

        return clusters

    # ------------------------------------------------------------------
    # Generate deterministic cluster ID
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_cluster_id(
        memory_id: str | None,
    ) -> str:

        if memory_id:
            return f"cluster-{memory_id}"

        return "cluster-unknown"