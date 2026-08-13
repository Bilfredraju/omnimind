from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantVectorStore:
    """Local Qdrant vector store for OmniMind."""

    def __init__(
        self,
        collection_name: str = "omnimind_documents",
        vector_size: int = 384,
        storage_path: str = "data/vector_store/qdrant",
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size

        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(
            path=str(storage)
        )

        self._create_collection()

    def _create_collection(self):
        """Create the collection if it doesn't exist."""

        existing_collections = [
            collection.name
            for collection in self.client.get_collections().collections
        ]

        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

            print(
                f"Created Qdrant collection: "
                f"{self.collection_name}"
            )

        else:
            print(
                f"Qdrant collection already exists: "
                f"{self.collection_name}"
            )

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        """Store document chunks and their embeddings."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            payload = {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }

            points.append(
                PointStruct(
                    id=index,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(
            f"Stored {len(points)} chunks in Qdrant."
        )

    def count(self) -> int:
        """Return the number of stored vectors."""

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count