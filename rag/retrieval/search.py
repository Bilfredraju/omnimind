from pathlib import Path

from qdrant_client import QdrantClient

from rag.embeddings.embedder import EmbeddingModel


class SemanticRetriever:
    """Retrieve relevant document chunks from Qdrant."""

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

        self.embedder = EmbeddingModel()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search Qdrant for chunks relevant to the query."""

        query_embedding = self.embedder.encode_single(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        ).points

        retrieved_documents = []

        for result in results:
            payload = result.payload

            retrieved_documents.append(
                {
                    "score": result.score,
                    "text": payload["text"],
                    "metadata": payload["metadata"],
                }
            )

        return retrieved_documents

    def close(self):
        """Close the Qdrant client cleanly."""
        self.client.close()