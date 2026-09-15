from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COLLECTION_NAME = "omnimind_documents"
VECTOR_SIZE = 384

# Persistent Qdrant server.
QDRANT_URL = "http://127.0.0.1:6333"


class QdrantVectorStore:
    """
    Persistent multi-document vector store backed by a Qdrant server.

    The Qdrant server owns the underlying persistent storage.
    Multiple OmniMind processes can safely connect to it.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        url: str = QDRANT_URL,
    ):
        self.collection_name = collection_name
        self.url = url

        self.client = QdrantClient(
            url=self.url,
        )

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections

        if any(
            collection.name == self.collection_name
            for collection in collections
        ):
            print(
                f"Qdrant collection already exists: "
                f"{self.collection_name}"
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created Qdrant collection: "
            f"{self.collection_name}"
        )

    # ------------------------------------------------------------------
    # Stable point IDs
    # ------------------------------------------------------------------

    @staticmethod
    def _point_id(
        document_id: str,
        chunk_id: str,
    ) -> str:
        """
        Generate a deterministic UUID from document + chunk IDs.
        """

        return str(
            uuid5(
                NAMESPACE_URL,
                f"{document_id}::{chunk_id}",
            )
        )

    # ------------------------------------------------------------------
    # Add documents
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks and embeddings must have the same length"
            )

        if not chunks:
            return 0

        points: list[PointStruct] = []

        for chunk, embedding in zip(chunks, embeddings):
            metadata = dict(
                chunk.get("metadata", {})
            )

            document_id = metadata.get("document_id")
            chunk_id = metadata.get("chunk_id")

            if not document_id:
                raise ValueError(
                    "Each chunk must contain metadata.document_id"
                )

            if not chunk_id:
                raise ValueError(
                    "Each chunk must contain metadata.chunk_id"
                )

            point_id = self._point_id(
                document_id=document_id,
                chunk_id=chunk_id,
            )

            payload = {
                "text": chunk.get("text", ""),
                "metadata": metadata,
                "document_id": document_id,
                "chunk_id": chunk_id,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        query_filter = None

        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        output: list[dict[str, Any]] = []

        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}

            metadata = dict(
                payload.get("metadata", {})
            )

            output.append(
                {
                    "result_id": str(result.id),
                    "rank": rank,
                    "score": float(result.score),
                    "text": payload.get("text", ""),
                    "metadata": metadata,
                    "document_id": payload.get(
                        "document_id",
                        metadata.get("document_id"),
                    ),
                    "chunk_id": payload.get(
                        "chunk_id",
                        metadata.get("chunk_id"),
                    ),
                    "document_name": metadata.get(
                        "document_name"
                    ),
                    "page": metadata.get("page"),
                }
            )

        return output

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def count(
        self,
        document_id: Optional[str] = None,
    ) -> int:
        query_filter = None

        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )

        return self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        ).count

    # ------------------------------------------------------------------
    # Document existence
    # ------------------------------------------------------------------

    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        return self.count(
            document_id=document_id
        ) > 0

    # ------------------------------------------------------------------
    # Delete document
    # ------------------------------------------------------------------

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        existing_count = self.count(
            document_id=document_id
        )

        if existing_count == 0:
            return 0

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id
                    ),
                )
            ]
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=query_filter,
            wait=True,
        )

        return existing_count

    # ------------------------------------------------------------------
    # Collection information
    # ------------------------------------------------------------------

    def collection_info(self) -> dict[str, Any]:
        info = self.client.get_collection(
            self.collection_name
        )

        return {
            "collection_name": self.collection_name,
            "vector_size": VECTOR_SIZE,
            "distance": "cosine",
            "storage_path": None,
            "qdrant_url": self.url,
            "count": self.count(),
            "points_count": info.points_count,
            "indexed_vectors_count": (
                info.indexed_vectors_count
            ),
        }

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.client.close()