"""
Persistent Qdrant vector store for OmniMind.

Supports:
- Persistent local Qdrant storage
- Multi-document indexing
- Globally stable vector IDs
- Document-level filtering
- Deterministic re-indexing
- Document-specific deletion
- Metadata preservation
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    FilterSelector,
    VectorParams,
)


class QdrantVectorStore:
    """
    Persistent local Qdrant vector store for OmniMind.

    Each vector is identified deterministically using:

        document_id + chunk_id

    This prevents vector ID collisions when multiple documents
    are indexed into the same collection.
    """

    def __init__(
        self,
        collection_name: str = "omnimind_documents",
        vector_size: int = 384,
        storage_path: str = "data/vector_store/qdrant",
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size

        storage = Path(storage_path)
        storage.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.storage_path = storage

        self.client = QdrantClient(
            path=str(storage)
        )

        self._create_collection()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _create_collection(self):
        """Create the collection if it doesn't exist."""

        existing_collections = [
            collection.name
            for collection in (
                self.client
                .get_collections()
                .collections
            )
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

    # ------------------------------------------------------------------
    # Stable IDs
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_point_id(
        document_id: str,
        chunk_id: str,
    ) -> str:
        """
        Generate a deterministic UUID for a document chunk.

        The same document_id + chunk_id combination always produces
        the same Qdrant point ID.
        """

        identity = (
            f"{str(document_id).strip()}::"
            f"{str(chunk_id).strip()}"
        )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                identity,
            )
        )

    @staticmethod
    def _fallback_chunk_id(
        chunk: dict[str, Any],
        index: int,
    ) -> str:
        """
        Generate a deterministic chunk ID if the chunk doesn't
        already contain one.
        """

        metadata = chunk.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        text = str(
            chunk.get(
                "text",
                "",
            )
        )

        page = str(
            metadata.get(
                "page",
                metadata.get(
                    "page_number",
                    "",
                ),
            )
        )

        raw_identity = (
            f"{page}::{index}::{text}"
        )

        digest = hashlib.sha256(
            raw_identity.encode("utf-8")
        ).hexdigest()

        return f"chunk-{digest[:16]}"

    @classmethod
    def _get_document_id(
        cls,
        chunk: dict[str, Any],
    ) -> str:
        """Extract document_id from a chunk."""

        metadata = chunk.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        document_id = metadata.get(
            "document_id"
        )

        if not document_id:
            document_id = chunk.get(
                "document_id"
            )

        if not document_id:
            raise ValueError(
                "Chunk is missing required document_id."
            )

        return str(document_id)

    @classmethod
    def _get_chunk_id(
        cls,
        chunk: dict[str, Any],
        index: int,
    ) -> str:
        """Extract chunk_id or create a deterministic fallback."""

        metadata = chunk.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        chunk_id = metadata.get(
            "chunk_id"
        )

        if not chunk_id:
            chunk_id = chunk.get(
                "chunk_id"
            )

        if chunk_id:
            return str(chunk_id)

        return cls._fallback_chunk_id(
            chunk,
            index,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> list[str]:
        """
        Store document chunks and embeddings.

        Existing points with the same deterministic ID are updated
        rather than duplicated.

        Returns:
            List of Qdrant point IDs.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not chunks:
            return []

        points: list[PointStruct] = []
        point_ids: list[str] = []

        for index, (
            chunk,
            embedding,
        ) in enumerate(
            zip(chunks, embeddings)
        ):
            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                raise ValueError(
                    f"Chunk at index {index} "
                    f"contains empty text."
                )

            document_id = self._get_document_id(
                chunk
            )

            chunk_id = self._get_chunk_id(
                chunk,
                index,
            )

            point_id = self._stable_point_id(
                document_id,
                chunk_id,
            )

            metadata = chunk.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            stored_metadata = dict(
                metadata
            )

            stored_metadata[
                "document_id"
            ] = document_id

            stored_metadata[
                "chunk_id"
            ] = chunk_id

            payload = {
                "text": text,
                "metadata": stored_metadata,
                "document_id": document_id,
                "chunk_id": chunk_id,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=list(embedding),
                    payload=payload,
                )
            )

            point_ids.append(point_id)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        print(
            f"Stored {len(points)} chunks in Qdrant."
        )

        return point_ids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search vectors using cosine similarity.

        Optionally restrict results to one document.
        """

        if top_k <= 0:
            return []

        query_filter = None

        if document_id is not None:
            document_id = str(
                document_id
            ).strip()

            if not document_id:
                raise ValueError(
                    "document_id cannot be empty."
                )

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

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_embedding),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results: list[dict[str, Any]] = []

        for rank, point in enumerate(
            response.points,
            start=1,
        ):
            payload = (
                point.payload
                or {}
            )

            metadata = payload.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            result = {
                "result_id": str(point.id),
                "score": float(point.score),
                "rank": rank,
                "text": payload.get(
                    "text",
                    "",
                ),
                "metadata": metadata,
                "document_id": payload.get(
                    "document_id",
                    metadata.get(
                        "document_id"
                    ),
                ),
                "chunk_id": payload.get(
                    "chunk_id",
                    metadata.get(
                        "chunk_id"
                    ),
                ),
            }

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def count(
        self,
        document_id: str | None = None,
    ) -> int:
        """
        Return total vector count.

        If document_id is provided, return only vectors belonging
        to that document.
        """

        if document_id is None:
            result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )

            return int(result.count)

        document_id = str(
            document_id
        ).strip()

        if not document_id:
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

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        )

        return int(result.count)

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        """Return True if the document has indexed vectors."""

        return (
            self.count(
                document_id=document_id
            )
            > 0
        )

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete all vectors belonging to a document.
        """

        document_id = str(
            document_id
        ).strip()

        if not document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

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
            points_selector=FilterSelector(
                filter=query_filter
            ),
            wait=True,
        )

        print(
            f"Deleted vectors for document: "
            f"{document_id}"
        )

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    def collection_info(self) -> dict[str, Any]:
        """Return collection statistics."""

        info = self.client.get_collection(
            collection_name=self.collection_name
        )

        return {
            "collection_name": self.collection_name,
            "vector_size": self.vector_size,
            "distance": "cosine",
            "storage_path": str(
                self.storage_path.resolve()
            ),
            "count": self.count(),
            "points_count": getattr(
                info,
                "points_count",
                None,
            ),
            "indexed_vectors_count": getattr(
                info,
                "indexed_vectors_count",
                None,
            ),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the Qdrant client cleanly."""

        if self.client is not None:
            self.client.close()
            self.client = None