"""
Persistent Qdrant vector store for OmniMind.

Supports:
- Persistent local Qdrant storage
- Globally stable point IDs
- Multi-document indexing
- Document-level filtering
- Deterministic re-indexing
- Metadata preservation
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantVectorStore:
    """
    Persistent local Qdrant vector store.

    Vector IDs are derived deterministically from the document_id and
    chunk_id, preventing collisions between different documents.
    """

    DEFAULT_COLLECTION_NAME = "omnimind_documents"
    DEFAULT_VECTOR_SIZE = 384
    DEFAULT_STORAGE_PATH = "data/vector_store/qdrant"

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        storage_path: str | Path = DEFAULT_STORAGE_PATH,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.storage_path = Path(storage_path)

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = QdrantClient(
            path=str(self.storage_path)
        )

        self._create_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _create_collection(self) -> None:
        """
        Create the collection if it does not already exist.
        """
        collections = self.client.get_collections()

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    # ------------------------------------------------------------------
    # Stable ID generation
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
            f"{document_id.strip()}::{chunk_id.strip()}"
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
        Generate a deterministic chunk ID when the ingestion pipeline
        does not provide one.
        """
        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        text = str(
            chunk.get("text", "")
        )

        page = str(
            metadata.get(
                "page",
                metadata.get("page_number", ""),
            )
        )

        raw_identity = (
            f"{page}::{index}::{text}"
        )

        return hashlib.sha256(
            raw_identity.encode("utf-8")
        ).hexdigest()[:24]

    @classmethod
    def _get_document_id(
        cls,
        chunk: dict[str, Any],
    ) -> str:
        """
        Extract document_id from chunk metadata.
        """
        metadata = chunk.get("metadata", {})

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
        """
        Extract chunk_id from chunk metadata or generate a fallback.
        """
        metadata = chunk.get("metadata", {})

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
    # Document indexing
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[str]:
        """
        Add or update document chunks in Qdrant.

        Because point IDs are deterministic, indexing the same chunk
        again updates the existing point rather than creating a
        duplicate.

        Args:
            chunks:
                Chunk dictionaries containing text and metadata.

            embeddings:
                Embedding vectors corresponding to chunks.

        Returns:
            List of Qdrant point IDs.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return []

        points: list[models.PointStruct] = []
        point_ids: list[str] = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            text = str(
                chunk.get("text", "")
            ).strip()

            if not text:
                raise ValueError(
                    f"Chunk at index {index} contains empty text."
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

            # Preserve the original metadata while guaranteeing the
            # identity fields exist at the top level of metadata.
            stored_metadata = dict(metadata)

            stored_metadata["document_id"] = (
                document_id
            )
            stored_metadata["chunk_id"] = (
                chunk_id
            )

            payload = {
                "text": text,
                "metadata": stored_metadata,
                "document_id": document_id,
                "chunk_id": chunk_id,
            }

            points.append(
                models.PointStruct(
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
        Search the vector store.

        Args:
            query_embedding:
                Query embedding vector.

            top_k:
                Number of results to return.

            document_id:
                Optional document-level filter.

        Returns:
            Ranked search results.
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

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
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

        results = []

        for rank, point in enumerate(
            response.points,
            start=1,
        ):
            payload = point.payload or {}

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
    # Document-specific helpers
    # ------------------------------------------------------------------

    def count(
        self,
        document_id: str | None = None,
    ) -> int:
        """
        Return the number of indexed vectors.

        If document_id is provided, return only vectors belonging
        to that document.
        """
        if document_id is None:
            return int(
                self.client.count(
                    collection_name=self.collection_name,
                    exact=True,
                ).count
            )

        document_id = str(
            document_id
        ).strip()

        if not document_id:
            return 0

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    ),
                )
            ]
        )

        return int(
            self.client.count(
                collection_name=self.collection_name,
                count_filter=query_filter,
                exact=True,
            ).count
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

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    ),
                )
            ]
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=query_filter
            ),
            wait=True,
        )

    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        """
        Check whether a document has indexed vectors.
        """
        return self.count(
            document_id=document_id
        ) > 0

    # ------------------------------------------------------------------
    # Collection information
    # ------------------------------------------------------------------

    def collection_info(self) -> dict[str, Any]:
        """
        Return useful collection statistics.
        """
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

    def close(self) -> None:
        """
        Close the Qdrant client.
        """
        self.client.close()