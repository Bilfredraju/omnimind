from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from rag.embeddings.embedder import EmbeddingModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION_NAME = "omnimind_documents"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_VECTOR_SIZE = 384


# ---------------------------------------------------------------------------
# Semantic Retriever
# ---------------------------------------------------------------------------

class SemanticRetriever:
    """
    Semantic document retriever backed by persistent Qdrant Server.

    Supports:
        - Global semantic search
        - Document-specific semantic search
        - Persistent Qdrant storage
        - Metadata-preserving results
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        qdrant_url: str = DEFAULT_QDRANT_URL,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.qdrant_url = qdrant_url

        # ---------------------------------------------------------------
        # Connect to persistent Qdrant Server
        # ---------------------------------------------------------------

        self.client = QdrantClient(
            url=self.qdrant_url,
        )

        # ---------------------------------------------------------------
        # Embedding model
        # ---------------------------------------------------------------

        self.embedder = EmbeddingModel()

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        return query

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if top_k > 50:
            raise ValueError(
                "top_k cannot exceed 50."
            )

        return top_k

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search Qdrant for semantically relevant document chunks.

        Args:
            query:
                Natural-language search query.

            top_k:
                Number of results to return.

            document_id:
                Optional document ID.

                If supplied, retrieval is restricted to vectors
                belonging to that document.

                If omitted, retrieval searches the entire collection.

        Returns:
            List of normalized retrieval results.
        """

        # ---------------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------------

        query = self._validate_query(query)
        top_k = self._validate_top_k(top_k)

        if document_id is not None:
            document_id = str(document_id).strip()

            if not document_id:
                raise ValueError(
                    "document_id cannot be empty."
                )

        # ---------------------------------------------------------------
        # Encode query
        # ---------------------------------------------------------------

        query_embedding = self.embedder.encode_single(
            query
        )

        # ---------------------------------------------------------------
        # Build optional document filter
        # ---------------------------------------------------------------

        query_filter = None

        if document_id is not None:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchValue,
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

        # ---------------------------------------------------------------
        # Query persistent Qdrant Server
        # ---------------------------------------------------------------

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results = response.points

        # ---------------------------------------------------------------
        # Normalize results
        # ---------------------------------------------------------------

        retrieved_documents: list[dict[str, Any]] = []

        for result in results:
            payload = result.payload or {}

            metadata = payload.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            # Preserve document/chunk identifiers.
            result_document_id = (
                payload.get("document_id")
                or metadata.get("document_id")
                or document_id
            )

            chunk_id = (
                payload.get("chunk_id")
                or metadata.get("chunk_id")
            )

            retrieved_documents.append(
                {
                    "score": float(result.score),
                    "text": payload.get(
                        "text",
                        "",
                    ),
                    "metadata": metadata,
                    "document_id": result_document_id,
                    "chunk_id": chunk_id,
                    "document_name": (
                        payload.get(
                            "document_name"
                        )
                        or metadata.get(
                            "document_name"
                        )
                    ),
                    "page": (
                        payload.get("page")
                        or metadata.get("page")
                    ),
                }
            )

        return retrieved_documents

    # -------------------------------------------------------------------
    # Collection information
    # -------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the number of vectors in the collection.
        """
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return int(result.count)

    # -------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------

    def close(self) -> None:
        """
        Close the Qdrant client cleanly.
        """
        self.client.close()