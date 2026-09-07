"""
Unified document ingestion pipeline for OmniMind.

Pipeline:

PDF
 ↓
DocumentManager
 ↓
Loader
 ↓
Sentence-aware Chunker
 ↓
EmbeddingModel
 ↓
Persistent Qdrant
 ↓
Registry status = indexed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.embeddings.embedder import EmbeddingModel
from rag.ingestion.chunker import chunk_documents
from rag.ingestion.document_manager import DocumentManager
from rag.ingestion.loader import load_pdf
from rag.retrieval.vector_store import QdrantVectorStore


class DocumentIngestionPipeline:
    """
    End-to-end document ingestion pipeline.
    """

    def __init__(
        self,
        document_manager: DocumentManager | None = None,
        vector_store: QdrantVectorStore | None = None,
        embedding_model: EmbeddingModel | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap must be non-negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.document_manager = (
            document_manager
            if document_manager is not None
            else DocumentManager()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else QdrantVectorStore()
        )

        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else EmbeddingModel()
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Single-document ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        file_path: str | Path,
        *,
        allow_duplicate: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest one PDF end-to-end.

        Returns a structured ingestion result.
        """

        path = (
            Path(file_path)
            .expanduser()
            .resolve()
        )

        document = self.document_manager.add_document(
            path,
            allow_duplicate=allow_duplicate,
        )

        document_id = document[
            "document_id"
        ]

        try:
            # ----------------------------------------------------------
            # Mark processing
            # ----------------------------------------------------------

            self.document_manager.update_status(
                document_id,
                "processing",
            )

            # ----------------------------------------------------------
            # Load PDF
            # ----------------------------------------------------------

            pages = load_pdf(path)

            if not pages:
                raise ValueError(
                    f"PDF contains no readable text pages: {path}"
                )

            # ----------------------------------------------------------
            # Chunk
            # ----------------------------------------------------------

            chunks = chunk_documents(
                pages,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            if not chunks:
                raise ValueError(
                    f"No chunks were generated for document: {path}"
                )

            # ----------------------------------------------------------
            # Embeddings
            # ----------------------------------------------------------

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = (
                self.embedding_model.encode(
                    texts
                )
            )

            if len(embeddings) != len(chunks):
                raise ValueError(
                    "Embedding count does not match chunk count."
                )

            # ----------------------------------------------------------
            # Qdrant
            # ----------------------------------------------------------

            point_ids = (
                self.vector_store.add_documents(
                    chunks,
                    embeddings,
                )
            )

            if point_ids is None:
                raise ValueError(
                    "Vector store did not return point IDs."
                )

            if len(point_ids) != len(chunks):
                raise ValueError(
                    "Qdrant point count does not match chunk count."
                )

            # ----------------------------------------------------------
            # Verify actual indexed count
            # ----------------------------------------------------------

            indexed_count = (
                self.vector_store.count(
                    document_id=document_id
                )
            )

            if indexed_count != len(chunks):
                raise ValueError(
                    "Qdrant indexed vector count does not "
                    "match generated chunk count."
                )

            # ----------------------------------------------------------
            # Mark indexed
            # ----------------------------------------------------------

            updated_document = (
                self.document_manager.update_status(
                    document_id,
                    "indexed",
                )
            )

            # ----------------------------------------------------------
            # Result
            # ----------------------------------------------------------

            return {
                "success": True,
                "document_id": document_id,
                "document_name": document.get(
                    "document_name"
                ),
                "file_path": str(path),
                "status": updated_document[
                    "status"
                ],
                "page_count": document.get(
                    "page_count",
                    len(pages),
                ),
                "text_page_count": document.get(
                    "text_page_count",
                    len(pages),
                ),
                "chunk_count": len(chunks),
                "vector_count": len(point_ids),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

        except Exception:
            # ----------------------------------------------------------
            # Failure handling
            # ----------------------------------------------------------

            try:
                self.document_manager.update_status(
                    document_id,
                    "failed",
                )
            except Exception:
                pass

            raise

    # ------------------------------------------------------------------
    # Batch ingestion
    # ------------------------------------------------------------------

    def ingest_documents(
        self,
        file_paths: list[str | Path],
        *,
        allow_duplicate: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ingest multiple documents sequentially.

        A failure in one document does not stop the remaining
        documents.
        """

        results: list[dict[str, Any]] = []

        for file_path in file_paths:

            try:
                result = self.ingest_document(
                    file_path,
                    allow_duplicate=allow_duplicate,
                )

                results.append(result)

            except Exception as exc:

                path = (
                    Path(file_path)
                    .expanduser()
                    .resolve()
                )

                document = (
                    self.document_manager
                    .get_document_by_path(path)
                )

                results.append(
                    {
                        "success": False,
                        "document_id": (
                            document.get(
                                "document_id"
                            )
                            if document
                            else None
                        ),
                        "document_name": path.name,
                        "file_path": str(path),
                        "status": (
                            document.get(
                                "status"
                            )
                            if document
                            else "failed"
                        ),
                        "error": str(exc),
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_document_status(
        self,
        document_id: str,
    ) -> dict[str, Any] | None:
        """Return registry information for a document."""

        return self.document_manager.get_document(
            document_id
        )

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove_document(
        self,
        document_id: str,
        *,
        delete_vectors: bool = True,
    ) -> bool:
        """
        Remove a document from OmniMind.

        The original PDF is never deleted by this method.
        """

        document = (
            self.document_manager.get_document(
                document_id
            )
        )

        if document is None:
            return False

        if delete_vectors:
            self.vector_store.delete_document(
                document_id
            )

        return self.document_manager.remove_document(
            document_id
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying vector store."""

        if self.vector_store is not None:
            self.vector_store.close()