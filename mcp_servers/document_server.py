from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from mcp.server import MCPServer

from rag.embeddings.embedder import EmbeddingModel
from rag.ingestion.document_manager import DocumentManager
from rag.retrieval.vector_store import QdrantVectorStore
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.citations import build_citation


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5
MAX_TOP_K = 10


# ============================================================
# PERSISTENT DOCUMENT KNOWLEDGE BASE
# ============================================================


class PersistentDocumentKnowledgeBase:
    """
    Persistent multi-document knowledge base.

    Retrieval architecture:

        User Query
             |
             v
        EmbeddingModel
             |
             v
        Persistent Qdrant
             |
             v
        Document-aware semantic retrieval
             |
             v
        Cross-Encoder reranking
             |
             v
        Citation-aware results

    The document registry is used as the source of truth for
    registered/indexed documents.

    Qdrant stores the actual searchable document chunks.
    """

    def __init__(
        self,
        *,
        registry_path: str | Path | None = None,
        collection_name: str = "omnimind_documents",
        vector_size: int = 384,
        storage_path: str | Path = "data/vector_store/qdrant",
        rerank_candidates: int = 10,
    ) -> None:

        self.document_manager = DocumentManager(
            registry_path=registry_path,
        )

        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            vector_size=vector_size,
            storage_path=str(storage_path),
        )

        self.embedding_model = EmbeddingModel()

        self.reranker = CrossEncoderReranker()

        self.rerank_candidates = max(
            1,
            int(rerank_candidates),
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        return query

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(
                f"top_k must be between "
                f"1 and {MAX_TOP_K}."
            )

        return top_k

    def _validate_document_id(
        self,
        document_id: str | None,
    ) -> str | None:
        """
        Validate an optional document ID.

        A document ID is accepted only when the document is
        registered and indexed.
        """

        if document_id is None:
            return None

        document_id = str(
            document_id
        ).strip()

        if not document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        document = self.document_manager.get_document(
            document_id
        )

        if document is None:
            raise ValueError(
                f"Document not found: {document_id}"
            )

        if document.get("status") != "indexed":
            raise ValueError(
                f"Document is not indexed: {document_id}"
            )

        if not self.vector_store.document_exists(
            document_id
        ):
            raise ValueError(
                f"Document has no indexed vectors: "
                f"{document_id}"
            )

        return document_id

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:

        query = self._validate_query(query)

        top_k = self._validate_top_k(top_k)

        document_id = self._validate_document_id(
            document_id
        )

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding = (
            self.embedding_model.encode_single(
                query
            )
        )

        # ----------------------------------------------------
        # Retrieve a larger candidate pool before reranking.
        # ----------------------------------------------------

        candidate_k = min(
            MAX_TOP_K * 2,
            max(
                top_k * 2,
                self.rerank_candidates,
            ),
        )

        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            document_id=document_id,
        )

        if not candidates:
            return []

        # ----------------------------------------------------
        # Cross-encoder reranking
        #
        # Always rerank using the ORIGINAL user query.
        # ----------------------------------------------------

        rerank_candidates = candidates[
            : self.rerank_candidates
        ]

        reranked = self.reranker.rerank(
            query,
            rerank_candidates,
            top_k=top_k,
        )

        # ----------------------------------------------------
        # Normalize final result objects.
        # ----------------------------------------------------

        results: list[dict[str, Any]] = []

        for rank, result in enumerate(
            reranked[:top_k],
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            result_document_id = result.get(
                "document_id"
            )

            if result_document_id is None:
                result_document_id = metadata.get(
                    "document_id"
                )

            result_chunk_id = result.get(
                "chunk_id"
            )

            if result_chunk_id is None:
                result_chunk_id = metadata.get(
                    "chunk_id"
                )

            enriched = {
                "result_id": result.get(
                    "result_id"
                ),

                "rank": rank,

                "text": result.get(
                    "text",
                    "",
                ),

                "score": result.get(
                    "score"
                ),

                "rerank_score": result.get(
                    "rerank_score"
                ),

                "document_id": (
                    result_document_id
                ),

                "chunk_id": (
                    result_chunk_id
                ),

                "source": metadata.get(
                    "source"
                ),

                "document_name": metadata.get(
                    "document_name"
                ),

                "page": metadata.get(
                    "page"
                ),

                "page_number": metadata.get(
                    "page_number"
                ),

                "chunk_index": metadata.get(
                    "chunk_index"
                ),

                "metadata": metadata,

                "retrieval_source": (
                    "persistent_qdrant"
                ),

                "document_filter": (
                    document_id
                ),
            }

            results.append(enriched)

        # ----------------------------------------------------
        # Structured citations
        # ----------------------------------------------------

        for index, result in enumerate(
            results,
            start=1,
        ):
            result["citation"] = build_citation(
                result,
                index,
            )

        return results

    # ========================================================
    # DOCUMENT INFORMATION
    # ========================================================

    def get_document_info(
        self,
        document_id: str | None = None,
    ) -> dict[str, Any]:

        if document_id is not None:
            document_id = str(
                document_id
            ).strip()

            document = (
                self.document_manager.get_document(
                    document_id
                )
            )

            if document is None:
                raise ValueError(
                    f"Document not found: "
                    f"{document_id}"
                )

            vector_count = (
                self.vector_store.count(
                    document_id=document_id
                )
            )

            return {
                "document": document,
                "vector_count": vector_count,
                "indexed": vector_count > 0,
            }

        registry_info = (
            self.document_manager.get_registry_info()
        )

        collection_info = (
            self.vector_store.collection_info()
        )

        return {
            "registry": registry_info,
            "vector_store": collection_info,
        }

    # ========================================================
    # LIST DOCUMENTS
    # ========================================================

    def list_documents(
        self,
        status: str | None = None,
    ) -> list[dict[str, Any]]:

        documents = (
            self.document_manager.list_documents(
                status=status
            )
        )

        enriched_documents = []

        for document in documents:

            document_id = document.get(
                "document_id"
            )

            vector_count = (
                self.vector_store.count(
                    document_id=document_id
                )
                if document_id
                else 0
            )

            enriched = dict(document)

            enriched[
                "vector_count"
            ] = vector_count

            enriched[
                "indexed"
            ] = vector_count > 0

            enriched_documents.append(
                enriched
            )

        return enriched_documents

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        self.vector_store.close()


# ============================================================
# GLOBAL KNOWLEDGE BASE
# ============================================================

_KNOWLEDGE_BASE: (
    PersistentDocumentKnowledgeBase | None
) = None


def get_knowledge_base(
) -> PersistentDocumentKnowledgeBase:

    global _KNOWLEDGE_BASE

    if _KNOWLEDGE_BASE is None:

        _KNOWLEDGE_BASE = (
            PersistentDocumentKnowledgeBase()
        )

    return _KNOWLEDGE_BASE


# ============================================================
# MCP SERVER
# ============================================================

server = MCPServer(
    name="document-server"
)


# ============================================================
# SEARCH DOCUMENTS TOOL
# ============================================================


@server.tool()
def search_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    document_id: str | None = None,
) -> dict[str, Any]:

    """
    Search persistent OmniMind document memory.

    Supports:

        Global search:
            search_documents(
                query="machine learning"
            )

        Document-specific search:
            search_documents(
                query="machine learning",
                document_id="doc-..."
            )

    Retrieval:

        Query
          ↓
        Embedding
          ↓
        Persistent Qdrant
          ↓
        Optional document filter
          ↓
        Cross-encoder reranking
          ↓
        Structured citations
    """

    kb = get_knowledge_base()

    results = kb.search(
        query=query,
        top_k=top_k,
        document_id=document_id,
    )

    return {
        "query": query.strip(),

        "document_id": document_id,

        "results": results,

        "count": len(results),

        "retrieval": {
            "semantic": True,
            "bm25": False,
            "fusion": "persistent_qdrant",
            "persistent": True,
            "document_filter": (
                document_id is not None
            ),
            "reranking": "cross_encoder",
            "citations": True,
        },
    }


# ============================================================
# DOCUMENT INFORMATION TOOL
# ============================================================


@server.tool()
def get_document_info(
    document_id: str | None = None,
) -> dict[str, Any]:

    """
    Return information about all indexed documents
    or one specific document.
    """

    return get_knowledge_base().get_document_info(
        document_id=document_id
    )


# ============================================================
# LIST DOCUMENTS TOOL
# ============================================================


@server.tool()
def list_documents(
    status: str | None = None,
) -> dict[str, Any]:

    """
    List registered documents.

    Optional status values:

        registered
        processing
        indexed
        failed
        removed
    """

    documents = (
        get_knowledge_base().list_documents(
            status=status
        )
    )

    return {
        "documents": documents,
        "count": len(documents),
    }


# ============================================================
# SERVER ENTRY POINT
# ============================================================


if __name__ == "__main__":
    server.run()