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
from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.citations import build_citation


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

DEFAULT_SEMANTIC_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_RRF_K = 60

DEFAULT_RERANK_CANDIDATES = 10
MAX_TOP_K = 10


class DocumentKnowledgeBase:
    """
    In-memory document knowledge base used by the MCP document server.

    Retrieval pipeline:

        Document
          ↓
        Sentence-aware chunks
          ↓
        Embeddings + BM25
          ↓
        Hybrid RRF
          ↓
        Cross-encoder reranking
          ↓
        Citation-aware results
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> None:
        self.file_path = Path(file_path).resolve()

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise ValueError(
                f"Path is not a file: {self.file_path}"
            )

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if semantic_weight < 0:
            raise ValueError("semantic_weight cannot be negative.")

        if keyword_weight < 0:
            raise ValueError("keyword_weight cannot be negative.")

        if semantic_weight == 0 and keyword_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if rerank_candidates <= 0:
            raise ValueError(
                "rerank_candidates must be greater than zero."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        self.rerank_candidates = rerank_candidates

        # ---------------------------------------------------------
        # Load document
        # ---------------------------------------------------------
        documents = load_pdf(str(self.file_path))

        self.chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # ---------------------------------------------------------
        # Embedding model
        # ---------------------------------------------------------
        self.embedding_model = EmbeddingModel()

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        if texts:
            self.embeddings = self.embedding_model.encode(texts)
        else:
            self.embeddings = []

        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------
        self.bm25 = BM25Retriever(self.chunks)

        # ---------------------------------------------------------
        # Cross encoder
        # ---------------------------------------------------------
        self.reranker = CrossEncoderReranker()

    # =============================================================
    # Utility helpers
    # =============================================================

    @staticmethod
    def _result_id(chunk: dict[str, Any], index: int) -> str:
        metadata = chunk.get("metadata", {})

        if isinstance(metadata, dict):
            chunk_id = metadata.get("chunk_id")

            if chunk_id:
                return str(chunk_id)

        chunk_id = chunk.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

        return f"chunk-{index}"

    @staticmethod
    def _metadata(result: dict[str, Any]) -> dict[str, Any]:
        metadata = result.get("metadata")

        if isinstance(metadata, dict):
            return metadata

        return {}

    # =============================================================
    # Semantic retrieval
    # =============================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        if not self.chunks:
            return []

        query_embedding = self.embedding_model.encode_single(query)

        scored: list[dict[str, Any]] = []

        for index, embedding in enumerate(self.embeddings):
            score = float(
                sum(
                    a * b
                    for a, b in zip(query_embedding, embedding)
                )
            )

            chunk = self.chunks[index]

            scored.append(
                {
                    "result_id": self._result_id(chunk, index),
                    "score": score,
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                }
            )

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        results = scored[:top_k]

        for rank, result in enumerate(results, start=1):
            result["semantic_rank"] = rank

        return results

    # =============================================================
    # Hybrid RRF retrieval
    # =============================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        if not self.chunks:
            return []

        candidate_k = min(
            len(self.chunks),
            max(top_k * 5, 20),
        )

        semantic_results = self.semantic_search(
            query,
            top_k=candidate_k,
        )

        keyword_results = self.bm25.search(
            query,
            top_k=candidate_k,
        )

        merged: dict[str, dict[str, Any]] = {}

        # ---------------------------------------------------------
        # Semantic candidates
        # ---------------------------------------------------------
        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            result_id = result["result_id"]

            if result_id not in merged:
                merged[result_id] = {
                    "result_id": result_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_score": None,
                    "keyword_score": None,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "retrieval_sources": [],
                }

            merged[result_id]["semantic_score"] = result["score"]
            merged[result_id]["semantic_rank"] = rank

            if "semantic" not in merged[result_id]["retrieval_sources"]:
                merged[result_id]["retrieval_sources"].append(
                    "semantic"
                )

        # ---------------------------------------------------------
        # BM25 candidates
        # ---------------------------------------------------------
        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):
            result_id = result["result_id"]

            if result_id not in merged:
                merged[result_id] = {
                    "result_id": result_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_score": None,
                    "keyword_score": None,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "retrieval_sources": [],
                }

            merged[result_id]["keyword_score"] = result.get(
                "bm25_score",
                result.get("score"),
            )

            merged[result_id]["keyword_rank"] = rank

            if "bm25" not in merged[result_id]["retrieval_sources"]:
                merged[result_id]["retrieval_sources"].append(
                    "bm25"
                )

        # ---------------------------------------------------------
        # Reciprocal Rank Fusion
        # ---------------------------------------------------------
        for result in merged.values():
            semantic_rank = result["semantic_rank"]
            keyword_rank = result["keyword_rank"]

            semantic_rrf = (
                self.semantic_weight
                / (self.rrf_k + semantic_rank)
                if semantic_rank is not None
                else 0.0
            )

            keyword_rrf = (
                self.keyword_weight
                / (self.rrf_k + keyword_rank)
                if keyword_rank is not None
                else 0.0
            )

            result["semantic_rrf_score"] = semantic_rrf
            result["keyword_rrf_score"] = keyword_rrf
            result["hybrid_score"] = (
                semantic_rrf + keyword_rrf
            )
            result["retrieval_source_count"] = len(
                result["retrieval_sources"]
            )

        results = list(merged.values())

        results.sort(
            key=lambda item: (
                item["hybrid_score"],
                item["retrieval_source_count"],
                -(
                    item["semantic_rank"]
                    if item["semantic_rank"] is not None
                    else 10**9
                ),
            ),
            reverse=True,
        )

        return results[:candidate_k]

    # =============================================================
    # Citation-aware final retrieval
    # =============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        top_k = min(
            max(1, top_k),
            MAX_TOP_K,
        )

        candidates = self.hybrid_search(
            query,
            top_k=top_k,
        )

        if not candidates:
            return []

        rerank_k = min(
            len(candidates),
            max(top_k, self.rerank_candidates),
        )

        rerank_candidates = candidates[:rerank_k]

        reranked = self.reranker.rerank(
            query,
            rerank_candidates,
        )

        results: list[dict[str, Any]] = []

        for rank, result in enumerate(
            reranked[:top_k],
            start=1,
        ):
            metadata = dict(
                result.get("metadata", {})
            )

            enriched = {
                "text": result["text"],
                "score": result.get(
                    "score",
                    result.get("rerank_score"),
                ),
                "rerank_score": result.get(
                    "rerank_score"
                ),
                "hybrid_score": result.get(
                    "hybrid_score"
                ),
                "semantic_score": result.get(
                    "semantic_score"
                ),
                "keyword_score": result.get(
                    "keyword_score"
                ),
                "semantic_rank": result.get(
                    "semantic_rank"
                ),
                "keyword_rank": result.get(
                    "keyword_rank"
                ),
                "retrieval_sources": result.get(
                    "retrieval_sources",
                    [],
                ),
                "retrieval_source_count": result.get(
                    "retrieval_source_count",
                    0,
                ),
                "result_id": result.get(
                    "result_id"
                ),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "document_id": metadata.get(
                    "document_id"
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
                "chunk": metadata.get(
                    "chunk"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "metadata": metadata,
            }

            results.append(enriched)

        # ---------------------------------------------------------
        # Build structured citations
        # ---------------------------------------------------------
        for index, result in enumerate(
            results,
            start=1,
        ):
            result["citation"] = build_citation(
                result,
                index,
            )

        return results

    # =============================================================
    # Document information
    # =============================================================

    def get_document_info(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "document_name": self.file_path.name,
            "chunk_count": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunking_strategy": "sentence_aware",
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "fusion": "rrf",
            "rrf_k": self.rrf_k,
            "reranking": "cross_encoder",
            "rerank_candidates": self.rerank_candidates,
            "citations": True,
        }


# ================================================================
# Global knowledge base
# ================================================================

_KNOWLEDGE_BASE: DocumentKnowledgeBase | None = None


def get_knowledge_base() -> DocumentKnowledgeBase:
    global _KNOWLEDGE_BASE

    if _KNOWLEDGE_BASE is None:
        default_document = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "sample.pdf"
        )


        _KNOWLEDGE_BASE = DocumentKnowledgeBase(
            default_document
        )

    return _KNOWLEDGE_BASE


# ================================================================
# MCP server
# ================================================================

server = MCPServer(
    name="document-server"
)


@server.tool()
def search_documents(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search the document knowledge base.

    Returns hybrid RRF + cross-encoder reranked
    results with source attribution and citations.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty.")

    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")

    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(
            f"top_k must be between 1 and {MAX_TOP_K}."
        )

    kb = get_knowledge_base()

    results = kb.search(
        query,
        top_k=top_k,
    )

    return {
        "query": query,
        "results": results,
        "retrieval": {
            "semantic": True,
            "bm25": True,
            "fusion": "rrf",
            "rrf_k": DEFAULT_RRF_K,
            "reranking": "cross_encoder",
            "citations": True,
        },
    }


@server.tool()
def get_document_info() -> dict[str, Any]:
    """
    Return metadata about the loaded document
    and retrieval configuration.
    """

    return get_knowledge_base().get_document_info()


if __name__ == "__main__":
    server.run()
