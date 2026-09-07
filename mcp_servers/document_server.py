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
from rag.retrieval.query_expansion import QueryExpander


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

DEFAULT_SEMANTIC_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_RRF_K = 60

DEFAULT_RERANK_CANDIDATES = 10

# Multi-query retrieval
DEFAULT_MAX_QUERIES = 3

# Original user query receives the highest priority.
#
# Query 1 = original query
# Query 2 = keyword-oriented expansion
# Query 3 = semantic/domain expansion
#
# These weights are intentionally conservative so that a weak
# expansion cannot overpower the user's original intent.
DEFAULT_MULTI_QUERY_WEIGHTS = [1.0, 0.5, 0.5]

MAX_TOP_K = 10


# ============================================================
# DOCUMENT KNOWLEDGE BASE
# ============================================================

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
        Multi-query weighted RRF
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
        max_queries: int = DEFAULT_MAX_QUERIES,
    ) -> None:

        self.file_path = Path(file_path).resolve()

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise ValueError(
                f"Path is not a file: {self.file_path}"
            )

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if semantic_weight < 0:
            raise ValueError(
                "semantic_weight cannot be negative."
            )

        if keyword_weight < 0:
            raise ValueError(
                "keyword_weight cannot be negative."
            )

        if semantic_weight == 0 and keyword_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        if rerank_candidates <= 0:
            raise ValueError(
                "rerank_candidates must be greater than zero."
            )

        if max_queries <= 0:
            raise ValueError(
                "max_queries must be greater than zero."
            )

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        self.rerank_candidates = rerank_candidates

        self.max_queries = min(
            max_queries,
            len(DEFAULT_MULTI_QUERY_WEIGHTS),
        )

        self.multi_query_weights = (
            DEFAULT_MULTI_QUERY_WEIGHTS[:self.max_queries]
        )

        self.query_expander = QueryExpander(
            max_queries=self.max_queries
        )

        # ---------------------------------------------------------
        # Load document
        # ---------------------------------------------------------

        documents = load_pdf(
            str(self.file_path)
        )

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
            self.embeddings = self.embedding_model.encode(
                texts
            )
        else:
            self.embeddings = []

        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------

        self.bm25 = BM25Retriever(
            self.chunks
        )

        # ---------------------------------------------------------
        # Cross encoder
        # ---------------------------------------------------------

        self.reranker = CrossEncoderReranker()

    # =============================================================
    # Utility helpers
    # =============================================================

    @staticmethod
    def _result_id(
        chunk: dict[str, Any],
        index: int,
    ) -> str:

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
    def _metadata(
        result: dict[str, Any],
    ) -> dict[str, Any]:

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

        query_embedding = (
            self.embedding_model.encode_single(
                query
            )
        )

        scored: list[dict[str, Any]] = []

        for index, embedding in enumerate(
            self.embeddings
        ):

            score = float(
                sum(
                    a * b
                    for a, b in zip(
                        query_embedding,
                        embedding,
                    )
                )
            )

            chunk = self.chunks[index]

            scored.append(
                {
                    "result_id": self._result_id(
                        chunk,
                        index,
                    ),
                    "score": score,
                    "text": chunk["text"],
                    "metadata": chunk.get(
                        "metadata",
                        {},
                    ),
                }
            )

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        results = scored[:top_k]

        for rank, result in enumerate(
            results,
            start=1,
        ):
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

            merged[result_id][
                "semantic_score"
            ] = result["score"]

            merged[result_id][
                "semantic_rank"
            ] = rank

            if (
                "semantic"
                not in merged[result_id][
                    "retrieval_sources"
                ]
            ):
                merged[result_id][
                    "retrieval_sources"
                ].append("semantic")

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

            merged[result_id][
                "keyword_score"
            ] = result.get(
                "bm25_score",
                result.get("score"),
            )

            merged[result_id][
                "keyword_rank"
            ] = rank

            if (
                "bm25"
                not in merged[result_id][
                    "retrieval_sources"
                ]
            ):
                merged[result_id][
                    "retrieval_sources"
                ].append("bm25")

        # ---------------------------------------------------------
        # Weighted Hybrid RRF
        # ---------------------------------------------------------

        for result in merged.values():

            semantic_rank = result[
                "semantic_rank"
            ]

            keyword_rank = result[
                "keyword_rank"
            ]

            semantic_rrf = (
                self.semantic_weight
                / (
                    self.rrf_k
                    + semantic_rank
                )
                if semantic_rank is not None
                else 0.0
            )

            keyword_rrf = (
                self.keyword_weight
                / (
                    self.rrf_k
                    + keyword_rank
                )
                if keyword_rank is not None
                else 0.0
            )

            result[
                "semantic_rrf_score"
            ] = semantic_rrf

            result[
                "keyword_rrf_score"
            ] = keyword_rrf

            result[
                "hybrid_score"
            ] = (
                semantic_rrf
                + keyword_rrf
            )

            result[
                "retrieval_source_count"
            ] = len(
                result[
                    "retrieval_sources"
                ]
            )

        results = list(
            merged.values()
        )

        results.sort(
            key=lambda item: (
                item["hybrid_score"],
                item[
                    "retrieval_source_count"
                ],
                -(
                    item["semantic_rank"]
                    if item[
                        "semantic_rank"
                    ] is not None
                    else 10**9
                ),
            ),
            reverse=True,
        )

        return results[:candidate_k]

    # =============================================================
    # Weighted Multi-Query Retrieval
    # =============================================================

    def multi_query_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        """
        Perform multi-query retrieval using weighted RRF.

        Query priorities:

            Original query       -> 1.0
            Keyword expansion    -> 0.5
            Semantic expansion   -> 0.5

        The original query therefore remains the strongest
        retrieval signal.

        Each expanded query performs its own hybrid retrieval.
        The resulting candidate sets are then fused using
        weighted Reciprocal Rank Fusion.
        """

        if not query.strip():
            return []

        if not self.chunks:
            return []

        if top_k <= 0:
            return []

        expanded_queries = (
            self.query_expander.expand(
                query
            )
        )

        if not expanded_queries:
            return []

        candidate_k = min(
            len(self.chunks),
            max(top_k * 5, 20),
        )

        fused: dict[str, dict[str, Any]] = {}

        # ---------------------------------------------------------
        # Retrieve for every query variation
        # ---------------------------------------------------------

        for query_index, expanded_query in enumerate(
            expanded_queries
        ):

            hybrid_results = self.hybrid_search(
                expanded_query,
                top_k=candidate_k,
            )

            # Use configured weight.
            #
            # If more queries are somehow generated than
            # configured, use the last configured weight.
            if query_index < len(
                self.multi_query_weights
            ):
                weight = (
                    self.multi_query_weights[
                        query_index
                    ]
                )
            else:
                weight = (
                    self.multi_query_weights[-1]
                )

            # -----------------------------------------------------
            # Weighted RRF
            # -----------------------------------------------------

            for rank, result in enumerate(
                hybrid_results,
                start=1,
            ):

                result_id = result.get(
                    "result_id"
                )

                if not result_id:
                    continue

                contribution = (
                    weight
                    / (
                        self.rrf_k
                        + rank
                    )
                )

                if result_id not in fused:
                    fused[result_id] = {
                        "result": dict(
                            result
                        ),
                        "multi_query_score": 0.0,
                        "multi_query_matches": 0,
                        "multi_query_sources": [],
                        "multi_query_ranks": [],
                    }

                fused[result_id][
                    "multi_query_score"
                ] += contribution

                fused[result_id][
                    "multi_query_matches"
                ] += 1

                fused[result_id][
                    "multi_query_sources"
                ].append(
                    query_index + 1
                )

                fused[result_id][
                    "multi_query_ranks"
                ].append(rank)

        # ---------------------------------------------------------
        # Build fused results
        # ---------------------------------------------------------

        ranked: list[dict[str, Any]] = []

        for item in fused.values():

            result = dict(
                item["result"]
            )

            result[
                "multi_query_score"
            ] = item[
                "multi_query_score"
            ]

            result[
                "multi_query_matches"
            ] = item[
                "multi_query_matches"
            ]

            result[
                "multi_query_sources"
            ] = item[
                "multi_query_sources"
            ]

            result[
                "multi_query_ranks"
            ] = item[
                "multi_query_ranks"
            ]

            ranked.append(result)

        # ---------------------------------------------------------
        # Final multi-query ranking
        # ---------------------------------------------------------

        ranked.sort(
            key=lambda result: (
                result.get(
                    "multi_query_score",
                    0.0,
                ),
                result.get(
                    "multi_query_matches",
                    0,
                ),
                result.get(
                    "hybrid_score",
                    0.0,
                ),
            ),
            reverse=True,
        )

        final_results = ranked[:top_k]

        # ---------------------------------------------------------
        # Assign multi-query rank
        # ---------------------------------------------------------

        for rank, result in enumerate(
            final_results,
            start=1,
        ):
            result[
                "multi_query_rank"
            ] = rank

        return final_results

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

        # ---------------------------------------------------------
        # Multi-query hybrid retrieval
        # ---------------------------------------------------------

        candidates = self.multi_query_search(
            query,
            top_k=top_k,
        )

        if not candidates:
            return []

        # ---------------------------------------------------------
        # Cross-encoder reranking
        #
        # Reranking is intentionally performed against the
        # ORIGINAL user query rather than an expanded query.
        # This keeps final ranking aligned with user intent.
        # ---------------------------------------------------------

        rerank_k = min(
            len(candidates),
            max(
                top_k,
                self.rerank_candidates,
            ),
        )

        rerank_candidates = candidates[
            :rerank_k
        ]

        reranked = self.reranker.rerank(
            query,
            rerank_candidates,
        )

        # ---------------------------------------------------------
        # Build final result objects
        # ---------------------------------------------------------

        results: list[
            dict[str, Any]
        ] = []

        for rank, result in enumerate(
            reranked[:top_k],
            start=1,
        ):

            metadata = dict(
                result.get(
                    "metadata",
                    {},
                )
            )

            enriched = {
                "text": result[
                    "text"
                ],

                "score": result.get(
                    "score",
                    result.get(
                        "rerank_score"
                    ),
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

                # -------------------------------------------------
                # Multi-query metadata
                # -------------------------------------------------

                "multi_query_score": result.get(
                    "multi_query_score"
                ),

                "multi_query_rank": result.get(
                    "multi_query_rank"
                ),

                "multi_query_matches": result.get(
                    "multi_query_matches"
                ),

                "multi_query_sources": result.get(
                    "multi_query_sources",
                    [],
                ),

                "multi_query_ranks": result.get(
                    "multi_query_ranks",
                    [],
                ),

                # -------------------------------------------------
                # Identity
                # -------------------------------------------------

                "result_id": result.get(
                    "result_id"
                ),

                "chunk_id": metadata.get(
                    "chunk_id"
                ),

                "document_id": metadata.get(
                    "document_id"
                ),

                # -------------------------------------------------
                # Source metadata
                # -------------------------------------------------

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

            results.append(
                enriched
            )

        # ---------------------------------------------------------
        # Build structured citations
        # ---------------------------------------------------------

        for index, result in enumerate(
            results,
            start=1,
        ):

            result[
                "citation"
            ] = build_citation(
                result,
                index,
            )

        return results

    # =============================================================
    # Document information
    # =============================================================

    def get_document_info(
        self,
    ) -> dict[str, Any]:

        return {
            "file_path": str(
                self.file_path
            ),

            "document_name": (
                self.file_path.name
            ),

            "chunk_count": len(
                self.chunks
            ),

            "chunk_size": (
                self.chunk_size
            ),

            "chunk_overlap": (
                self.chunk_overlap
            ),

            "chunking_strategy": (
                "sentence_aware"
            ),

            # -----------------------------------------------------
            # Hybrid retrieval
            # -----------------------------------------------------

            "semantic_weight": (
                self.semantic_weight
            ),

            "keyword_weight": (
                self.keyword_weight
            ),

            "fusion": "rrf",

            "rrf_k": (
                self.rrf_k
            ),

            # -----------------------------------------------------
            # Multi-query retrieval
            # -----------------------------------------------------

            "multi_query": True,

            "max_queries": (
                self.max_queries
            ),

            "multi_query_weights": (
                self.multi_query_weights
            ),

            "query_expansion": (
                "deterministic"
            ),

            # -----------------------------------------------------
            # Reranking
            # -----------------------------------------------------

            "reranking": (
                "cross_encoder"
            ),

            "rerank_candidates": (
                self.rerank_candidates
            ),

            # -----------------------------------------------------
            # Citations
            # -----------------------------------------------------

            "citations": True,
        }


# ================================================================
# GLOBAL KNOWLEDGE BASE
# ================================================================

_KNOWLEDGE_BASE: (
    DocumentKnowledgeBase | None
) = None


def get_knowledge_base() -> DocumentKnowledgeBase:

    global _KNOWLEDGE_BASE

    if _KNOWLEDGE_BASE is None:

        default_document = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "raw"
            / "sample.pdf"
        )

        _KNOWLEDGE_BASE = (
            DocumentKnowledgeBase(
                default_document
            )
        )

    return _KNOWLEDGE_BASE


# ================================================================
# MCP SERVER
# ================================================================

server = MCPServer(
    name="document-server"
)


# ================================================================
# SEARCH DOCUMENTS TOOL
# ================================================================

@server.tool()
def search_documents(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:

    """
    Search the document knowledge base.

    Retrieval pipeline:

        Query
          ↓
        Query Expansion
          ↓
        Hybrid Semantic + BM25
          ↓
        Weighted Multi-Query RRF
          ↓
        Cross-Encoder Reranking
          ↓
        Structured Citations

    Returns document evidence with source attribution,
    retrieval scores, multi-query metadata, and citations.
    """

    if not isinstance(query, str):
        raise TypeError(
            "query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "query cannot be empty."
        )

    if not isinstance(top_k, int):
        raise TypeError(
            "top_k must be an integer."
        )

    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(
            f"top_k must be between "
            f"1 and {MAX_TOP_K}."
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

            "fusion": "weighted_multi_query_rrf",

            "rrf_k": (
                kb.rrf_k
            ),

            "multi_query": True,

            "max_queries": (
                kb.max_queries
            ),

            "multi_query_weights": (
                kb.multi_query_weights
            ),

            "query_expansion": (
                "deterministic"
            ),

            "reranking": (
                "cross_encoder"
            ),

            "citations": True,
        },
    }


# ================================================================
# DOCUMENT INFO TOOL
# ================================================================

@server.tool()
def get_document_info() -> dict[str, Any]:

    """
    Return metadata about the loaded document
    and retrieval configuration.
    """

    return get_knowledge_base().get_document_info()


# ================================================================
# SERVER ENTRY POINT
# ================================================================

if __name__ == "__main__":
    server.run()