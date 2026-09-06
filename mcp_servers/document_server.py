from __future__ import annotations

from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MCP
# ============================================================

from mcp.server import MCPServer


# ============================================================
# RAG COMPONENTS
# ============================================================

from rag.embeddings.embedder import EmbeddingModel
from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.reranker import CrossEncoderReranker


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

DEFAULT_SEMANTIC_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_RRF_K = 60

DEFAULT_RERANK_CANDIDATES = 10
MAX_TOP_K = 10


# ============================================================
# MCP SERVER
# ============================================================

mcp = MCPServer(
    "OmniMind Document Server"
)


# ============================================================
# DOCUMENT KNOWLEDGE BASE
# ============================================================

class DocumentKnowledgeBase:
    """
    In-memory document retrieval system for the
    OmniMind Document MCP Server.

    Retrieval pipeline:

        PDF
          ↓
        Chunking
          ↓
        Embeddings
          ↓
        Semantic Search
          +
        BM25 Keyword Search
          ↓
        Reciprocal Rank Fusion
          ↓
        Cross-Encoder Reranking
          ↓
        Retrieved Evidence

    The MCP server intentionally uses an in-memory
    semantic index rather than opening the application's
    persistent Qdrant database.
    """

    def __init__(
        self,
        pdf_path: Path,
    ):
        self.pdf_path = pdf_path

        # ----------------------------------------------------
        # Validate PDF
        # ----------------------------------------------------

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        if not self.pdf_path.is_file():
            raise ValueError(
                f"PDF path is not a file: {self.pdf_path}"
            )

        print(
            f"Loading document: {self.pdf_path}",
            flush=True,
        )

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        documents = load_pdf(
            str(self.pdf_path)
        )

        # ----------------------------------------------------
        # Chunk documents
        # ----------------------------------------------------

        self.chunks = chunk_documents(
            documents,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )

        print(
            f"Loaded {len(self.chunks)} document chunks.",
            flush=True,
        )

        # ----------------------------------------------------
        # Embedding model
        # ----------------------------------------------------

        print(
            "Loading embedding model: "
            "BAAI/bge-small-en-v1.5",
            flush=True,
        )

        self.embedder = EmbeddingModel()

        print(
            "Embedding model loaded successfully.",
            flush=True,
        )

        # ----------------------------------------------------
        # Create embeddings
        # ----------------------------------------------------

        print(
            "Creating in-memory document embeddings...",
            flush=True,
        )

        self.embeddings = self.embedder.encode(
            [
                chunk["text"]
                for chunk in self.chunks
            ]
        )

        print(
            "Document embeddings created.",
            flush=True,
        )

        # ----------------------------------------------------
        # BM25 keyword retriever
        # ----------------------------------------------------

        self.bm25 = BM25Retriever(
            self.chunks
        )

        # ----------------------------------------------------
        # Cross-encoder reranker
        # ----------------------------------------------------

        print(
            "Loading cross-encoder reranker...",
            flush=True,
        )

        self.reranker = CrossEncoderReranker()

        print(
            "Cross-encoder reranker loaded successfully.",
            flush=True,
        )

    # ========================================================
    # RESULT ID
    # ========================================================

    @staticmethod
    def _result_key(
        result: dict,
        fallback_index: int,
    ) -> str:
        """
        Build a stable identifier for a retrieved chunk.

        Prefer the deterministic chunk_id generated by the
        ingestion pipeline.
        """

        metadata = result.get(
            "metadata",
            {},
        )

        chunk_id = metadata.get(
            "chunk_id"
        )

        if chunk_id:
            return str(chunk_id)

        return (
            f"{metadata.get('source', 'unknown')}:"
            f"{metadata.get('page', 0)}:"
            f"{metadata.get('chunk_index', fallback_index)}"
        )

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search the in-memory document embeddings.

        Embeddings from the current model are normalized,
        therefore the dot product acts as cosine similarity.
        """

        if not query or not query.strip():
            return []

        if not self.chunks:
            return []

        top_k = max(
            1,
            min(
                top_k,
                len(self.chunks),
            ),
        )

        query_embedding = (
            self.embedder.encode_single(
                query
            )
        )

        scored = []

        # ----------------------------------------------------
        # Calculate cosine-style similarity
        # ----------------------------------------------------

        for index, embedding in enumerate(
            self.embeddings
        ):
            score = sum(
                a * b
                for a, b in zip(
                    query_embedding,
                    embedding,
                )
            )

            scored.append(
                (
                    float(score),
                    index,
                )
            )

        # ----------------------------------------------------
        # Highest score first
        # ----------------------------------------------------

        scored.sort(
            key=lambda item: (
                item[0],
                -item[1],
            ),
            reverse=True,
        )

        results = []

        for rank, (
            score,
            index,
        ) in enumerate(
            scored[:top_k],
            start=1,
        ):
            chunk = self.chunks[index]

            result = {
                "result_id": self._result_key(
                    chunk,
                    index,
                ),
                "score": score,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "semantic_rank": rank,
            }

            results.append(result)

        return results

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Combine semantic and BM25 retrieval using
        Reciprocal Rank Fusion (RRF).

        RRF avoids directly comparing raw semantic and
        BM25 scores because those scores use different
        scales.

        A chunk appearing in both retrieval systems receives
        evidence from both signals.
        """

        if not query or not query.strip():
            return []

        if not self.chunks:
            return []

        # ----------------------------------------------------
        # Candidate pool
        # ----------------------------------------------------

        candidate_k = max(
            top_k * 5,
            20,
        )

        candidate_k = min(
            candidate_k,
            len(self.chunks),
        )

        # ----------------------------------------------------
        # Semantic retrieval
        # ----------------------------------------------------

        semantic_results = (
            self.semantic_search(
                query=query,
                top_k=candidate_k,
            )
        )

        # ----------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------

        keyword_results = (
            self.bm25.search(
                query=query,
                top_k=candidate_k,
            )
        )

        combined: dict[str, dict] = {}

        # ----------------------------------------------------
        # Add semantic rankings
        # ----------------------------------------------------

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            key = self._result_key(
                result,
                rank,
            )

            entry = combined.setdefault(
                key,
                {
                    "result_id": key,
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "semantic_rrf_score": 0.0,
                    "keyword_rrf_score": 0.0,
                    "retrieval_sources": [],
                },
            )

            entry["semantic_score"] = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            entry["semantic_rank"] = rank

            entry["semantic_rrf_score"] = (
                DEFAULT_SEMANTIC_WEIGHT
                / (
                    DEFAULT_RRF_K
                    + rank
                )
            )

            if "semantic" not in entry[
                "retrieval_sources"
            ]:
                entry[
                    "retrieval_sources"
                ].append(
                    "semantic"
                )

        # ----------------------------------------------------
        # Add BM25 rankings
        # ----------------------------------------------------

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):
            key = self._result_key(
                result,
                rank,
            )

            entry = combined.setdefault(
                key,
                {
                    "result_id": key,
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "semantic_rrf_score": 0.0,
                    "keyword_rrf_score": 0.0,
                    "retrieval_sources": [],
                },
            )

            entry["keyword_score"] = float(
                result.get(
                    "normalized_score",
                    result.get(
                        "score",
                        0.0,
                    ),
                )
            )

            entry["keyword_rank"] = rank

            entry["keyword_rrf_score"] = (
                DEFAULT_KEYWORD_WEIGHT
                / (
                    DEFAULT_RRF_K
                    + rank
                )
            )

            if "bm25" not in entry[
                "retrieval_sources"
            ]:
                entry[
                    "retrieval_sources"
                ].append(
                    "bm25"
                )

        # ----------------------------------------------------
        # Calculate final RRF score
        # ----------------------------------------------------

        results = []

        for result in combined.values():

            result["hybrid_score"] = (
                result[
                    "semantic_rrf_score"
                ]
                + result[
                    "keyword_rrf_score"
                ]
            )

            result[
                "retrieval_source_count"
            ] = len(
                result[
                    "retrieval_sources"
                ]
            )

            results.append(result)

        # ----------------------------------------------------
        # Sort
        #
        # Priority:
        #   1. RRF score
        #   2. Results appearing in both systems
        #   3. Better semantic rank
        # ----------------------------------------------------

        results.sort(
            key=lambda item: (
                item["hybrid_score"],
                item[
                    "retrieval_source_count"
                ],
                -(
                    item["semantic_rank"]
                    or 10**9
                ),
            ),
            reverse=True,
        )

        return results[:top_k]

    # ========================================================
    # FULL RETRIEVAL PIPELINE
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Execute the complete document retrieval pipeline:

            Hybrid RRF retrieval
                    ↓
            Cross-encoder reranking
                    ↓
            Final evidence
        """

        if not query or not query.strip():
            return []

        if not self.chunks:
            return []

        top_k = max(
            1,
            min(
                top_k,
                MAX_TOP_K,
            ),
        )

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        candidates = (
            self.hybrid_search(
                query=query,
                top_k=max(
                    DEFAULT_RERANK_CANDIDATES,
                    top_k,
                ),
            )
        )

        if not candidates:
            return []

        # ----------------------------------------------------
        # Cross-encoder reranking
        # ----------------------------------------------------

        reranked = (
            self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=top_k,
            )
        )

        # ----------------------------------------------------
        # Normalize output
        # ----------------------------------------------------

        results = []

        for result in reranked:

            metadata = result.get(
                "metadata",
                {},
            )

            rerank_score = float(
                result.get(
                    "rerank_score",
                    0.0,
                )
            )

            results.append(
                {
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "score": rerank_score,
                    "rerank_score": rerank_score,

                    # Hybrid retrieval information
                    "hybrid_score": float(
                        result.get(
                            "hybrid_score",
                            0.0,
                        )
                    ),
                    "semantic_score": float(
                        result.get(
                            "semantic_score",
                            0.0,
                        )
                    ),
                    "keyword_score": float(
                        result.get(
                            "keyword_score",
                            0.0,
                        )
                    ),

                    # Retrieval ranks
                    "semantic_rank": result.get(
                        "semantic_rank"
                    ),
                    "keyword_rank": result.get(
                        "keyword_rank"
                    ),

                    # Retrieval sources
                    "retrieval_sources": result.get(
                        "retrieval_sources",
                        [],
                    ),
                    "retrieval_source_count": result.get(
                        "retrieval_source_count",
                        0,
                    ),

                    # Stable identifiers
                    "result_id": result.get(
                        "result_id"
                    ),
                    "chunk_id": metadata.get(
                        "chunk_id"
                    ),
                    "document_id": metadata.get(
                        "document_id"
                    ),

                    # Citation metadata
                    "source": metadata.get(
                        "source",
                        "Unknown",
                    ),
                    "document_name": metadata.get(
                        "document_name",
                        metadata.get(
                            "source",
                            "Unknown",
                        ),
                    ),
                    "page": metadata.get(
                        "page",
                        "Unknown",
                    ),
                    "page_number": metadata.get(
                        "page_number",
                        metadata.get(
                            "page",
                            "Unknown",
                        ),
                    ),
                    "chunk": metadata.get(
                        "chunk_index",
                        "Unknown",
                    ),
                    "chunk_index": metadata.get(
                        "chunk_index",
                        "Unknown",
                    ),
                }
            )

        return results

    # ========================================================
    # RESOURCE CLEANUP
    # ========================================================

    def close(self):
        """
        Release model resources where supported.
        """

        try:
            self.reranker.close()
        except AttributeError:
            pass

        try:
            self.embedder.close()
        except AttributeError:
            pass


# ============================================================
# LAZY KNOWLEDGE BASE
# ============================================================

knowledge_base = None


def get_knowledge_base() -> DocumentKnowledgeBase:
    """
    Lazily initialize the document knowledge base.

    Heavy ML models are loaded only when a document
    search tool actually needs them.
    """

    global knowledge_base

    if knowledge_base is None:

        knowledge_base = DocumentKnowledgeBase(
            pdf_path=PDF_PATH
        )

    return knowledge_base


# ============================================================
# MCP TOOL: DOCUMENT INFORMATION
# ============================================================

@mcp.tool()
def get_document_info() -> dict:
    """
    Return metadata about the indexed document.
    """

    kb = get_knowledge_base()

    try:
        documents = load_pdf(
            str(kb.pdf_path)
        )

        pages = len(documents)

    except Exception:
        pages = 0

    return {
        "document": kb.pdf_path.name,
        "pages": pages,
        "chunks": len(
            kb.chunks
        ),
        "embedding_model": (
            "BAAI/bge-small-en-v1.5"
        ),
        "chunking_strategy": (
            "sentence_aware"
        ),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "retrieval": (
            "semantic + BM25 + "
            "Reciprocal Rank Fusion"
        ),
        "semantic_weight": (
            DEFAULT_SEMANTIC_WEIGHT
        ),
        "keyword_weight": (
            DEFAULT_KEYWORD_WEIGHT
        ),
        "rrf_k": DEFAULT_RRF_K,
        "reranker_model": (
            "BAAI/bge-reranker-base"
        ),
        "vector_database": (
            "Qdrant used by main RAG pipeline"
        ),
        "mcp_retrieval": (
            "In-memory semantic + BM25 "
            "with RRF fusion and "
            "cross-encoder reranking"
        ),
        "status": "indexed",
    }


# ============================================================
# MCP TOOL: DOCUMENT SEARCH
# ============================================================

@mcp.tool()
def search_documents(
    query: str,
    top_k: int = 5,
) -> dict:
    """
    Search the uploaded document using:

        Semantic retrieval
            +
        BM25 retrieval
            ↓
        RRF hybrid fusion
            ↓
        Cross-encoder reranking

    Args:
        query:
            Natural-language question or search query.

        top_k:
            Number of final reranked results.
            Maximum allowed value is 10.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not isinstance(
        query,
        str,
    ):
        return {
            "query": "",
            "results": [],
            "count": 0,
            "error": (
                "Search query must be a string."
            ),
        }

    query = query.strip()

    if not query:

        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": (
                "Search query cannot be empty."
            ),
        }

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    try:
        top_k = int(top_k)
    except (
        TypeError,
        ValueError,
    ):

        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": (
                "top_k must be an integer."
            ),
        }

    top_k = max(
        1,
        min(
            top_k,
            MAX_TOP_K,
        ),
    )

    try:

        # ----------------------------------------------------
        # Lazy-load knowledge base
        # ----------------------------------------------------

        kb = get_knowledge_base()

        # ----------------------------------------------------
        # Execute complete retrieval pipeline
        # ----------------------------------------------------

        results = kb.search(
            query=query,
            top_k=top_k,
        )

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "retrieval": {
                "semantic": True,
                "bm25": True,
                "fusion": "rrf",
                "rrf_k": DEFAULT_RRF_K,
                "reranking": "cross_encoder",
            },
        }

    except Exception as exc:

        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": str(exc),
        }


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":
    mcp.run()