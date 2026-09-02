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

    This implementation intentionally does NOT open
    the local Qdrant database.

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
        Hybrid Rank Fusion
          ↓
        Cross-Encoder Reranking
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
            chunk_size=800,
            chunk_overlap=150,
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
    # SEMANTIC SEARCH
    # ========================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:

        query_embedding = (
            self.embedder.encode_single(
                query
            )
        )

        scored = []

        # ----------------------------------------------------
        # Calculate cosine-style similarity
        #
        # Embeddings from the current model are normalized,
        # so dot product is sufficient here.
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
            key=lambda item: item[0],
            reverse=True,
        )

        results = []

        for score, index in scored[:top_k]:

            chunk = self.chunks[index]

            results.append(
                {
                    "score": score,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )

        return results


    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:

        # ----------------------------------------------------
        # Semantic retrieval
        # ----------------------------------------------------

        semantic_results = (
            self.semantic_search(
                query=query,
                top_k=len(self.chunks),
            )
        )

        # ----------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------

        keyword_results = (
            self.bm25.search(
                query=query,
                top_k=len(self.chunks),
            )
        )

        combined = {}

        # ----------------------------------------------------
        # Add semantic rankings
        # ----------------------------------------------------

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):

            metadata = result["metadata"]

            key = (
                metadata["source"],
                metadata["page"],
                metadata["chunk_index"],
            )

            combined.setdefault(
                key,
                {
                    "text": result["text"],
                    "metadata": metadata,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                },
            )

            combined[key][
                "semantic_score"
            ] = 1 / rank

        # ----------------------------------------------------
        # Add BM25 rankings
        # ----------------------------------------------------

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):

            metadata = result["metadata"]

            key = (
                metadata["source"],
                metadata["page"],
                metadata["chunk_index"],
            )

            combined.setdefault(
                key,
                {
                    "text": result["text"],
                    "metadata": metadata,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                },
            )

            combined[key][
                "keyword_score"
            ] = 1 / rank

        # ----------------------------------------------------
        # Weighted rank fusion
        #
        # Semantic = 70%
        # Keyword  = 30%
        # ----------------------------------------------------

        results = []

        for result in combined.values():

            hybrid_score = (
                0.7
                * result["semantic_score"]
                +
                0.3
                * result["keyword_score"]
            )

            result["hybrid_score"] = (
                hybrid_score
            )

            results.append(result)

        # ----------------------------------------------------
        # Sort by hybrid score
        # ----------------------------------------------------

        results.sort(
            key=lambda item: item[
                "hybrid_score"
            ],
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

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        candidates = (
            self.hybrid_search(
                query=query,
                top_k=10,
            )
        )

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

            results.append(
                {
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "score": result.get(
                        "rerank_score",
                        0.0,
                    ),
                    "source": metadata.get(
                        "source",
                        "Unknown",
                    ),
                    "page": metadata.get(
                        "page",
                        "Unknown",
                    ),
                    "chunk": metadata.get(
                        "chunk_index",
                        "Unknown",
                    ),
                }
            )

        return results


# ============================================================
# LAZY KNOWLEDGE BASE
# ============================================================

knowledge_base = None


def get_knowledge_base() -> DocumentKnowledgeBase:
    """
    Lazily initialize the document knowledge base.

    Heavy ML models are loaded only when a document
    search tool actually needs them.

    This prevents the MCP server from performing
    expensive model initialization during startup.
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

    # --------------------------------------------------------
    # Page count
    #
    # sample.pdf currently contains 19 pages.
    # --------------------------------------------------------

    try:
        pages = len(
            load_pdf(
                str(kb.pdf_path)
            )
        )
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
        "reranker_model": (
            "BAAI/bge-reranker-base"
        ),
        "vector_database": (
            "Qdrant used by main RAG pipeline"
        ),
        "mcp_retrieval": (
            "In-memory semantic + BM25 "
            "with cross-encoder reranking"
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
    Search the uploaded document using the
    hybrid retrieval and reranking pipeline.

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
    # Limit top_k
    # --------------------------------------------------------

    top_k = max(
        1,
        min(top_k, 10),
    )

    try:

        # ----------------------------------------------------
        # Lazy-load knowledge base
        # ----------------------------------------------------

        kb = get_knowledge_base()

        # ----------------------------------------------------
        # Execute retrieval
        # ----------------------------------------------------

        results = kb.search(
            query=query,
            top_k=top_k,
        )

        return {
            "query": query,
            "results": results,
            "count": len(results),
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