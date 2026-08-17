from pathlib import Path

from agents.state import AgentState
from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker


class RAGAgent:
    """Agent responsible for retrieving evidence."""

    def __init__(
        self,
        pdf_path: str,
    ):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        # Load document
        documents = load_pdf(
            str(self.pdf_path)
        )

        # Create chunks
        self.chunks = chunk_documents(
            documents,
            chunk_size=800,
            chunk_overlap=150,
        )

        # Hybrid retrieval
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            semantic_weight=0.7,
            keyword_weight=0.3,
        )

        # Reranker
        self.reranker = CrossEncoderReranker()

    def run(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Retrieve relevant evidence and add it to AgentState.
        """

        query = state["query"]

        # -------------------------------------------------
        # 1. Hybrid retrieval
        # -------------------------------------------------

        candidates = self.retriever.search(
            query=query,
            top_k=10,
        )

        # -------------------------------------------------
        # 2. Reranking
        # -------------------------------------------------

        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=5,
        )

        # -------------------------------------------------
        # 3. Store evidence in shared state
        # -------------------------------------------------

        rag_results = []

        for result in reranked:
            metadata = result["metadata"]

            rag_results.append(
                {
                    "text": result["text"],
                    "score": result["rerank_score"],
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk": metadata["chunk_index"],
                }
            )

        return {
            **state,
            "rag_results": rag_results,
            "current_step": "rag_complete",
        }

    def close(self):
        """Release retrieval resources."""

        self.retriever.close()