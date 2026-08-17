from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker
from memory.context_manager import ContextManager
from models.llm.groq_provider import GroqProvider

from rag.generation.prompt import build_rag_prompt


class RAGPipeline:
    """Complete OmniMind RAG pipeline."""

    def __init__(
        self,
        pdf_path: str,
    ):
        self.pdf_path = pdf_path

        # Load document
        documents = load_pdf(
            pdf_path
        )

        # Chunk document
        self.chunks = chunk_documents(
            documents,
            chunk_size=800,
            chunk_overlap=150,
        )

        print(
            f"Loaded {len(documents)} pages."
        )

        print(
            f"Created {len(self.chunks)} chunks."
        )

        # Retrieval
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            semantic_weight=0.7,
            keyword_weight=0.3,
        )

        # Reranker
        self.reranker = CrossEncoderReranker()

        # Context manager
        self.context_manager = ContextManager(
            max_context_chars=5000,
        )

        # LLM
        self.llm = GroqProvider()

    def ask(
        self,
        query: str,
    ) -> dict:
        """Answer a question using the RAG pipeline."""

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
        # 3. Context management
        # -------------------------------------------------

        context_result = (
            self.context_manager.build_context(
                reranked
            )
        )

        # -------------------------------------------------
        # 4. Build prompt
        # -------------------------------------------------

        prompt = build_rag_prompt(
            query=query,
            context=context_result["context"],
        )

        # -------------------------------------------------
        # 5. Generate answer
        # -------------------------------------------------

        answer = self.llm.generate(
            prompt
        )

        return {
            "query": query,
            "answer": answer,
            "sources": context_result[
                "documents"
            ],
        }

    def close(self):
        """Close resources used by the RAG pipeline."""
        self.retriever.close()