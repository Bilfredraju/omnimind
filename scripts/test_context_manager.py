import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker
from memory.context_manager import ContextManager


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


print("=" * 60)
print("OMNIMIND CONTEXT MANAGER TEST")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load PDF
# ---------------------------------------------------------

documents = load_pdf(
    str(PDF_PATH)
)

print(
    f"\nPages loaded: {len(documents)}"
)


# ---------------------------------------------------------
# 2. Chunk document
# ---------------------------------------------------------

chunks = chunk_documents(
    documents,
    chunk_size=800,
    chunk_overlap=150,
)

print(
    f"Chunks created: {len(chunks)}"
)


# ---------------------------------------------------------
# 3. Hybrid retrieval
# ---------------------------------------------------------

hybrid_retriever = HybridRetriever(
    chunks=chunks,
    semantic_weight=0.7,
    keyword_weight=0.3,
)


query = (
    "What datasets were used to evaluate "
    "the RAG models?"
)


print("\nQuery:")
print(query)


candidates = hybrid_retriever.search(
    query=query,
    top_k=10,
)

print(
    f"\nHybrid candidates: {len(candidates)}"
)


# ---------------------------------------------------------
# 4. Reranking
# ---------------------------------------------------------

reranker = CrossEncoderReranker()

reranked_documents = reranker.rerank(
    query=query,
    documents=candidates,
    top_k=5,
)

print(
    f"Reranked documents: "
    f"{len(reranked_documents)}"
)


# ---------------------------------------------------------
# 5. Context management
# ---------------------------------------------------------

context_manager = ContextManager(
    max_context_chars=5000,
)


context_result = (
    context_manager.build_context(
        reranked_documents
    )
)


# ---------------------------------------------------------
# 6. Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CONTEXT MANAGEMENT RESULTS")
print("=" * 60)

print(
    f"\nDocuments selected: "
    f"{context_result['total_documents']}"
)

print(
    f"Total context characters: "
    f"{context_result['total_characters']}"
)

print("\n" + "-" * 60)
print("FINAL LLM CONTEXT")
print("-" * 60)

print(
    context_result["context"]
)


print("\n" + "=" * 60)
print("CONTEXT MANAGEMENT SUCCESSFUL")
print("=" * 60)