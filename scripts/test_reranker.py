import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


print("=" * 60)
print("OMNIMIND RERANKER TEST")
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
# 2. Create chunks
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

print(
    "\nRunning hybrid retrieval..."
)


candidates = hybrid_retriever.search(
    query=query,
    top_k=10,
)


print(
    f"Candidates retrieved: "
    f"{len(candidates)}"
)


# ---------------------------------------------------------
# 4. Cross-encoder reranking
# ---------------------------------------------------------

reranker = CrossEncoderReranker()


print(
    "\nRunning cross-encoder reranking..."
)


results = reranker.rerank(
    query=query,
    documents=candidates,
    top_k=5,
)


# ---------------------------------------------------------
# 5. Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RERANKED RESULTS")
print("=" * 60)


for index, result in enumerate(
    results,
    start=1,
):

    metadata = result["metadata"]

    print(
        f"\nResult {index}"
    )

    print("-" * 60)

    print(
        f"Rerank score: "
        f"{result['rerank_score']:.4f}"
    )

    print(
        f"Hybrid score: "
        f"{result['hybrid_score']:.4f}"
    )

    print(
        f"Source: {metadata['source']}"
    )

    print(
        f"Page: {metadata['page']}"
    )

    print(
        f"Chunk: {metadata['chunk_index']}"
    )

    print("\nText:")

    print(
        result["text"][:700]
    )


print("\n" + "=" * 60)
print("RERANKING SUCCESSFUL")
print("=" * 60)