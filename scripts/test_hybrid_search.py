import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.retrieval.hybrid import HybridRetriever


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


print("=" * 60)
print("OMNIMIND HYBRID SEARCH TEST")
print("=" * 60)


# ---------------------------------------------------------
# Load document
# ---------------------------------------------------------

documents = load_pdf(
    str(PDF_PATH)
)

print(
    f"\nPages loaded: {len(documents)}"
)


# ---------------------------------------------------------
# Chunk document
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
# Create hybrid retriever
# ---------------------------------------------------------

retriever = HybridRetriever(
    chunks=chunks,
    semantic_weight=0.7,
    keyword_weight=0.3,
)


# ---------------------------------------------------------
# Query
# ---------------------------------------------------------

query = (
    "What datasets were used to evaluate "
    "the RAG models?"
)


print("\nQuery:")
print(query)

print("\nSearching...")


results = retriever.search(
    query=query,
    top_k=5,
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("HYBRID SEARCH RESULTS")
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
        f"Hybrid score: "
        f"{result['hybrid_score']:.4f}"
    )

    print(
        f"Semantic score: "
        f"{result['semantic_score']:.4f}"
    )

    print(
        f"Keyword score: "
        f"{result['keyword_score']:.4f}"
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
print("HYBRID SEARCH SUCCESSFUL")
print("=" * 60)