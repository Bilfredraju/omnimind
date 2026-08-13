import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.embeddings.embedder import EmbeddingModel
from rag.retrieval.vector_store import QdrantVectorStore


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample.pdf"


print("=" * 60)
print("OMNIMIND QDRANT VECTOR STORE TEST")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load PDF
# ---------------------------------------------------------

documents = load_pdf(str(PDF_PATH))

print(f"\nPages loaded: {len(documents)}")


# ---------------------------------------------------------
# 2. Chunk documents
# ---------------------------------------------------------

chunks = chunk_documents(
    documents,
    chunk_size=800,
    chunk_overlap=150,
)

print(f"Chunks created: {len(chunks)}")


# ---------------------------------------------------------
# 3. Generate embeddings
# ---------------------------------------------------------

embedder = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedder.encode(texts)

print(
    f"Embeddings generated: {len(embeddings)}"
)


# ---------------------------------------------------------
# 4. Create Qdrant store
# ---------------------------------------------------------

vector_store = QdrantVectorStore()

print(
    f"\nVectors before insertion: "
    f"{vector_store.count()}"
)


# ---------------------------------------------------------
# 5. Store embeddings
# ---------------------------------------------------------

vector_store.add_documents(
    chunks,
    embeddings,
)


# ---------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------

print(
    f"Vectors after insertion: "
    f"{vector_store.count()}"
)


print("\n" + "=" * 60)
print("QDRANT VECTOR STORE SUCCESSFUL")
print("=" * 60)