import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents
from rag.embeddings.embedder import EmbeddingModel


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample.pdf"


print("=" * 60)
print("OMNIMIND EMBEDDING TEST")
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
# 3. Load embedding model
# ---------------------------------------------------------

embedder = EmbeddingModel()


# ---------------------------------------------------------
# 4. Generate embeddings
# ---------------------------------------------------------

texts = [chunk["text"] for chunk in chunks]

embeddings = embedder.encode(texts)


# ---------------------------------------------------------
# 5. Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EMBEDDING RESULTS")
print("=" * 60)

print(f"Number of embeddings: {len(embeddings)}")

if embeddings:
    print(f"Embedding dimension: {len(embeddings[0])}")

    print("\nFirst embedding:")
    print(embeddings[0][:10])

print("\n" + "=" * 60)
print("EMBEDDING GENERATION SUCCESSFUL")
print("=" * 60)