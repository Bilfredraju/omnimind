import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.loader import load_pdf
from rag.ingestion.chunker import chunk_documents


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample.pdf"


# Load PDF
documents = load_pdf(str(PDF_PATH))

# Create chunks
chunks = chunk_documents(
    documents,
    chunk_size=800,
    chunk_overlap=150,
)


print("=" * 60)
print("OMNIMIND CHUNKING TEST")
print("=" * 60)

print(f"Pages loaded: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

print("\nFIRST 5 CHUNKS")
print("=" * 60)

for i, chunk in enumerate(chunks[:5], start=1):
    print(f"\nChunk {i}")
    print("-" * 60)

    print(
        f"Source: {chunk['metadata']['source']}"
    )

    print(
        f"Page: {chunk['metadata']['page']}"
    )

    print(
        f"Chunk index: {chunk['metadata']['chunk_index']}"
    )

    print(
        f"Characters: {len(chunk['text'])}"
    )

    print("-" * 60)
    print(chunk["text"][:500])


print("\n" + "=" * 60)
print("CHUNKING SUCCESSFUL")
print("=" * 60)