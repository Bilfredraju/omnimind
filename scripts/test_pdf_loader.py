import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.loader import load_pdf


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample.pdf"


documents = load_pdf(str(PDF_PATH))

print("=" * 60)
print("OMNIMIND PDF LOADER TEST")
print("=" * 60)

print(f"PDF: {PDF_PATH}")
print(f"Pages extracted: {len(documents)}")

for document in documents[:3]:
    print("\n" + "-" * 60)
    print(f"Source: {document['metadata']['source']}")
    print(f"Page: {document['metadata']['page']}")
    print("-" * 60)
    print(document["text"][:1000])

print("\n" + "=" * 60)
print("PDF LOADING SUCCESSFUL")
print("=" * 60)