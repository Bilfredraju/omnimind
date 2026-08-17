import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.generation.rag_pipeline import RAGPipeline


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.pdf"
)


print("=" * 60)
print("OMNIMIND — RAG QUESTION ANSWERING")
print("=" * 60)


pipeline = RAGPipeline(
    pdf_path=str(PDF_PATH)
)


query = input(
    "\nAsk OmniMind a question: "
)


print("\nProcessing...")


result = pipeline.ask(
    query
)


print("\n" + "=" * 60)
print("OMNIMIND ANSWER")
print("=" * 60)

print(
    result["answer"]
)


print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)


seen = set()

for source in result["sources"]:

    metadata = source["metadata"]

    key = (
        metadata["source"],
        metadata["page"],
        metadata["chunk_index"],
    )

    if key in seen:
        continue

    seen.add(key)

    print(
        f"- {metadata['source']} "
        f"| Page {metadata['page']} "
        f"| Chunk {metadata['chunk_index']}"
    )


print("\n" + "=" * 60)
print("RAG PIPELINE COMPLETE")
print("=" * 60)

pipeline.close()