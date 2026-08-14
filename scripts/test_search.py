import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval.search import SemanticRetriever


print("=" * 60)
print("OMNIMIND SEMANTIC SEARCH TEST")
print("=" * 60)


retriever = SemanticRetriever()


query = "What is Retrieval-Augmented Generation?"


print(f"\nQuery:")
print(query)

print("\nSearching Qdrant...")


results = retriever.search(
    query=query,
    top_k=5,
)


print("\n" + "=" * 60)
print(f"TOP {len(results)} RESULTS")
print("=" * 60)


for index, result in enumerate(results, start=1):

    metadata = result["metadata"]

    print(f"\nResult {index}")
    print("-" * 60)

    print(f"Similarity score: {result['score']:.4f}")
    print(f"Source: {metadata['source']}")
    print(f"Page: {metadata['page']}")
    print(f"Chunk: {metadata['chunk_index']}")

    print("\nText:")
    print(result["text"][:700])


print("\n" + "=" * 60)
print("SEMANTIC SEARCH SUCCESSFUL")
print("=" * 60)