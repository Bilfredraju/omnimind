from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from rag.retrieval.vector_store import QdrantVectorStore


def main():
    store = QdrantVectorStore()
    client = store.client
    collection_name = store.collection_name

    print(f"Collection: {collection_name}")

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    legacy_ids = []

    for point in points:
        payload = point.payload or {}
        document_id = payload.get("document_id")

        if not document_id:
            legacy_ids.append(point.id)

    print(f"Total points: {len(points)}")
    print(f"Legacy points without document_id: {len(legacy_ids)}")

    if not legacy_ids:
        print("No legacy vectors found.")
        store.close()
        return

    print("Legacy point IDs:")
    for point_id in legacy_ids:
        print(f"  {point_id}")

    client.delete(
        collection_name=collection_name,
        points_selector=legacy_ids,
        wait=True,
    )

    remaining = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    print(f"Remaining Qdrant points: {remaining}")

    store.close()


if __name__ == "__main__":
    main()