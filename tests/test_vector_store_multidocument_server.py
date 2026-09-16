from pathlib import Path
import sys
import uuid

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval.vector_store import QdrantVectorStore


QDRANT_URL = "http://127.0.0.1:6333"


def make_chunk(
    document_id: str,
    chunk_id: str,
    text: str,
    page: int,
):
    return {
        "text": text,
        "metadata": {
            "document_id": document_id,
            "document_name": f"{document_id}.pdf",
            "page": page,
            "page_number": page,
            "chunk_id": chunk_id,
        },
    }


def main():
    collection_name = f"test_multidocument_{uuid.uuid4().hex[:8]}"

    print("=" * 70)
    print("Qdrant Multi-Document Vector Store Test")
    print("=" * 70)
    print(f"Collection: {collection_name}")
    print()

    store = QdrantVectorStore(
        collection_name=collection_name,
        url=QDRANT_URL,
    )

    try:
        # ---------------------------------------------------------
        # Document A
        # ---------------------------------------------------------

        document_a = "doc-A"

        chunks_a = [
            make_chunk(
                document_a,
                "chunk-A-1",
                "machine learning predictive maintenance",
                1,
            ),
            make_chunk(
                document_a,
                "chunk-A-2",
                "sensor vibration temperature analysis",
                2,
            ),
        ]

        embeddings_a = [
            [1.0] + [0.0] * 383,
            [0.9, 0.1] + [0.0] * 382,
        ]

        added_a = store.add_documents(
            chunks_a,
            embeddings_a,
        )

        assert added_a == 2
        assert store.count(document_id=document_a) == 2

        print("1. Document A indexing: PASSED")

        # ---------------------------------------------------------
        # Document B
        # ---------------------------------------------------------

        document_b = "doc-B"

        chunks_b = [
            make_chunk(
                document_b,
                "chunk-B-1",
                "retrieval augmented generation",
                1,
            ),
            make_chunk(
                document_b,
                "chunk-B-2",
                "large language model evaluation",
                2,
            ),
        ]

        embeddings_b = [
            [0.0, 1.0] + [0.0] * 382,
            [0.0, 0.9, 0.1] + [0.0] * 381,
        ]

        added_b = store.add_documents(
            chunks_b,
            embeddings_b,
        )

        assert added_b == 2
        assert store.count(document_id=document_b) == 2

        print("2. Document B indexing: PASSED")

        # ---------------------------------------------------------
        # Multi-document count
        # ---------------------------------------------------------

        assert store.count() == 4

        print("3. Multi-document count: PASSED")

        # ---------------------------------------------------------
        # Document isolation
        # ---------------------------------------------------------

        query_embedding = [1.0] + [0.0] * 383

        results_a = store.search(
            query_embedding=query_embedding,
            top_k=10,
            document_id=document_a,
        )

        assert len(results_a) == 2
        assert all(
            result["document_id"] == document_a
            for result in results_a
        )

        print("4. Document A filtering: PASSED")

        results_b = store.search(
            query_embedding=query_embedding,
            top_k=10,
            document_id=document_b,
        )

        assert len(results_b) == 2
        assert all(
            result["document_id"] == document_b
            for result in results_b
        )

        print("5. Document B filtering: PASSED")

        # ---------------------------------------------------------
        # Re-index same chunks
        # ---------------------------------------------------------

        added_again = store.add_documents(
            chunks_a,
            embeddings_a,
        )

        assert added_again == 2
        assert store.count(document_id=document_a) == 2
        assert store.count() == 4

        print("6. Deterministic re-indexing: PASSED")

        # ---------------------------------------------------------
        # Delete one document
        # ---------------------------------------------------------

        deleted = store.delete_document(document_a)

        assert deleted == 2
        assert store.count(document_id=document_a) == 0
        assert store.count(document_id=document_b) == 2
        assert store.count() == 2

        print("7. Document deletion isolation: PASSED")

        print()
        print("=" * 70)
        print("ALL MULTI-DOCUMENT TESTS PASSED")
        print("=" * 70)

    finally:
        store.close()

        # Remove temporary test collection.
        cleanup = QdrantClient(url=QDRANT_URL)

        try:
            cleanup.delete_collection(
                collection_name=collection_name
            )
        finally:
            cleanup.close()


if __name__ == "__main__":
    main()
