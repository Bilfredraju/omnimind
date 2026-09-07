from pathlib import Path
import shutil
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval.vector_store import QdrantVectorStore


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
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "qdrant"

        store = QdrantVectorStore(
            collection_name="test_multidocument",
            vector_size=3,
            storage_path=storage_path,
        )

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
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ]

        ids_a = store.add_documents(
            chunks_a,
            embeddings_a,
        )

        assert len(ids_a) == 2
        assert len(set(ids_a)) == 2

        print("1. Document A indexing: PASSED")

        # ---------------------------------------------------------
        # Document B
        # ---------------------------------------------------------

        document_b = "doc-B"

        chunks_b = [
            make_chunk(
                document_b,
                "chunk-B-1",
                "financial forecasting revenue analysis",
                1,
            ),
            make_chunk(
                document_b,
                "chunk-B-2",
                "market trends business intelligence",
                2,
            ),
        ]

        embeddings_b = [
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]

        ids_b = store.add_documents(
            chunks_b,
            embeddings_b,
        )

        assert len(ids_b) == 2
        assert len(set(ids_b)) == 2

        print("2. Document B indexing: PASSED")

        # ---------------------------------------------------------
        # Verify global uniqueness
        # ---------------------------------------------------------

        all_ids = ids_a + ids_b

        assert len(all_ids) == 4
        assert len(set(all_ids)) == 4

        print("3. Global vector ID uniqueness: PASSED")

        # ---------------------------------------------------------
        # Verify total count
        # ---------------------------------------------------------

        assert store.count() == 4

        print("4. Total vector count: PASSED")

        # ---------------------------------------------------------
        # Verify document-level counts
        # ---------------------------------------------------------

        assert store.count(
            document_id=document_a
        ) == 2

        assert store.count(
            document_id=document_b
        ) == 2

        print("5. Document-level counts: PASSED")

        # ---------------------------------------------------------
        # Verify cross-document search
        # ---------------------------------------------------------

        results = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=4,
        )

        assert len(results) == 4

        result_document_ids = {
            result["document_id"]
            for result in results
        }

        assert document_a in result_document_ids
        assert document_b in result_document_ids

        print("6. Cross-document search: PASSED")

        # ---------------------------------------------------------
        # Verify document filter
        # ---------------------------------------------------------

        filtered_results = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            document_id=document_a,
        )

        assert len(filtered_results) == 2

        assert all(
            result["document_id"] == document_a
            for result in filtered_results
        )

        print("7. Document-level search filter: PASSED")

        # ---------------------------------------------------------
        # Verify metadata
        # ---------------------------------------------------------

        first = filtered_results[0]

        assert first["metadata"]["document_id"] == document_a
        assert first["metadata"]["chunk_id"]
        assert first["chunk_id"]

        print("8. Metadata preservation: PASSED")

        # ---------------------------------------------------------
        # Verify deterministic re-indexing
        # ---------------------------------------------------------

        ids_a_again = store.add_documents(
            chunks_a,
            embeddings_a,
        )

        assert ids_a_again == ids_a
        assert store.count() == 4
        assert store.count(
            document_id=document_a
        ) == 2

        print("9. Deterministic re-indexing: PASSED")

        # ---------------------------------------------------------
        # Verify document_exists
        # ---------------------------------------------------------

        assert store.document_exists(
            document_a
        )

        assert store.document_exists(
            document_b
        )

        assert not store.document_exists(
            "nonexistent-document"
        )

        print("10. Document existence check: PASSED")

        # ---------------------------------------------------------
        # Verify persistence
        # ---------------------------------------------------------

        store.close()

        store_reloaded = QdrantVectorStore(
            collection_name="test_multidocument",
            vector_size=3,
            storage_path=storage_path,
        )

        assert store_reloaded.count() == 4

        assert store_reloaded.count(
            document_id=document_a
        ) == 2

        assert store_reloaded.count(
            document_id=document_b
        ) == 2

        print("11. Qdrant persistence/reload: PASSED")

        # ---------------------------------------------------------
        # Verify deletion of one document
        # ---------------------------------------------------------

        store_reloaded.delete_document(
            document_a
        )

        assert store_reloaded.count() == 2

        assert store_reloaded.count(
            document_id=document_a
        ) == 0

        assert store_reloaded.count(
            document_id=document_b
        ) == 2

        print("12. Document-specific deletion: PASSED")

        store_reloaded.close()

    print()
    print("=" * 60)
    print("Multi-document Qdrant tests: ALL PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()