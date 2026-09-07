"""
Integration tests for the OmniMind Document Management API.

Tests:
    1. Health endpoint
    2. PDF upload
    3. Document registration
    4. Indexed status
    5. Global semantic search
    6. Document-specific semantic search
    7. Document lookup
    8. Document deletion
    9. Source PDF preservation
    10. Vector deletion
"""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app, get_pipeline


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "raw" / "sample.pdf"


# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cleanup_test_document(document_id: str | None) -> None:
    """Remove the test document and its vectors if it exists."""
    if not document_id:
        return

    try:
        pipeline = get_pipeline()
        pipeline.remove_document(document_id)
    except Exception:
        pass


def cleanup_existing_sample_documents() -> None:
    """
    Remove previous registrations of sample.pdf.

    This handles both:
    - registrations pointing to data/raw/sample.pdf
    - previous API/test registrations named sample.pdf
    """
    pipeline = get_pipeline()

    # ---------------------------------------------------------------
    # Cleanup by exact source path
    # ---------------------------------------------------------------

    existing_document = (
        pipeline.document_manager.get_document_by_path(
            SAMPLE_PDF.resolve()
        )
    )

    if existing_document:
        pipeline.remove_document(
            existing_document["document_id"]
        )

    # ---------------------------------------------------------------
    # Cleanup any remaining document named sample.pdf
    # ---------------------------------------------------------------

    existing_documents = (
        pipeline.document_manager.list_documents()
    )

    for existing_document in existing_documents:
        if existing_document.get("document_name") == SAMPLE_PDF.name:
            pipeline.remove_document(
                existing_document["document_id"]
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "omnimind-document-api"

    print("1. Health endpoint: PASSED")


def test_document_api_end_to_end():
    assert SAMPLE_PDF.exists(), (
        f"Sample PDF does not exist: {SAMPLE_PDF}"
    )

    # ---------------------------------------------------------------
    # Remove previous test/API registrations
    # ---------------------------------------------------------------

    cleanup_existing_sample_documents()

    document_id = None
    stored_path = None

    try:
        # ---------------------------------------------------------------
        # 2. Upload PDF
        # ---------------------------------------------------------------

        with SAMPLE_PDF.open("rb") as pdf:
            response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        SAMPLE_PDF.name,
                        pdf,
                        "application/pdf",
                    )
                },
            )

        assert response.status_code == 200, response.text

        upload_data = response.json()

        assert upload_data["status"] == "indexed"
        assert upload_data["filename"] == SAMPLE_PDF.name

        document = upload_data["document"]

        assert document["success"] is True
        assert document["status"] == "indexed"
        assert document["page_count"] == 19
        assert document["text_page_count"] == 19
        assert document["chunk_count"] == 105
        assert document["vector_count"] == 105

        document_id = document["document_id"]

        print("2. PDF upload + ingestion: PASSED")
        print(f"   Document ID: {document_id}")
        print(f"   Pages: {document['page_count']}")
        print(f"   Chunks: {document['chunk_count']}")
        print(f"   Vectors: {document['vector_count']}")

        # ---------------------------------------------------------------
        # 3. Registry
        # ---------------------------------------------------------------

        response = client.get("/documents")

        assert response.status_code == 200

        data = response.json()

        matching = [
            item
            for item in data["documents"]
            if item["document_id"] == document_id
        ]

        assert len(matching) == 1

        print("3. Document registry: PASSED")

        # ---------------------------------------------------------------
        # 4. Document lookup
        # ---------------------------------------------------------------

        response = client.get(
            f"/documents/{document_id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["document_id"] == document_id
        assert data["document_name"] == SAMPLE_PDF.name
        assert data["status"] == "indexed"

        print("4. Document lookup: PASSED")

        # ---------------------------------------------------------------
        # 5. Document status
        # ---------------------------------------------------------------

        response = client.get(
            f"/documents/{document_id}/status"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["document_id"] == document_id
        assert data["status"] == "indexed"

        print("5. Document status: PASSED")

        # ---------------------------------------------------------------
        # 6. Global semantic search
        # ---------------------------------------------------------------

        response = client.get(
            "/search",
            params={
                "q": "machine learning",
                "top_k": 5,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["query"] == "machine learning"
        assert data["count"] == 5
        assert len(data["results"]) == 5

        print("6. Global semantic search: PASSED")

        # ---------------------------------------------------------------
        # 7. Document-specific search
        # ---------------------------------------------------------------

        response = client.get(
            f"/documents/{document_id}/search",
            params={
                "q": "machine learning",
                "top_k": 5,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["document_id"] == document_id
        assert data["count"] == 5
        assert len(data["results"]) == 5

        assert all(
            result["document_id"] == document_id
            for result in data["results"]
        )

        print("7. Document-specific search: PASSED")

        # ---------------------------------------------------------------
        # 8. Qdrant vector count
        # ---------------------------------------------------------------

        pipeline = get_pipeline()

        vector_count = pipeline.vector_store.count(
            document_id=document_id
        )

        assert vector_count == 105

        assert pipeline.vector_store.document_exists(
            document_id
        )

        print("8. Qdrant document vectors: PASSED")

        # ---------------------------------------------------------------
        # 9. Verify uploaded source PDF
        # ---------------------------------------------------------------

        stored_path = Path(
            document["file_path"]
        )

        assert stored_path.exists()
        assert stored_path.is_file()
        assert stored_path.stat().st_size > 0

        print("9. Uploaded source PDF: PASSED")

        # ---------------------------------------------------------------
        # 10. Delete document
        # ---------------------------------------------------------------

        response = client.delete(
            f"/documents/{document_id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "removed"
        assert data["document_id"] == document_id
        assert data["source_preserved"] is True

        print("10. Document deletion: PASSED")

        # ---------------------------------------------------------------
        # 11. Verify vectors are gone
        # ---------------------------------------------------------------

        vector_count_after = pipeline.vector_store.count(
            document_id=document_id
        )

        assert vector_count_after == 0

        assert not pipeline.vector_store.document_exists(
            document_id
        )

        print("11. Qdrant vectors removed: PASSED")

        # ---------------------------------------------------------------
        # 12. Verify registry entry is gone
        # ---------------------------------------------------------------

        response = client.get(
            f"/documents/{document_id}"
        )

        assert response.status_code == 404

        print("12. Registry entry removed: PASSED")

        # ---------------------------------------------------------------
        # 13. Verify original source is still present
        # ---------------------------------------------------------------

        assert stored_path is not None
        assert stored_path.exists()
        assert stored_path.is_file()

        print("13. Source PDF preserved after deletion: PASSED")

    finally:
        # ---------------------------------------------------------------
        # Final cleanup
        # ---------------------------------------------------------------

        cleanup_test_document(document_id)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("OmniMind Document API Integration Tests")
    print("=" * 60)
    print()

    test_health()
    test_document_api_end_to_end()

    print()
    print("=" * 60)
    print("Document API tests: ALL PASSED")
    print("=" * 60)