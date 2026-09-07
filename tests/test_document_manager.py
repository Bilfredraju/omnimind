from pathlib import Path
import json
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.document_manager import DocumentManager


def main():
    sample_pdf = PROJECT_ROOT / "data" / "raw" / "sample.pdf"

    if not sample_pdf.exists():
        raise FileNotFoundError(f"Sample PDF not found: {sample_pdf}")

    with tempfile.TemporaryDirectory() as temp_dir:
        registry_path = Path(temp_dir) / "document_registry.json"

        manager = DocumentManager(registry_path=registry_path)

        # ---------------------------------------------------------
        # 1. Register document
        # ---------------------------------------------------------
        document = manager.add_document(sample_pdf)

        assert document["document_name"] == "sample.pdf"
        assert document["file_path"] == str(sample_pdf.resolve())
        assert document["source_type"] == "pdf"
        assert document["status"] == "registered"
        assert document["page_count"] == 19
        assert document["file_size_bytes"] > 0
        assert document["document_id"]

        document_id = document["document_id"]

        print("1. Document registration: PASSED")
        print(f"   Document ID: {document_id}")
        print(f"   Pages: {document['page_count']}")

        # ---------------------------------------------------------
        # 2. Count
        # ---------------------------------------------------------
        assert manager.count() == 1

        print("2. Registry count: PASSED")

        # ---------------------------------------------------------
        # 3. Retrieve by document ID
        # ---------------------------------------------------------
        retrieved = manager.get_document(document_id)

        assert retrieved is not None
        assert retrieved["document_id"] == document_id
        assert retrieved["document_name"] == "sample.pdf"

        print("3. Get document by ID: PASSED")

        # ---------------------------------------------------------
        # 4. Retrieve by path
        # ---------------------------------------------------------
        retrieved_by_path = manager.get_document_by_path(sample_pdf)

        assert retrieved_by_path is not None
        assert retrieved_by_path["document_id"] == document_id

        print("4. Get document by path: PASSED")

        # ---------------------------------------------------------
        # 5. Duplicate detection
        # ---------------------------------------------------------
        duplicate_detected = False

        try:
            manager.add_document(sample_pdf)
        except ValueError:
            duplicate_detected = True

        assert duplicate_detected

        print("5. Duplicate detection: PASSED")

        # ---------------------------------------------------------
        # 6. Update status
        # ---------------------------------------------------------
        updated = manager.update_status(
            document_id,
            "indexed",
        )

        assert updated["status"] == "indexed"

        print("6. Status update: PASSED")

        # ---------------------------------------------------------
        # 7. Persistence
        # ---------------------------------------------------------
        assert registry_path.exists()

        with registry_path.open("r", encoding="utf-8") as f:
            raw_registry = json.load(f)

        assert document_id in raw_registry["documents"]

        print("7. Registry persistence: PASSED")

        # ---------------------------------------------------------
        # 8. Reload manager from disk
        # ---------------------------------------------------------
        manager_reloaded = DocumentManager(
            registry_path=registry_path
        )

        assert manager_reloaded.count() == 1

        reloaded = manager_reloaded.get_document(document_id)

        assert reloaded is not None
        assert reloaded["document_id"] == document_id
        assert reloaded["status"] == "indexed"

        print("8. Registry reload: PASSED")

        # ---------------------------------------------------------
        # 9. List documents
        # ---------------------------------------------------------
        documents = manager_reloaded.list_documents()

        assert len(documents) == 1
        assert documents[0]["document_id"] == document_id

        print("9. List documents: PASSED")

        # ---------------------------------------------------------
        # 10. Remove registry entry
        # ---------------------------------------------------------
        removed = manager_reloaded.remove_document(document_id)

        assert removed is True
        assert manager_reloaded.count() == 0
        assert manager_reloaded.get_document(document_id) is None

        # IMPORTANT:
        # Removing a registry entry must NOT delete the source PDF.
        assert sample_pdf.exists()

        print("10. Registry removal: PASSED")
        print("    Source PDF preserved: PASSED")

    print()
    print("=" * 60)
    print("DocumentManager tests: ALL PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()