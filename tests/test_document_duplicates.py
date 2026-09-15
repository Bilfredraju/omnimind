from pathlib import Path

import fitz
import pytest

from rag.ingestion.document_manager import DocumentManager


def create_pdf(path: Path, title: str = "Test Document"):
    """Create a minimal valid PDF using the project's PyMuPDF dependency."""

    document = fitz.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        title,
    )

    metadata = document.metadata
    metadata["title"] = title
    document.set_metadata(metadata)

    document.save(str(path))
    document.close()


def test_duplicate_content_is_detected(tmp_path):
    """The same PDF content should not be registered twice by default."""

    file1 = tmp_path / "handbook.pdf"
    file2 = tmp_path / "handbook-copy.pdf"

    create_pdf(file1, title="Same Document")

    # Make the second file byte-for-byte identical.
    file2.write_bytes(file1.read_bytes())

    manager = DocumentManager(
        registry_path=tmp_path / "document_registry.json"
    )

    first = manager.add_document(file1)

    assert first["document_id"]

    with pytest.raises(
        ValueError,
        match="Document is already registered",
    ):
        manager.add_document(file2)

    assert manager.count() == 1


def test_identical_files_produce_same_document_id(tmp_path):
    """Identical files should have the same stable document ID."""

    file1 = tmp_path / "document-a.pdf"
    file2 = tmp_path / "document-b.pdf"

    create_pdf(file1, title="Identical Document")
    file2.write_bytes(file1.read_bytes())

    manager = DocumentManager(
        registry_path=tmp_path / "document_registry.json"
    )

    first = manager.add_document(file1)

    second = manager.add_document(
        file2,
        allow_duplicate=True,
    )

    assert first["document_id"] == second["document_id"]
    assert manager.count() == 1


def test_different_content_is_not_treated_as_duplicate(tmp_path):
    """Different PDF content should receive different document IDs."""

    file1 = tmp_path / "original.pdf"
    file2 = tmp_path / "modified.pdf"

    create_pdf(file1, title="Original Document")
    create_pdf(file2, title="Modified Document")

    manager = DocumentManager(
        registry_path=tmp_path / "document_registry.json"
    )

    first = manager.add_document(file1)
    second = manager.add_document(file2)

    assert first["document_id"] != second["document_id"]
    assert manager.count() == 2
