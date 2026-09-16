from pathlib import Path

import fitz

from rag.ingestion.document_manager import DocumentManager
from rag.ingestion.ingestion_pipeline import DocumentIngestionPipeline


class FakeEmbeddingModel:
    """Deterministic embedding model for isolated pipeline testing."""

    def encode(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    """Minimal vector store that tracks document-level replacement."""

    def __init__(self):
        self.documents = {}

    def add_documents(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk["metadata"]
            document_id = metadata["document_id"]
            chunk_id = metadata["chunk_id"]

            if document_id not in self.documents:
                self.documents[document_id] = {}

            self.documents[document_id][chunk_id] = {
                "chunk": chunk,
                "embedding": embedding,
            }

        return len(chunks)

    def count(self, document_id=None):
        if document_id is None:
            return sum(
                len(chunks)
                for chunks in self.documents.values()
            )

        return len(self.documents.get(document_id, {}))

    def document_exists(self, document_id):
        return self.count(document_id=document_id) > 0

    def delete_document(self, document_id):
        existing = self.count(document_id=document_id)
        self.documents.pop(document_id, None)
        return existing

    def close(self):
        pass


def create_pdf(path: Path, text: str):
    """Create a small valid PDF for the ingestion test."""

    document = fitz.open()

    page = document.new_page()
    page.insert_text((72, 72), text)

    document.save(str(path))
    document.close()


def test_reindexing_same_document_does_not_duplicate_vectors(tmp_path):
    """Re-indexing the same document must keep exactly one vector per chunk."""

    pdf_path = tmp_path / "document.pdf"
    registry_path = tmp_path / "document_registry.json"

    create_pdf(
        pdf_path,
        "OmniMind predictive maintenance sensor analysis.",
    )

    document_manager = DocumentManager(
        registry_path=registry_path
    )

    vector_store = FakeVectorStore()
    embedding_model = FakeEmbeddingModel()

    pipeline = DocumentIngestionPipeline(
        document_manager=document_manager,
        vector_store=vector_store,
        embedding_model=embedding_model,
        chunk_size=800,
        chunk_overlap=150,
    )

    # First ingestion.
    first = pipeline.ingest_document(pdf_path)

    assert first["success"] is True
    assert first["status"] == "indexed"

    document_id = first["document_id"]
    first_chunk_count = first["chunk_count"]

    assert first_chunk_count > 0
    assert vector_store.count(document_id) == first_chunk_count

    # The current pipeline requires allow_duplicate=True for a second
    # registration attempt. This test verifies that deterministic point
    # IDs/upsert behavior prevents vector duplication.
    second = pipeline.ingest_document(
        pdf_path,
        allow_duplicate=True,
    )

    assert second["success"] is True
    assert second["status"] == "indexed"
    assert second["document_id"] == document_id
    assert second["chunk_count"] == first_chunk_count

    # Vector count must remain unchanged after re-indexing.
    assert vector_store.count(document_id) == first_chunk_count
    assert vector_store.count() == first_chunk_count

    pipeline.close()
