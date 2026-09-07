from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.embeddings.embedder import EmbeddingModel
from rag.ingestion.document_manager import DocumentManager
from rag.ingestion.ingestion_pipeline import (
    DocumentIngestionPipeline,
)
from rag.retrieval.vector_store import QdrantVectorStore


def main():
    sample_pdf = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "sample.pdf"
    )

    if not sample_pdf.exists():
        raise FileNotFoundError(
            f"Sample PDF not found: {sample_pdf}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        registry_path = (
            temp_dir
            / "document_registry.json"
        )

        qdrant_path = (
            temp_dir
            / "qdrant"
        )

        document_manager = DocumentManager(
            registry_path=registry_path
        )

        vector_store = QdrantVectorStore(
            collection_name="test_ingestion_pipeline",
            vector_size=384,
            storage_path=qdrant_path,
        )

        embedding_model = EmbeddingModel()

        pipeline = DocumentIngestionPipeline(
            document_manager=document_manager,
            vector_store=vector_store,
            embedding_model=embedding_model,
            chunk_size=800,
            chunk_overlap=150,
        )

        # ---------------------------------------------------------
        # 1. Full ingestion
        # ---------------------------------------------------------

        result = pipeline.ingest_document(
            sample_pdf
        )

        assert result["success"] is True
        assert result["status"] == "indexed"
        assert result["document_name"] == "sample.pdf"
        assert result["page_count"] == 19
        assert result["chunk_count"] > 0
        assert result["vector_count"] == result["chunk_count"]

        document_id = result["document_id"]

        print("1. End-to-end ingestion: PASSED")
        print(f"   Document ID: {document_id}")
        print(f"   Pages: {result['page_count']}")
        print(f"   Chunks: {result['chunk_count']}")

        # ---------------------------------------------------------
        # 2. Registry status
        # ---------------------------------------------------------

        registered = document_manager.get_document(
            document_id
        )

        assert registered is not None
        assert registered["status"] == "indexed"

        print("2. Registry status = indexed: PASSED")

        # ---------------------------------------------------------
        # 3. Vector count
        # ---------------------------------------------------------

        vector_count = vector_store.count(
            document_id=document_id
        )

        assert vector_count == result["chunk_count"]

        print("3. Qdrant vector count: PASSED")

        # ---------------------------------------------------------
        # 4. Document exists in vector store
        # ---------------------------------------------------------

        assert vector_store.document_exists(
            document_id
        )

        print("4. Vector document existence: PASSED")

        # ---------------------------------------------------------
        # 5. Semantic search
        # ---------------------------------------------------------

        query_embedding = embedding_model.encode_single(
            "RAG model evaluation datasets"
        )

        search_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=5,
            document_id=document_id,
        )

        assert len(search_results) > 0

        assert all(
            result["document_id"] == document_id
            for result in search_results
        )

        print("5. Indexed document semantic search: PASSED")

        # ---------------------------------------------------------
        # 6. Persistence
        # ---------------------------------------------------------

        pipeline.close()

        vector_store_reloaded = QdrantVectorStore(
            collection_name="test_ingestion_pipeline",
            vector_size=384,
            storage_path=qdrant_path,
        )

        assert vector_store_reloaded.count(
            document_id=document_id
        ) == result["chunk_count"]

        print("6. Qdrant persistence: PASSED")

        # ---------------------------------------------------------
        # 7. Registry persistence
        # ---------------------------------------------------------

        document_manager_reloaded = DocumentManager(
            registry_path=registry_path
        )

        persisted_document = (
            document_manager_reloaded.get_document(
                document_id
            )
        )

        assert persisted_document is not None
        assert persisted_document["status"] == "indexed"

        print("7. Registry persistence: PASSED")

        # ---------------------------------------------------------
        # 8. Remove document
        # ---------------------------------------------------------

        pipeline_reloaded = DocumentIngestionPipeline(
            document_manager=document_manager_reloaded,
            vector_store=vector_store_reloaded,
            embedding_model=embedding_model,
        )

        removed = pipeline_reloaded.remove_document(
            document_id
        )

        assert removed is True

        assert (
            document_manager_reloaded.get_document(
                document_id
            )
            is None
        )

        assert (
            vector_store_reloaded.count(
                document_id=document_id
            )
            == 0
        )

        # Source PDF must still exist.
        assert sample_pdf.exists()

        print("8. Document removal: PASSED")
        print("   Source PDF preserved: PASSED")

        pipeline_reloaded.close()

    print()
    print("=" * 60)
    print("Unified ingestion pipeline tests: ALL PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()