from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from rag.ingestion.ingestion_pipeline import DocumentIngestionPipeline


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OmniMind Document API",
    description="Document ingestion and knowledge-base management API for OmniMind.",
    version="20.4.0",
)


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

_pipeline: Optional[DocumentIngestionPipeline] = None


def get_pipeline() -> DocumentIngestionPipeline:
    global _pipeline

    if _pipeline is None:
        _pipeline = DocumentIngestionPipeline()

    return _pipeline


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "omnimind-document-api",
    }


# ---------------------------------------------------------------------------
# Upload + ingest
# ---------------------------------------------------------------------------

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    filename = Path(file.filename).name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    destination = DOCUMENTS_DIR / filename

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        destination.write_bytes(contents)

        pipeline = get_pipeline()

        result = pipeline.ingest_document(destination)

        return {
            "status": "indexed",
            "filename": filename,
            "document": result,
        }

    except HTTPException:
        if destination.exists() and destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

@app.get("/documents")
def list_documents():
    pipeline = get_pipeline()

    documents = pipeline.document_manager.list_documents()

    return {
        "count": len(documents),
        "documents": documents,
    }


# ---------------------------------------------------------------------------
# Get document
# ---------------------------------------------------------------------------

@app.get("/documents/{document_id}")
def get_document(document_id: str):
    pipeline = get_pipeline()

    document = pipeline.document_manager.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    return document


# ---------------------------------------------------------------------------
# Document status
# ---------------------------------------------------------------------------

@app.get("/documents/{document_id}/status")
def get_document_status(document_id: str):
    pipeline = get_pipeline()

    status = pipeline.get_document_status(document_id)

    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    return status


# ---------------------------------------------------------------------------
# Delete document
# ---------------------------------------------------------------------------

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    pipeline = get_pipeline()

    document = pipeline.document_manager.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    file_path = Path(document["file_path"])

    result = pipeline.remove_document(document_id)

    return {
        "status": "removed",
        "document_id": document_id,
        "source_preserved": file_path.exists(),
        "result": result,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/search")
def search_documents(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    document_id: Optional[str] = Query(None),
):
    pipeline = get_pipeline()

    try:
        query_embedding = pipeline.embedding_model.encode_single(q)

        results = pipeline.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )

        return {
            "query": q,
            "top_k": top_k,
            "document_id": document_id,
            "count": len(results),
            "results": results,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Document-specific search
# ---------------------------------------------------------------------------

@app.get("/documents/{document_id}/search")
def search_document(
    document_id: str,
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
):
    pipeline = get_pipeline()

    document = pipeline.document_manager.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    try:
        query_embedding = pipeline.embedding_model.encode_single(q)

        results = pipeline.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )

        return {
            "query": q,
            "document_id": document_id,
            "count": len(results),
            "results": results,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {exc}",
        ) from exc