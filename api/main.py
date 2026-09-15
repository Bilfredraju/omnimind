from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from agents.graph import OmniMindGraph
from rag.ingestion.document_manager import DocumentManager
from rag.ingestion.ingestion_pipeline import DocumentIngestionPipeline
from rag.retrieval.search import SemanticRetriever


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DOCUMENT_REGISTRY_PATH = PROCESSED_DIR / "document_registry.json"
DEFAULT_PDF_PATH = RAW_DIR / "sample.pdf"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="OmniMind API",
    version="20.5.4",
)


# ============================================================
# Lazy shared instances
# ============================================================

_pipeline: DocumentIngestionPipeline | None = None
_graph: OmniMindGraph | None = None


# ============================================================
# Shared document manager
# ============================================================

document_manager = DocumentManager(
    registry_path=DOCUMENT_REGISTRY_PATH,
)


# ============================================================
# Pipeline
# ============================================================

def get_pipeline() -> DocumentIngestionPipeline:
    """
    Return the shared document ingestion pipeline.

    The same DocumentManager is injected so that the API,
    registry, ingestion pipeline, and deletion operations
    operate on the same persistent document registry.
    """
    global _pipeline

    if _pipeline is None:
        _pipeline = DocumentIngestionPipeline(
            document_manager=document_manager,
        )

    return _pipeline


# ============================================================
# Graph
# ============================================================

def get_graph() -> OmniMindGraph:
    """
    Return the shared OmniMind graph.

    The graph is created lazily so importing the API does not
    immediately load embedding or other heavy models.
    """
    global _graph

    if _graph is None:
        _graph = OmniMindGraph(
            pdf_path=str(DEFAULT_PDF_PATH),
        )

    return _graph


# ============================================================
# Request / Response models
# ============================================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    query: str
    current_step: str
    route: str
    sources: list[dict[str, Any]]
    rag_results: list[dict[str, Any]]
    error: str


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {
        "status": "ok",
        "service": "omnimind-document-api",
    }


# ============================================================
# Root
# ============================================================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "OmniMind API",
        "version": "20.5.4",
        "status": "running",
    }


# ============================================================
# Document upload
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload and index a document.

    Flow:

        Upload
          ↓
        Persist source PDF
          ↓
        DocumentIngestionPipeline
          ↓
        Load
          ↓
        Chunk
          ↓
        Embed
          ↓
        Persistent Qdrant
          ↓
        Registry = indexed
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    filename = Path(file.filename).name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    destination = DOCUMENTS_DIR / filename

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        with destination.open("wb") as output_file:
            output_file.write(content)

        pipeline = get_pipeline()

        result = pipeline.ingest_document(
            str(destination),
            allow_duplicate=False,
        )

        return {
            **result,
            "filename": filename,
            "file_path": str(destination),
            "size": len(content),
            "document": result,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Document listing
# ============================================================

@app.get("/documents")
def list_documents() -> dict[str, Any]:
    """
    Return all registered documents.
    """

    try:
        documents = document_manager.list_documents()

        return {
            "count": len(documents),
            "documents": documents,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Document details
# ============================================================

@app.get("/documents/{document_id}")
def get_document(
    document_id: str,
) -> dict[str, Any]:
    """
    Return information about one document.
    """

    try:
        document = document_manager.get_document(
            document_id,
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}",
            )

        return document

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Document status
# ============================================================

@app.get("/documents/{document_id}/status")
def get_document_status(
    document_id: str,
) -> dict[str, Any]:
    """
    Return processing/indexing status for a document.
    """

    try:
        document = document_manager.get_document(
            document_id,
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}",
            )

        return {
            "document_id": document_id,
            "status": document.get("status"),
            "document": document,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Global semantic document search
# ============================================================

@app.get("/search")
def search_all_documents(
    q: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search across all indexed documents using semantic retrieval.
    """

    if not q or not q.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    try:
        retriever = SemanticRetriever()

        try:
            results = retriever.search(
                query=q,
                top_k=top_k,
            )
        finally:
            retriever.close()

        return {
            "query": q,
            "count": len(results),
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Document-specific semantic search
# ============================================================

@app.get("/documents/{document_id}/search")
def search_document(
    document_id: str,
    q: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search only within one indexed document.
    """

    if not q or not q.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    try:
        document = document_manager.get_document(
            document_id,
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}",
            )

        retriever = SemanticRetriever()

        try:
            results = retriever.search(
                query=q,
                top_k=top_k,
                document_id=document_id,
            )
        finally:
            retriever.close()

        return {
            "query": q,
            "document_id": document_id,
            "count": len(results),
            "results": results,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Document deletion
# ============================================================

@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
) -> dict[str, Any]:
    """
    Remove a document from the active OmniMind knowledge base.

    Removes:

        - Qdrant vectors
        - Document registry entry

    Preserves:

        - Original uploaded source PDF

    This allows the original source document to remain available
    for auditing, re-indexing, or future processing.
    """

    try:
        # --------------------------------------------------------
        # 1. Lookup document
        # --------------------------------------------------------

        document = document_manager.get_document(
            document_id,
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}",
            )

        # --------------------------------------------------------
        # 2. Delete vectors from persistent Qdrant
        # --------------------------------------------------------

        pipeline = get_pipeline()

        deleted_vectors = pipeline.vector_store.delete_document(
            document_id,
        )

        # --------------------------------------------------------
        # 3. Remove document from registry
        #
        # DocumentManager uses remove_document().
        # --------------------------------------------------------

        removed_from_registry = document_manager.remove_document(
            document_id,
        )

        # --------------------------------------------------------
        # 4. Preserve original source PDF
        # --------------------------------------------------------

        file_path = document.get(
            "file_path",
        )

        source_preserved = False

        if file_path:
            source_path = Path(file_path)
            source_preserved = (
                source_path.exists()
                and source_path.is_file()
            )

        # --------------------------------------------------------
        # 5. Standardized response
        # --------------------------------------------------------

        return {
            "status": "removed",
            "success": bool(removed_from_registry),
            "document_id": document_id,
            "document_name": document.get(
                "document_name",
            ),
            "deleted_vectors": deleted_vectors,
            "removed_from_registry": bool(
                removed_from_registry,
            ),
            "source_preserved": source_preserved,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Chat / RAG
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Run a query through the complete OmniMind graph.

    Flow:

        User Query
             ↓
        Memory Recall
             ↓
        Planner
             ↓
        RAG / Research
             ↓
        Analysis
             ↓
        Synthesis
             ↓
        Memory Write
             ↓
        Final Response
    """

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    try:
        graph = get_graph()

        initial_state = {
            "query": message,
            "current_step": "start",
            "error": "",
            "sources": [],
        }

        result = graph.run(
            initial_state,
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "OmniMind graph returned an invalid result.",
            )

        response_text = result.get(
            "response",
            result.get(
                "answer",
                "",
            ),
        )

        sources = result.get(
            "sources",
            [],
        )

        if not isinstance(sources, list):
            sources = []

        rag_results = result.get(
            "rag_results",
            [],
        )

        if not isinstance(rag_results, list):
            rag_results = []

        return ChatResponse(
            response=str(
                response_text or "",
            ),
            query=str(
                result.get(
                    "query",
                    message,
                )
                or message
            ),
            current_step=str(
                result.get(
                    "current_step",
                    "",
                )
                or ""
            ),
            route=str(
                result.get(
                    "route",
                    "",
                )
                or ""
            ),
            sources=sources,
            rag_results=rag_results,
            error=str(
                result.get(
                    "error",
                    "",
                )
                or ""
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Application entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )