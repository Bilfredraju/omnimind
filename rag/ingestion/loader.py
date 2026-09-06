from hashlib import sha256
from pathlib import Path

import pymupdf


def _build_document_id(pdf_path: Path) -> str:
    """
    Build a stable document identifier from the resolved file path.

    The identifier is deterministic for the same file location and can
    be used to associate pages/chunks with their source document.
    """
    normalized_path = str(pdf_path.resolve()).lower()
    digest = sha256(normalized_path.encode("utf-8")).hexdigest()
    return f"doc-{digest[:16]}"


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Load a PDF page by page with production-oriented source metadata.

    Returns:
        A list of dictionaries containing:
        - page text
        - document metadata
        - page metadata

    Existing metadata keys such as ``source`` and ``page`` are preserved
    for backward compatibility.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not path.is_file():
        raise ValueError(f"The provided PDF path is not a file: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    document_id = _build_document_id(path)

    documents = []

    with pymupdf.open(str(path)) as pdf:
        page_count = len(pdf)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "document_id": document_id,
                        "document_name": path.name,
                        "source": path.name,
                        "page": page_number,
                        "page_number": page_number,
                        "page_count": page_count,
                        "file_path": str(path.resolve()),
                        "source_type": "pdf",
                    },
                }
            )

    return documents