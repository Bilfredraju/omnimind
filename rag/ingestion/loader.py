from pathlib import Path

import pymupdf


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Load a PDF and extract text page by page.

    Returns:
        A list of dictionaries containing page text and metadata.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    documents = []

    pdf = pymupdf.open(pdf_path)

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                    "file_path": str(path),
                },
            }
        )

    pdf.close()

    return documents