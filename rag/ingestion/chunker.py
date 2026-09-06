from hashlib import sha256
from typing import List


def _build_chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
) -> str:
    """
    Build a deterministic identifier for a document chunk.
    """
    raw_id = f"{document_id}:{page_number}:{chunk_index}"
    digest = sha256(raw_id.encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[dict]:
    """
    Split page-level documents into overlapping chunks.

    Chunk metadata includes stable document/chunk identifiers and
    character offsets so downstream retrieval and citation systems
    can trace a chunk back to its source.

    Args:
        documents:
            Documents returned by the PDF loader.

        chunk_size:
            Maximum number of characters in each chunk.

        chunk_overlap:
            Number of characters shared between adjacent chunks.

    Returns:
        A list of chunk dictionaries with preserved metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks = []

    for document in documents:
        text = document.get("text", "").strip()

        if not text:
            continue

        metadata = document.get("metadata", {}).copy()

        document_id = metadata.get("document_id", "unknown-document")
        page_number = metadata.get(
            "page_number",
            metadata.get("page", 1),
        )

        page_chunks = []

        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = _build_chunk_id(
                    document_id=document_id,
                    page_number=page_number,
                    chunk_index=chunk_index,
                )

                chunk_metadata = metadata.copy()

                # Stable source identity.
                chunk_metadata["document_id"] = document_id
                chunk_metadata["chunk_id"] = chunk_id

                # Preserve the existing field.
                chunk_metadata["chunk_index"] = chunk_index

                # Explicit page identity.
                chunk_metadata["page"] = page_number
                chunk_metadata["page_number"] = page_number

                # Character offsets in the original page text.
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = end

                page_chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": chunk_metadata,
                    }
                )

                chunk_index += 1

            if end >= text_length:
                break

            start = end - chunk_overlap

        # Store how many chunks were produced from this page.
        chunk_count = len(page_chunks)

        for chunk in page_chunks:
            chunk["metadata"]["chunk_count"] = chunk_count
            chunks.append(chunk)

    return chunks