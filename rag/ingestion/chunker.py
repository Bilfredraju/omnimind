from typing import List


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[dict]:
    """
    Split page-level documents into smaller overlapping chunks.

    Args:
        documents: Documents returned by the PDF loader.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        A list of chunk dictionaries with preserved metadata.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks = []

    for document in documents:
        text = document["text"].strip()
        metadata = document["metadata"].copy()

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index

                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": chunk_metadata,
                    }
                )

                chunk_index += 1

            if end >= len(text):
                break

            start = end - chunk_overlap

    return chunks