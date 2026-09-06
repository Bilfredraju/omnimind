from __future__ import annotations

import re
from hashlib import sha256
from typing import List


_SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])"
)

_PARAGRAPH_PATTERN = re.compile(
    r"\n\s*\n+"
)


def _build_chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
) -> str:
    """
    Build a deterministic chunk ID.

    The same document/page/chunk position will always produce
    the same ID.
    """
    raw_id = f"{document_id}:{page_number}:{chunk_index}"
    digest = sha256(raw_id.encode("utf-8")).hexdigest()

    return f"chunk-{digest[:16]}"


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """
    Split a paragraph into sentence-like units.

    This is intentionally lightweight and deterministic rather than
    relying on an external NLP model.
    """
    text = text.strip()

    if not text:
        return []

    sentences = _SENTENCE_PATTERN.split(text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def _split_into_units(text: str) -> list[str]:
    """
    Split document text into paragraph/sentence-aware units.

    Paragraphs are preferred as the primary semantic boundary.
    Long paragraphs are further divided into sentences.
    """
    text = _normalize_text(text)

    if not text:
        return []

    paragraphs = _PARAGRAPH_PATTERN.split(text)

    units: list[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        sentences = _split_sentences(paragraph)

        if sentences:
            units.extend(sentences)
        else:
            units.append(paragraph)

    return units


def _build_chunk_texts(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """
    Build chunks using sentence-aware boundaries.

    Returns:
        list of (chunk_text, start_offset, end_offset)

    The offsets refer to the normalized text representation.
    """
    normalized_text = _normalize_text(text)

    if not normalized_text:
        return []

    units = _split_into_units(normalized_text)

    if not units:
        return []

    chunks: list[tuple[str, int, int]] = []

    current_units: list[str] = []
    current_length = 0
    current_start = 0

    cursor = 0

    # Locate each semantic unit in the normalized text.
    located_units: list[tuple[str, int, int]] = []

    for unit in units:
        position = normalized_text.find(unit, cursor)

        if position == -1:
            position = cursor

        end_position = position + len(unit)

        located_units.append(
            (unit, position, end_position)
        )

        cursor = end_position

    def flush_current() -> None:
        nonlocal current_units
        nonlocal current_length
        nonlocal current_start

        if not current_units:
            return

        chunk_text = " ".join(current_units).strip()

        if chunk_text:
            chunk_end = current_start + len(chunk_text)

            chunks.append(
                (
                    chunk_text,
                    current_start,
                    chunk_end,
                )
            )

        current_units = []
        current_length = 0

    for unit, unit_start, unit_end in located_units:
        unit_length = len(unit)

        # A single sentence larger than chunk_size cannot be split
        # semantically, so preserve it as one chunk.
        if unit_length > chunk_size:
            if current_units:
                flush_current()

            chunks.append(
                (
                    unit,
                    unit_start,
                    unit_end,
                )
            )

            current_start = unit_end
            continue

        separator_length = 1 if current_units else 0

        projected_length = (
            current_length
            + separator_length
            + unit_length
        )

        if current_units and projected_length > chunk_size:
            flush_current()

            # Add semantic overlap using complete previous sentences.
            overlap_units: list[str] = []
            overlap_length = 0

            for previous in reversed(
                units[:len(overlap_units)]
            ):
                previous_length = len(previous)

                extra_separator = (
                    1 if overlap_units else 0
                )

                if (
                    overlap_length
                    + extra_separator
                    + previous_length
                    > chunk_overlap
                ):
                    break

                overlap_units.insert(0, previous)

                overlap_length += (
                    extra_separator + previous_length
                )

            current_units = overlap_units.copy()

            if current_units:
                current_length = len(
                    " ".join(current_units)
                )

                current_start = max(
                    0,
                    unit_start - current_length,
                )
            else:
                current_length = 0
                current_start = unit_start

        if not current_units:
            current_start = unit_start

        current_units.append(unit)

        current_length = len(
            " ".join(current_units)
        )

    flush_current()

    return chunks


def _build_chunks_with_fallback(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """
    Build semantic chunks.

    If semantic splitting produces an unexpected result, fall back
    to deterministic character-based chunking.
    """
    chunks = _build_chunk_texts(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if chunks:
        return chunks

    normalized_text = _normalize_text(text)

    fallback: list[tuple[str, int, int]] = []

    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk_text = normalized_text[start:end].strip()

        if chunk_text:
            fallback.append(
                (
                    chunk_text,
                    start,
                    end,
                )
            )

        if end >= text_length:
            break

        start = end - chunk_overlap

    return fallback


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[dict]:
    """
    Convert loaded documents/pages into semantic chunks.

    Chunking strategy:
        paragraph → sentence → size-aware grouping

    The public API remains compatible with the original implementation.

    Args:
        documents:
            Documents returned by the PDF loader.

        chunk_size:
            Target maximum chunk size in characters.

        chunk_overlap:
            Approximate overlap between neighboring chunks.

    Returns:
        List of chunk dictionaries containing text and metadata.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap must be non-negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks: list[dict] = []

    for document in documents:
        text = document.get("text", "").strip()

        if not text:
            continue

        metadata = document.get("metadata", {}).copy()

        document_id = metadata.get(
            "document_id",
            "unknown-document",
        )

        page_number = metadata.get(
            "page_number",
            metadata.get("page", 1),
        )

        semantic_chunks = _build_chunks_with_fallback(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        page_chunks: list[dict] = []

        for chunk_index, (
            chunk_text,
            chunk_start,
            chunk_end,
        ) in enumerate(semantic_chunks):

            chunk_id = _build_chunk_id(
                document_id=document_id,
                page_number=page_number,
                chunk_index=chunk_index,
            )

            chunk_metadata = metadata.copy()

            chunk_metadata["document_id"] = document_id
            chunk_metadata["chunk_id"] = chunk_id
            chunk_metadata["chunk_index"] = chunk_index

            chunk_metadata["page"] = page_number
            chunk_metadata["page_number"] = page_number

            chunk_metadata["chunk_start"] = chunk_start
            chunk_metadata["chunk_end"] = chunk_end

            chunk_metadata["character_count"] = len(
                chunk_text
            )

            chunk_metadata["chunking_strategy"] = (
                "sentence_aware"
            )

            page_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

        chunk_count = len(page_chunks)

        for chunk in page_chunks:
            chunk["metadata"]["chunk_count"] = chunk_count
            chunks.append(chunk)

    return chunks