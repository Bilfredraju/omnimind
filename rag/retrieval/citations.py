from __future__ import annotations

from typing import Any


def _metadata(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata")

    if isinstance(metadata, dict):
        return metadata

    return {}


def get_source(result: dict[str, Any]) -> str:
    metadata = _metadata(result)

    return str(
        result.get("source")
        or metadata.get("source")
        or metadata.get("file_path")
        or "unknown source"
    )


def get_document_name(result: dict[str, Any]) -> str:
    metadata = _metadata(result)

    return str(
        result.get("document_name")
        or metadata.get("document_name")
        or "unknown document"
    )


def get_page(result: dict[str, Any]) -> int | None:
    metadata = _metadata(result)

    page = (
        result.get("page")
        if result.get("page") is not None
        else metadata.get("page")
    )

    if page is None:
        page = (
            result.get("page_number")
            if result.get("page_number") is not None
            else metadata.get("page_number")
        )

    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def get_chunk_id(result: dict[str, Any]) -> str | None:
    metadata = _metadata(result)

    chunk_id = (
        result.get("chunk_id")
        or metadata.get("chunk_id")
        or result.get("result_id")
    )

    return str(chunk_id) if chunk_id is not None else None


def build_citation(result: dict[str, Any], citation_index: int) -> dict[str, Any]:
    return {
        "citation_id": f"[{citation_index}]",
        "document_name": get_document_name(result),
        "source": get_source(result),
        "page": get_page(result),
        "chunk_id": get_chunk_id(result),
    }


def build_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []

    seen: set[tuple[str, int | None, str | None]] = set()

    for result in results:
        citation = build_citation(
            result,
            len(citations) + 1,
        )

        identity = (
            citation["source"],
            citation["page"],
            citation["chunk_id"],
        )

        if identity in seen:
            continue

        seen.add(identity)
        citations.append(citation)

    # Re-number after duplicate removal.
    for index, citation in enumerate(citations, start=1):
        citation["citation_id"] = f"[{index}]"

    return citations