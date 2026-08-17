from typing import List


class ContextManager:
    """
    Select and prepare retrieved documents for the LLM context.

    The first version uses a character budget as a simple
    approximation of the model's token budget.
    """

    def __init__(
        self,
        max_context_chars: int = 12000,
    ):
        self.max_context_chars = max_context_chars

    def build_context(
        self,
        documents: List[dict],
    ) -> dict:
        """
        Build a context string from ranked documents.

        Documents are expected to contain:
        - text
        - metadata
        - rerank_score
        """

        selected_documents = []
        current_chars = 0

        for document in documents:
            text = document["text"].strip()

            if not text:
                continue

            # Stop if adding the next document exceeds the budget.
            if (
                current_chars + len(text)
                > self.max_context_chars
            ):
                continue

            selected_documents.append(document)

            current_chars += len(text)

        context_parts = []

        for index, document in enumerate(
            selected_documents,
            start=1,
        ):
            metadata = document["metadata"]

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            chunk_index = metadata.get(
                "chunk_index",
                "Unknown",
            )

            context_parts.append(
                f"""
[Source {index}]
Document: {source}
Page: {page}
Chunk: {chunk_index}

{document["text"]}
""".strip()
            )

        context = "\n\n".join(
            context_parts
        )

        return {
            "context": context,
            "documents": selected_documents,
            "total_documents": len(
                selected_documents
            ),
            "total_characters": len(context),
        }