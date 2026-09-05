from agents.state import AgentState
from memory.manager import MemoryManager
from memory.semantic_store import SemanticMemoryStore
from memory.extractor import MemoryExtractor


class MemoryAgent:
    """
    Handles long-term conversational memory.

    Responsibilities:
        1. Recall semantically relevant memories.
        2. Apply temporal filtering when the query contains
           a time reference.
        3. Extract important memories from conversations.
        4. Persist useful memories.
    """

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.semantic_store = SemanticMemoryStore()
        self.extractor = MemoryExtractor()

    def recall(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state.get(
            "query",
            "",
        ).strip()

        if not query:
            return {
                "memory_results": [],
                "memory_context": "",
                "current_step": "memory_recall_complete",
            }

        try:
            results = self.semantic_store.search(
                query=query,
                top_k=5,
                min_score=0.30,
            )

            memory_sections = []

            for index, memory in enumerate(
                results,
                start=1,
            ):
                metadata = memory.get(
                    "metadata",
                    {},
                )

                memory_id = memory.get(
                    "memory_id",
                    "unknown",
                )

                memory_type = metadata.get(
                    "type",
                    "unknown",
                )

                importance = metadata.get(
                    "importance",
                    0.0,
                )

                created_at = metadata.get(
                    "created_at",
                    "unknown",
                )

                relevance_score = memory.get(
                    "score",
                    0.0,
                )

                ranking_score = memory.get(
                    "ranking_score",
                    0.0,
                )

                temporal_filter = memory.get(
                    "temporal_filter",
                    "none",
                )

                memory_text = memory.get(
                    "text",
                    "",
                )

                memory_sections.append(
                    f"""
[Memory {index}]
Memory ID: {memory_id}
Type: {memory_type}
Importance: {importance}
Created At: {created_at}
Relevance Score: {relevance_score:.4f}
Ranking Score: {ranking_score:.4f}
Temporal Filter: {temporal_filter}

{memory_text}
""".strip()
                )

            memory_context = "\n\n".join(
                memory_sections
            )

            return {
                "memory_results": results,
                "memory_context": memory_context,
                "current_step": "memory_recall_complete",
            }

        except Exception as exc:

            return {
                "memory_results": [],
                "memory_context": "",
                "error": f"Memory recall failed: {exc}",
                "current_step": "memory_recall_complete",
            }

    def write(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state.get(
            "query",
            "",
        ).strip()

        final_answer = state.get(
            "final_answer",
            "",
        ).strip()

        if not query or not final_answer:
            return {
                "memory_written": False,
                "memory_count": 0,
                "current_step": "memory_write_complete",
            }

        try:
            # Preserve complete conversation history.
            self.memory_manager.remember(
                user_message=query,
                assistant_message=final_answer,
            )

            # Extract only useful long-term memories.
            extracted_memories = self.extractor.extract(
                user_message=query,
                assistant_message=final_answer,
            )

            stored_count = 0

            for memory in extracted_memories:
                self.semantic_store.add(
                    text=memory["text"],
                    metadata=memory["metadata"],
                )

                stored_count += 1

            return {
                "memory_written": True,
                "memory_count": stored_count,
                "current_step": "memory_write_complete",
            }

        except Exception as exc:

            return {
                "memory_written": False,
                "memory_count": 0,
                "error": f"Memory write failed: {exc}",
                "current_step": "memory_write_complete",
            }

    def close(self):
        """
        Reserved for future memory resources.
        """
        pass