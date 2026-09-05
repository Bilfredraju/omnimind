from agents.state import AgentState
from memory.manager import MemoryManager
from memory.semantic_store import SemanticMemoryStore
from memory.extractor import MemoryExtractor


class MemoryAgent:
    """
    Handles long-term conversational memory.

    Responsibilities:
        1. Recall semantically relevant memories.
        2. Extract important memories from conversations.
        3. Persist useful memories.
    """

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.semantic_store = SemanticMemoryStore()
        self.extractor = MemoryExtractor()

    # ==================================================
    # MEMORY RECALL
    # ==================================================

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
                min_score=0.35,
            )

            memory_sections = []

            for index, memory in enumerate(
                results,
                start=1,
            ):

                memory_sections.append(
                    f"""
[Memory {index}]
Type: {
    memory.get("metadata", {}).get(
        "type",
        "unknown",
    )
}
Importance: {
    memory.get("metadata", {}).get(
        "importance",
        0.0,
    )
}
Relevance Score: {memory["score"]:.4f}

{memory["text"]}
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
                "error": (
                    f"Memory recall failed: {exc}"
                ),
                "current_step": "memory_recall_complete",
            }

    # ==================================================
    # MEMORY WRITE
    # ==================================================

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
                "current_step": "memory_write_complete",
            }

        try:

            # ------------------------------------------
            # 1. Always preserve raw conversation history
            # ------------------------------------------

            self.memory_manager.remember(
                user_message=query,
                assistant_message=final_answer,
            )

            # ------------------------------------------
            # 2. Extract important semantic memories
            # ------------------------------------------

            extracted_memories = self.extractor.extract(
                user_message=query,
                assistant_message=final_answer,
            )

            # ------------------------------------------
            # 3. Store only useful semantic memories
            # ------------------------------------------

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
                "error": (
                    f"Memory write failed: {exc}"
                ),
                "current_step": "memory_write_complete",
            }

    def close(self):
        """
        Release resources if required in the future.
        """
        pass