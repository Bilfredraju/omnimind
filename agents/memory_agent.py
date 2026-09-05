from __future__ import annotations

from datetime import datetime
from typing import Any

from memory.consolidated_store import ConsolidatedMemoryStore
from memory.consolidated_time_parser import ConsolidatedTimeQueryParser
from memory.temporal_memory_pipeline import TemporalMemoryPipeline
from memory.manager import MemoryManager
from memory.semantic_store import SemanticMemoryStore
from memory.extractor import MemoryExtractor


class MemoryAgent:
    """
    OmniMind memory agent.

    Responsibilities:

    1. Recall relevant semantic memories.
    2. Detect temporal intent.
    3. Retrieve temporal/consolidated memories.
    4. Build memory context.
    5. Extract important memories from conversations.
    6. Persist useful memories.

    Supports both:

        memory_agent.recall("query")

    and:

        memory_agent.recall(state)

    Likewise:

        memory_agent.write("query", "answer")

    and:

        memory_agent.write(state)
    """

    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        memory_extractor: MemoryExtractor | None = None,
        semantic_store: SemanticMemoryStore | None = None,
        consolidated_store: ConsolidatedMemoryStore | None = None,
    ):
        # ==============================================================
        # CONVERSATIONAL MEMORY
        # ==============================================================

        self.memory_manager = (
            memory_manager
            if memory_manager is not None
            else MemoryManager()
        )

        # ==============================================================
        # SEMANTIC LONG-TERM MEMORY
        # ==============================================================

        self.semantic_store = (
            semantic_store
            if semantic_store is not None
            else SemanticMemoryStore()
        )

        # ==============================================================
        # MEMORY EXTRACTION
        # ==============================================================

        self.memory_extractor = (
            memory_extractor
            if memory_extractor is not None
            else MemoryExtractor()
        )

        # ==============================================================
        # CONSOLIDATED MEMORY
        # ==============================================================

        self.consolidated_store = (
            consolidated_store
            if consolidated_store is not None
            else ConsolidatedMemoryStore()
        )

        # ==============================================================
        # TEMPORAL MEMORY
        # ==============================================================

        self.time_parser = ConsolidatedTimeQueryParser()

        self.temporal_pipeline = TemporalMemoryPipeline(
            store=self.consolidated_store,
            time_parser=self.time_parser,
        )

    # ==================================================================
    # MEMORY RECALL
    # ==================================================================

    def recall(
        self,
        query_or_state: str | dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Recall semantic and temporal memories.

        Supports:

            memory_agent.recall("What did I decide?")

        and:

            memory_agent.recall(state)
        """

        # --------------------------------------------------------------
        # Normalize query
        # --------------------------------------------------------------

        if isinstance(query_or_state, dict):
            query = (
                query_or_state.get("query")
                or query_or_state.get("user_message")
                or ""
            )
        else:
            query = query_or_state or ""

        if not isinstance(query, str):
            query = str(query)

        # --------------------------------------------------------------
        # Empty query
        # --------------------------------------------------------------

        if not query.strip():
            return {
                "memory_results": [],
                "memory_context": "",
                "temporal_intent": {},
                "temporal_memory_results": [],
                "temporal_memory_context": "",
            }

        # ==============================================================
        # SEMANTIC LONG-TERM MEMORY
        # ==============================================================

        semantic_results: list[dict[str, Any]] = []

        try:
            semantic_results = self.semantic_store.search(
                query=query,
                top_k=top_k,
            )

        except TypeError:
            # Compatibility with positional search signatures.
            try:
                semantic_results = self.semantic_store.search(
                    query,
                    top_k,
                )

            except Exception as exc:
                print(
                    f"Warning: semantic memory search failed: {exc}"
                )
                semantic_results = []

        except Exception as exc:
            print(
                f"Warning: semantic memory search failed: {exc}"
            )
            semantic_results = []

        if semantic_results is None:
            semantic_results = []

        semantic_context = self._build_semantic_context(
            semantic_results
        )

        # ==============================================================
        # TEMPORAL MEMORY
        # ==============================================================

        temporal_results: list[dict[str, Any]] = []
        temporal_context = ""
        temporal_intent: dict[str, Any] = {}

        try:
            temporal_response = self.recall_temporal(
                query=query,
                top_k=top_k,
            )

            temporal_intent = temporal_response.get(
                "temporal_intent",
                {},
            )

            temporal_results = temporal_response.get(
                "temporal_memory_results",
                [],
            )

            temporal_context = temporal_response.get(
                "temporal_memory_context",
                "",
            )

        except Exception as exc:
            print(
                f"Warning: temporal memory recall failed: {exc}"
            )

        return {
            "memory_results": semantic_results,
            "memory_context": semantic_context,
            "temporal_intent": temporal_intent,
            "temporal_memory_results": temporal_results,
            "temporal_memory_context": temporal_context,
        }

    # ==================================================================
    # TEMPORAL MEMORY RECALL
    # ==================================================================

    def recall_temporal(
        self,
        query: str,
        top_k: int = 5,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve consolidated memories using temporal intent.

        Example:

            "What did I decide about my project 3 months ago?"
        """

        if not isinstance(query, str):
            return {
                "temporal_intent": {},
                "temporal_memory_results": [],
                "temporal_memory_context": "",
            }

        if not query.strip():
            return {
                "temporal_intent": {},
                "temporal_memory_results": [],
                "temporal_memory_context": "",
            }

        # ==============================================================
        # TEMPORAL INTENT
        # ==============================================================

        temporal_intent: dict[str, Any] = {}

        try:
            if now is not None:
                temporal_intent = self.time_parser.parse(
                    query,
                    now=now,
                )
            else:
                temporal_intent = self.time_parser.parse(
                    query
                )

        except TypeError:
            # Compatibility with parsers that do not accept `now`.
            try:
                temporal_intent = self.time_parser.parse(
                    query
                )

            except Exception as exc:
                print(
                    f"Warning: temporal parsing failed: {exc}"
                )

        except Exception as exc:
            print(
                f"Warning: temporal parsing failed: {exc}"
            )

        # ==============================================================
        # TEMPORAL PIPELINE
        # ==============================================================

        temporal_results: list[dict[str, Any]] = []
        temporal_context = ""

        try:
            # TemporalMemoryPipeline exposes `query()`.
            #
            # It accepts:
            #     query
            #     now
            #     top_k
            #
            # and returns:
            #     {
            #         "query": ...,
            #         "temporal_intent": ...,
            #         "results": ...,
            #         "context": ...
            #     }

            result = self.temporal_pipeline.query(
                query=query,
                top_k=top_k,
                now=now,
            )

            if isinstance(result, dict):
                temporal_results = result.get(
                    "results",
                    result.get(
                        "temporal_memory_results",
                        [],
                    ),
                )

                temporal_context = result.get(
                    "context",
                    result.get(
                        "temporal_memory_context",
                        "",
                    ),
                )

            elif isinstance(result, list):
                temporal_results = result

        except TypeError:
            # Compatibility with implementations that do not
            # accept the `now` parameter.
            try:
                result = self.temporal_pipeline.query(
                    query=query,
                    top_k=top_k,
                )

                if isinstance(result, dict):
                    temporal_results = result.get(
                        "results",
                        result.get(
                            "temporal_memory_results",
                            [],
                        ),
                    )

                    temporal_context = result.get(
                        "context",
                        result.get(
                            "temporal_memory_context",
                            "",
                        ),
                    )

                elif isinstance(result, list):
                    temporal_results = result

            except Exception as exc:
                print(
                    f"Warning: temporal retrieval failed: {exc}"
                )

        except Exception as exc:
            print(
                f"Warning: temporal retrieval failed: {exc}"
            )

        if temporal_results is None:
            temporal_results = []

        return {
            "temporal_intent": temporal_intent,
            "temporal_memory_results": temporal_results,
            "temporal_memory_context": temporal_context,
        }

    # ==================================================================
    # MEMORY WRITING
    # ==================================================================

    def write(
        self,
        state_or_query: dict[str, Any] | str,
        answer: str | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Persist useful conversation information.

        Supports:

            memory_agent.write(state)

        and:

            memory_agent.write(
                query,
                answer,
                sources,
            )
        """

        # ==============================================================
        # NORMALIZE INPUT
        # ==============================================================

        if isinstance(state_or_query, dict):

            state = state_or_query

            query = (
                state.get("query")
                or state.get("user_message")
                or ""
            )

            answer = (
                state.get("final_answer")
                or state.get("answer")
                or ""
            )

            if sources is None:
                sources = state.get(
                    "sources",
                    [],
                )

        else:

            query = state_or_query or ""
            answer = answer or ""

        sources = sources or []

        # ==============================================================
        # EMPTY INPUT
        # ==============================================================

        if not query and not answer:
            return {
                "memory_written": False,
                "memory_count": 0,
                "memories": [],
            }

        # ==============================================================
        # STORE CONVERSATION
        # ==============================================================

        conversation_added = False

        try:
            self.memory_manager.remember(
                user_message=query,
                assistant_message=answer,
            )

            conversation_added = True

        except Exception as exc:
            print(
                f"Warning: conversation memory write failed: {exc}"
            )

        # ==============================================================
        # EXTRACT IMPORTANT MEMORIES
        # ==============================================================

        extracted_memories: list[dict[str, Any]] = []

        try:
            # MemoryExtractor.extract() expects:
            #
            #     user_message
            #     assistant_message
            #
            # It does NOT accept sources.

            extracted_memories = (
                self.memory_extractor.extract(
                    user_message=query,
                    assistant_message=answer,
                )
            )

        except TypeError:
            # Positional compatibility.
            try:
                extracted_memories = (
                    self.memory_extractor.extract(
                        query,
                        answer,
                    )
                )

            except Exception as exc:
                print(
                    f"Warning: memory extraction failed: {exc}"
                )

        except Exception as exc:
            print(
                f"Warning: memory extraction failed: {exc}"
            )

        if extracted_memories is None:
            extracted_memories = []

        # ==============================================================
        # PERSIST SEMANTIC MEMORIES
        # ==============================================================

        persisted_memories: list[dict[str, Any]] = []

        for memory in extracted_memories:

            if not isinstance(memory, dict):
                continue

            text = memory.get(
                "text",
                "",
            )

            metadata = memory.get(
                "metadata",
                {},
            )

            if not text:
                continue

            if not isinstance(metadata, dict):
                metadata = {}

            try:
                # SemanticMemoryStore.add() requires:
                #
                #     add(text, metadata)
                #
                # The extractor returns exactly this structure.

                result = self.semantic_store.add(
                    text=text,
                    metadata=metadata,
                )

                persisted_memories.append(result)

            except TypeError:
                # Positional compatibility.
                try:
                    result = self.semantic_store.add(
                        text,
                        metadata,
                    )

                    persisted_memories.append(result)

                except Exception as exc:
                    print(
                        "Warning: semantic memory persistence "
                        f"failed: {exc}"
                    )

            except Exception as exc:
                print(
                    "Warning: semantic memory persistence "
                    f"failed: {exc}"
                )

        # ==============================================================
        # MEMORY WRITTEN
        # ==============================================================

        # `memory_written` indicates that something was actually
        # persisted. Conversation storage counts as a successful write,
        # while extracted semantic memories are reported separately.

        return {
            "memory_written": (
                conversation_added
                or len(persisted_memories) > 0
            ),
            "memory_count": len(persisted_memories),
            "memories": persisted_memories,
        }

    # ==================================================================
    # COMBINED CONTEXT
    # ==================================================================

    def build_combined_context(
        self,
        recall_result: dict[str, Any],
    ) -> str:
        """
        Combine semantic and temporal memory contexts into one
        context block for downstream reasoning.
        """

        if not recall_result:
            return ""

        semantic_context = recall_result.get(
            "memory_context",
            "",
        )

        temporal_context = recall_result.get(
            "temporal_memory_context",
            "",
        )

        sections: list[str] = []

        if semantic_context and semantic_context.strip():
            sections.append(
                "SEMANTIC MEMORY:\n"
                + semantic_context.strip()
            )

        if temporal_context and temporal_context.strip():
            sections.append(
                "TEMPORAL MEMORY:\n"
                + temporal_context.strip()
            )

        return "\n\n".join(sections)

    # ==================================================================
    # SEMANTIC CONTEXT
    # ==================================================================

    def _build_semantic_context(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        """
        Convert semantic-memory results into readable context.
        """

        if not results:
            return ""

        lines: list[str] = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            if not isinstance(result, dict):
                continue

            text = (
                result.get("text")
                or result.get("content")
                or result.get("memory")
                or ""
            )

            if not text:
                continue

            score = result.get(
                "ranking_score",
                result.get(
                    "score",
                    "",
                ),
            )

            metadata = result.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            memory_type = (
                result.get("type")
                or result.get("memory_type")
                or metadata.get("type")
                or ""
            )

            created_at = (
                result.get("created_at")
                or metadata.get("created_at")
                or ""
            )

            metadata_parts: list[str] = []

            if memory_type:
                metadata_parts.append(
                    f"type={memory_type}"
                )

            if created_at:
                metadata_parts.append(
                    f"created={created_at}"
                )

            if score != "":
                try:
                    metadata_parts.append(
                        f"score={float(score):.4f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    metadata_parts.append(
                        f"score={score}"
                    )

            result_metadata = ""

            if metadata_parts:
                result_metadata = (
                    " ["
                    + ", ".join(metadata_parts)
                    + "]"
                )

            lines.append(
                f"{index}. {text}{result_metadata}"
            )

        return "\n".join(lines)

    # ==================================================================
    # CLOSE
    # ==================================================================

    def close(self) -> None:
        """
        Release resources owned by the memory agent.
        """

        try:
            if hasattr(
                self.semantic_store,
                "close",
            ):
                self.semantic_store.close()

        except Exception:
            pass

        try:
            if hasattr(
                self.memory_manager,
                "close",
            ):
                self.memory_manager.close()

        except Exception:
            pass

        try:
            if hasattr(
                self.temporal_pipeline,
                "close",
            ):
                self.temporal_pipeline.close()

        except Exception:
            pass