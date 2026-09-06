from agents.state import AgentState
from mcp_clients.document_client import search_documents


class RAGAgent:
    """
    RAG Agent responsible for document retrieval.

    Document retrieval is delegated to the
    OmniMind Document MCP Server.

    Phase 17.3:
    Retrieval can use relevant semantic and temporal memory
    as additional retrieval context.

    Retrieval pipeline:

        User Query
            ↓
        Relevant Memory
            ↓
        Memory-Aware Retrieval Query
            ↓
        RAG Agent
            ↓
        Document MCP Client
            ↓
        Document MCP Server
            ↓
        Hybrid Retrieval
            ↓
        Cross-Encoder Reranking
            ↓
        Retrieved Evidence
    """

    def __init__(
        self,
        pdf_path: str | None = None,
    ):
        """
        Initialize the RAG Agent.

        pdf_path is retained for compatibility with the
        existing OmniMindGraph interface.

        The actual document retrieval is handled
        by the Document MCP Server.
        """

        self.pdf_path = pdf_path

    # ======================================================
    # MEMORY-AWARE QUERY
    # ======================================================

    def _build_retrieval_query(
        self,
        state: AgentState,
    ) -> str:
        """
        Build a focused document-retrieval query using
        the original user query and relevant memory.

        The original query always remains the primary
        retrieval signal.

        Memory is added only when useful.
        """

        query = state["query"]

        memory_context = state.get(
            "memory_context",
            "",
        ) or ""

        temporal_memory_context = state.get(
            "temporal_memory_context",
            "",
        ) or ""

        # --------------------------------------------------
        # No memory
        # --------------------------------------------------

        if (
            not memory_context.strip()
            and not temporal_memory_context.strip()
        ):
            return query

        sections = [
            f"User query:\n{query}"
        ]

        # --------------------------------------------------
        # Semantic memory
        # --------------------------------------------------

        if memory_context.strip():
            sections.append(
                "Relevant remembered context:\n"
                + memory_context.strip()
            )

        # --------------------------------------------------
        # Temporal memory
        # --------------------------------------------------

        if temporal_memory_context.strip():
            sections.append(
                "Relevant temporal context:\n"
                + temporal_memory_context.strip()
            )

        sections.append(
            "Use the remembered context only to improve "
            "retrieval relevance. Prioritize evidence "
            "directly relevant to the user query."
        )

        return "\n\n".join(sections)

    # ======================================================
    # RAG EXECUTION
    # ======================================================

    def run(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Retrieve relevant document evidence through MCP.

        Phase 17.3:
        The retrieval query may include relevant memory
        context while preserving the original user query.
        """

        query = state["query"]

        retrieval_query = self._build_retrieval_query(
            state
        )

        try:

            # ------------------------------------------------
            # Call Document MCP
            # ------------------------------------------------

            result = search_documents(
                query=retrieval_query,
                top_k=5,
            )

            # ------------------------------------------------
            # Extract results
            # ------------------------------------------------

            rag_results = result.get(
                "results",
                [],
            )

            if not isinstance(
                rag_results,
                list,
            ):
                rag_results = []

            # ------------------------------------------------
            # Return updated state
            # ------------------------------------------------

            return {
                **state,
                "rag_results": rag_results,
                "current_step": "rag_complete",
                "error": result.get(
                    "error",
                    "",
                ),
            }

        except Exception as exc:

            return {
                **state,
                "rag_results": [],
                "current_step": "rag_failed",
                "error": str(exc),
            }

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):
        """
        Compatibility method.

        Document MCP manages its own server-side
        resources for each request, so there is
        no persistent local retriever to close here.
        """

        return None