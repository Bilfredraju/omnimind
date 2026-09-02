from agents.state import AgentState
from mcp_clients.document_client import search_documents


class RAGAgent:
    """
    RAG Agent responsible for document retrieval.

    Document retrieval is delegated to the
    OmniMind Document MCP Server.

    Retrieval pipeline:

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

        The actual document retrieval is now handled
        entirely by the Document MCP Server.
        """

        self.pdf_path = pdf_path

    def run(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Retrieve relevant document evidence through MCP.
        """

        query = state["query"]

        try:

            # ------------------------------------------------
            # Call Document MCP
            # ------------------------------------------------

            result = search_documents(
                query=query,
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

    def close(self):
        """
        Compatibility method.

        Document MCP manages its own server-side
        resources for each request, so there is
        no persistent local retriever to close here.
        """

        return None