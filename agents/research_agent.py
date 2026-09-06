from agents.state import AgentState
from mcp_clients.research_client import search_web


class ResearchAgent:
    """
    Research Agent responsible for external web research.

    Web search is delegated to the OmniMind Research MCP Server.

    Memory-aware retrieval architecture:

        User Query
            ↓
        Clean Research Query
            ↓
        Research MCP Server
            ↓
        Web Evidence
            +
        Relevant Memory Context
            ↓
        Analysis / Synthesis

    Important:
        Memory is available to downstream reasoning, but is NOT
        injected into the external web-search query. This prevents
        remembered context from degrading search quality.
    """

    def _get_memory_context(self, state: AgentState) -> str:
        """
        Collect relevant memory context from the graph state.

        Memory is kept separate from the external research query.
        """

        sections = []

        memory_context = state.get("memory_context", "")
        if memory_context and memory_context.strip():
            sections.append(
                "SEMANTIC MEMORY:\n"
                + memory_context.strip()
            )

        temporal_memory_context = state.get(
            "temporal_memory_context",
            "",
        )

        if (
            temporal_memory_context
            and temporal_memory_context.strip()
        ):
            sections.append(
                "TEMPORAL MEMORY:\n"
                + temporal_memory_context.strip()
            )

        return "\n\n".join(sections)

    def _build_research_query(self, state: AgentState) -> str:
        """
        Build the clean external research query.

        The user's original query remains the primary research
        intent. Memory does not get appended to this query.
        """

        query = state["query"]

        if state.get("route") == "both":
            return (
                "recent developments in "
                "Retrieval-Augmented Generation"
            )

        return query

    def run(self, state: AgentState) -> AgentState:
        research_query = self._build_research_query(state)

        memory_context = self._get_memory_context(state)

        try:
            result = search_web(
                query=research_query,
                max_results=5,
            )

            research_results = result.get(
                "results",
                [],
            )

            if not isinstance(research_results, list):
                research_results = []

            sources = list(
                state.get("sources", [])
            )

            for item in research_results:
                sources.append(
                    {
                        "source": item.get(
                            "title",
                            "Web Search",
                        ),
                        "url": item.get(
                            "url",
                            "",
                        ),
                        "snippet": item.get(
                            "snippet",
                            "",
                        ),
                        "type": "web",
                    }
                )

            return {
                **state,
                "research_results": research_results,
                "sources": sources,
                "current_step": "research_complete",
                "research_memory_context": memory_context,
            }

        except Exception as exc:
            return {
                **state,
                "research_results": [],
                "sources": list(
                    state.get("sources", [])
                ),
                "current_step": "research_failed",
                "error": str(exc),
                "research_memory_context": memory_context,
            }