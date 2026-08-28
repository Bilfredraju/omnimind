from agents.state import AgentState
from mcp_clients.research_client import search_web


class ResearchAgent:
    """
    Research Agent responsible for external web research.

    Web search is delegated to the OmniMind Research MCP Server.
    """

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state["query"]

        # --------------------------------------------------
        # Create a focused research query.
        # --------------------------------------------------

        research_query = query

        if state.get("route") == "both":
            research_query = (
                "recent developments in "
                "Retrieval-Augmented Generation"
            )

        try:

            result = search_web(
                query=research_query,
                max_results=5,
            )

            research_results = result.get(
                "results",
                [],
            )

            if not isinstance(
                research_results,
                list,
            ):
                research_results = []

            sources = list(
                state.get(
                    "sources",
                    [],
                )
            )

            # --------------------------------------------------
            # Add web sources.
            # --------------------------------------------------

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
            }

        except Exception as exc:

            return {
                **state,
                "research_results": [],
                "sources": list(
                    state.get(
                        "sources",
                        [],
                    )
                ),
                "current_step": "research_failed",
                "error": str(exc),
            }