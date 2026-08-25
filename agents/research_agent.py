from agents.state import AgentState
from mcp_clients.research_client import search_web


class ResearchAgent:
    """
    Research Agent responsible for external web research.

    The agent delegates web searching to the OmniMind
    Research MCP Server through the MCP client adapter.
    """

    def run(self, state: AgentState) -> AgentState:
        query = state["query"]

        try:
            result = search_web(
                query=query,
                max_results=5,
            )

            research_results = result.get(
                "results",
                [],
            )

            # Convert research results into the shared
            # OmniMind source format.
            sources = list(
                state.get(
                    "sources",
                    [],
                )
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
            }

        except Exception as exc:

            return {
                **state,
                "research_results": [],
                "current_step": "research_failed",
                "error": str(exc),
            }