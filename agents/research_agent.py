from agents.state import AgentState


class ResearchAgent:
    """
    Agent responsible for external research.

    The first version defines the research interface and
    structured result format. External search tools will be
    connected through the tool/MCP layer.
    """

    def run(self, state: AgentState) -> AgentState:
        query = state["query"]

        # Temporary research result.
        # This will be replaced by the actual web/MCP
        # research tool in the next implementation.
        research_results = [
            {
                "title": "Research pending",
                "url": "",
                "snippet": (
                    f"External research requested for: "
                    f"{query}"
                ),
                "source": "research_agent",
            }
        ]

        return {
            **state,
            "research_results": research_results,
            "current_step": "research_complete",
        }