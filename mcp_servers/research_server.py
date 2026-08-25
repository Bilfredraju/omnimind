from mcp.server import MCPServer


mcp = MCPServer("OmniMind Research Server")


@mcp.tool()
async def search_web(query: str) -> dict:
    """
    Search the web for information related to a query.

    This is the research tool exposed through MCP.
    """

    if not query.strip():
        return {
            "query": query,
            "results": [],
            "error": "Search query cannot be empty.",
        }

    # Temporary implementation.
    # The actual web-search backend will be connected
    # after the MCP tool itself is verified.

    return {
        "query": query,
        "results": [
            {
                "title": "Research tool ready",
                "url": "",
                "snippet": (
                    f"MCP research tool received query: "
                    f"{query}"
                ),
            }
        ],
    }


if __name__ == "__main__":
    mcp.run()