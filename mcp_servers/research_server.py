from mcp.server import MCPServer
from ddgs import DDGS


mcp = MCPServer("OmniMind Research Server")


@mcp.tool()
def search_web(
    query: str,
    max_results: int = 5,
) -> dict:
    """
    Search the public web and return relevant search results.

    Args:
        query: The web search query.
        max_results: Maximum number of results to return.
    """

    query = query.strip()

    if not query:
        return {
            "query": query,
            "results": [],
            "error": "Search query cannot be empty.",
        }

    # Keep the number of results under control.
    max_results = max(
        1,
        min(max_results, 10),
    )

    try:
        results = DDGS().text(
            query,
            max_results=max_results,
            backend="duckduckgo",
        )

        formatted_results = []

        for result in results:
            formatted_results.append(
                {
                    "title": result.get(
                        "title",
                        "",
                    ),
                    "url": result.get(
                        "href",
                        result.get(
                            "url",
                            "",
                        ),
                    ),
                    "snippet": result.get(
                        "body",
                        result.get(
                            "description",
                            "",
                        ),
                    ),
                }
            )

        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
        }

    except Exception as exc:
        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": str(exc),
        }


if __name__ == "__main__":
    mcp.run()