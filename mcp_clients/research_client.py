import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVER_PATH = (
    PROJECT_ROOT
    / "mcp_servers"
    / "research_server.py"
)


class ResearchMCPClient:
    """
    Client adapter for the OmniMind Research MCP Server.

    The client launches the MCP server as a subprocess
    and communicates with it through the MCP stdio transport.
    """

    def __init__(self):
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[
                str(SERVER_PATH),
            ],
        )

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict:
        """
        Call the search_web MCP tool.
        """

        async with stdio_client(
            self.server_params
        ) as (read, write):

            async with ClientSession(
                read,
                write,
            ) as session:

                await session.initialize()

                result = await session.call_tool(
                    "search_web",
                    arguments={
                        "query": query,
                        "max_results": max_results,
                    },
                )

                # MCP 2.0 provides structured_content
                # for structured tool results.
                if result.structured_content:
                    return result.structured_content

                # Fallback for text-only responses.
                if result.content:

                    first_content = result.content[0]

                    if hasattr(
                        first_content,
                        "text",
                    ):
                        return {
                            "query": query,
                            "results": [],
                            "raw_text": first_content.text,
                        }

                return {
                    "query": query,
                    "results": [],
                }


def search_web(
    query: str,
    max_results: int = 5,
) -> dict:
    """
    Synchronous wrapper around the async MCP client.
    """

    return asyncio.run(
        ResearchMCPClient().search(
            query=query,
            max_results=max_results,
        )
    )