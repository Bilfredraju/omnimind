import asyncio
import json
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
        Call the search_web MCP tool and normalize
        the MCP response into a Python dictionary.
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

                # --------------------------------------------------
                # Response format 1:
                # MCP provides structured content.
                # --------------------------------------------------

                if result.structured_content:
                    return result.structured_content

                # --------------------------------------------------
                # Response format 2:
                # MCP returns JSON as text content.
                # --------------------------------------------------

                for content in result.content:

                    if not hasattr(
                        content,
                        "text",
                    ):
                        continue

                    raw_text = content.text.strip()

                    if not raw_text:
                        continue

                    try:
                        parsed = json.loads(
                            raw_text
                        )

                        if isinstance(
                            parsed,
                            dict,
                        ):
                            return parsed

                    except json.JSONDecodeError:
                        pass

                # --------------------------------------------------
                # No usable response.
                # --------------------------------------------------

                return {
                    "query": query,
                    "results": [],
                    "count": 0,
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