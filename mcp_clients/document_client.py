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
    / "document_server.py"
)


class DocumentMCPClient:
    """
    Client adapter for the OmniMind Document MCP Server.
    """

    def __init__(self):
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[
                str(SERVER_PATH),
            ],
        )

    async def get_document_info(self) -> dict:
        """
        Call the get_document_info MCP tool and normalize
        the response into a Python dictionary.
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
                    "get_document_info",
                    arguments={},
                )

                if result.structured_content:
                    return result.structured_content

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

                return {
                    "collection": "",
                    "document": "",
                    "pages": 0,
                    "chunks": 0,
                    "embedding_model": "",
                    "reranker_model": "",
                    "vector_database": "",
                    "status": "unknown",
                }


def get_document_info() -> dict:
    """
    Synchronous wrapper around the async MCP client.
    """

    return asyncio.run(
        DocumentMCPClient().get_document_info()
    )