import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVER_PATH = (
    PROJECT_ROOT
    / "mcp_servers"
    / "document_server.py"
)


# ============================================================
# DOCUMENT MCP CLIENT
# ============================================================

class DocumentMCPClient:
    """
    Client adapter for the OmniMind Document MCP Server.

    Available MCP tools:
        - get_document_info
        - search_documents
    """

    def __init__(self):

        self.server_params = (
            StdioServerParameters(
                command=sys.executable,
                args=[
                    str(SERVER_PATH),
                ],
            )
        )

    # ========================================================
    # INTERNAL MCP CALL
    # ========================================================

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """
        Connect to the Document MCP Server and
        execute the requested tool.
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
                    tool_name,
                    arguments=arguments,
                )

                # ------------------------------------------------
                # Preferred structured MCP response
                # ------------------------------------------------

                if result.structured_content:

                    return result.structured_content

                # ------------------------------------------------
                # Fallback: parse text response
                # ------------------------------------------------

                for content in result.content:

                    if not hasattr(
                        content,
                        "text",
                    ):
                        continue

                    raw_text = (
                        content.text.strip()
                    )

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
                        continue

                # ------------------------------------------------
                # Empty response fallback
                # ------------------------------------------------

                return {
                    "results": [],
                    "count": 0,
                    "error": (
                        "MCP server returned "
                        "an empty response."
                    ),
                }

    # ========================================================
    # DOCUMENT INFORMATION
    # ========================================================

    async def get_document_info(
        self,
    ) -> dict:
        """
        Get metadata about the indexed document.
        """

        return await self._call_tool(
            tool_name="get_document_info",
            arguments={},
        )

    # ========================================================
    # DOCUMENT SEARCH
    # ========================================================

    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict:
        """
        Search the document through the MCP server.
        """

        return await self._call_tool(
            tool_name="search_documents",
            arguments={
                "query": query,
                "top_k": top_k,
            },
        )


# ============================================================
# SYNCHRONOUS HELPERS
# ============================================================

def get_document_info() -> dict:
    """
    Synchronous helper for document metadata.
    """

    return asyncio.run(
        DocumentMCPClient().get_document_info()
    )


def search_documents(
    query: str,
    top_k: int = 5,
) -> dict:
    """
    Synchronous helper for document search.
    """

    return asyncio.run(
        DocumentMCPClient().search_documents(
            query=query,
            top_k=top_k,
        )
    )