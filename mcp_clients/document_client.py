from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


from mcp import (
    ClientSession,
    StdioServerParameters,
)

from mcp.client.stdio import stdio_client


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

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
        - list_documents
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
                # Preferred structured response
                # ------------------------------------------------

                if result.structured_content:

                    return result.structured_content

                # ------------------------------------------------
                # Text fallback
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
        document_id: str | None = None,
    ) -> dict:

        arguments = {}

        if document_id is not None:
            arguments[
                "document_id"
            ] = document_id

        return await self._call_tool(
            tool_name="get_document_info",
            arguments=arguments,
        )

    # ========================================================
    # LIST DOCUMENTS
    # ========================================================

    async def list_documents(
        self,
        status: str | None = None,
    ) -> dict:

        arguments = {}

        if status is not None:
            arguments[
                "status"
            ] = status

        return await self._call_tool(
            tool_name="list_documents",
            arguments=arguments,
        )

    # ========================================================
    # DOCUMENT SEARCH
    # ========================================================

    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> dict:

        arguments = {
            "query": query,
            "top_k": top_k,
        }

        if document_id is not None:
            arguments[
                "document_id"
            ] = document_id

        return await self._call_tool(
            tool_name="search_documents",
            arguments=arguments,
        )


# ============================================================
# SYNCHRONOUS HELPERS
# ============================================================


def get_document_info(
    document_id: str | None = None,
) -> dict:

    return asyncio.run(
        DocumentMCPClient().get_document_info(
            document_id=document_id
        )
    )


def list_documents(
    status: str | None = None,
) -> dict:

    return asyncio.run(
        DocumentMCPClient().list_documents(
            status=status
        )
    )


def search_documents(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> dict:

    return asyncio.run(
        DocumentMCPClient().search_documents(
            query=query,
            top_k=top_k,
            document_id=document_id,
        )
    )