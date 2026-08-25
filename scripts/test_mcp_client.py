import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_clients.research_client import search_web


print("=" * 60)
print("OMNIMIND MCP CLIENT TEST")
print("=" * 60)


query = (
    "latest developments in "
    "Retrieval-Augmented Generation"
)


print("\nQuery:")
print(query)

print("\nCalling MCP Research Server...")


result = search_web(
    query=query,
    max_results=5,
)


print("\n" + "=" * 60)
print("MCP RESPONSE")
print("=" * 60)


print(
    f"Query: {result.get('query', query)}"
)

print(
    f"Results: "
    f"{len(result.get('results', []))}"
)


for index, item in enumerate(
    result.get("results", []),
    start=1,
):

    print(f"\nResult {index}")

    print(
        f"Title: "
        f"{item.get('title', '')}"
    )

    print(
        f"URL: "
        f"{item.get('url', '')}"
    )

    print(
        f"Snippet: "
        f"{item.get('snippet', '')}"
    )


print("\n" + "=" * 60)
print("MCP CLIENT SUCCESSFUL")
print("=" * 60)