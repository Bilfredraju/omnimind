from ddgs import DDGS


print("=" * 60)
print("OMNIMIND WEB SEARCH TEST")
print("=" * 60)


query = "Retrieval Augmented Generation latest research"


print(f"\nQuery:")
print(query)


print("\nSearching...")


results = DDGS().text(
    query,
    max_results=5,
    backend="duckduckgo",
)


print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)


for index, result in enumerate(
    results,
    start=1,
):
    print(f"\nResult {index}")

    print(
        f"Title: "
        f"{result.get('title', '')}"
    )

    print(
        f"URL: "
        f"{result.get('href', result.get('url', ''))}"
    )

    print(
        f"Snippet: "
        f"{result.get('body', result.get('description', ''))}"
    )


print("\n" + "=" * 60)
print("WEB SEARCH SUCCESSFUL")
print("=" * 60)