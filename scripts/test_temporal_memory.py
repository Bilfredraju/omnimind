import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from memory.semantic_store import SemanticMemoryStore


def main():
    print("=" * 60)
    print("OMNIMIND TEMPORAL MEMORY TEST")
    print("=" * 60)

    store = SemanticMemoryStore()

    store.clear()

    now = datetime(
        2026,
        9,
        5,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    # --------------------------------------------------
    # Historical decision
    # --------------------------------------------------

    old_date = now - timedelta(
        days=90
    )

    store.add(
        text=(
            "Decision: I decided to use "
            "Qdrant as the vector database "
            "for my OmniMind project."
        ),
        metadata={
            "type": "decision",
            "importance": 0.9,
            "source": "conversation",
            "created_at": old_date.isoformat(),
        },
    )

    # --------------------------------------------------
    # Recent decision
    # --------------------------------------------------

    recent_date = now - timedelta(
        days=2
    )

    store.add(
        text=(
            "Decision: I decided to use "
            "PostgreSQL for application data."
        ),
        metadata={
            "type": "decision",
            "importance": 0.9,
            "source": "conversation",
            "created_at": recent_date.isoformat(),
        },
    )

    # --------------------------------------------------
    # Query historical memory
    # --------------------------------------------------

    query = (
        "What did I decide about my "
        "project 3 months ago?"
    )

    print()
    print(f"Query: {query}")

    results = store.search(
        query=query,
        top_k=5,
        min_score=0.0,
        now=now,
    )

    print()
    print(f"Results found: {len(results)}")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"Result {index}")
        print(
            f"Score: {result['score']:.4f}"
        )
        print(
            f"Temporal filter: "
            f"{result['temporal_filter']}"
        )
        print(
            f"Created: "
            f"{result['metadata']['created_at']}"
        )
        print(
            f"Text: {result['text']}"
        )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert len(results) >= 1

    combined_text = " ".join(
        result["text"].lower()
        for result in results
    )

    assert "qdrant" in combined_text

    # Recent PostgreSQL decision should NOT be
    # included in the 3-month historical window.
    assert "postgresql" not in combined_text

    # Temporal metadata should be returned.
    assert all(
        result["temporal_filter"]
        == "3 months ago"
        for result in results
    )

    store.clear()

    print()
    print("=" * 60)
    print("TEMPORAL MEMORY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()