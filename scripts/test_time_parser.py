import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from memory.time_parser import MemoryTimeParser


def main():
    print("=" * 60)
    print("OMNIMIND MEMORY TIME PARSER TEST")
    print("=" * 60)

    parser = MemoryTimeParser()

    fixed_now = datetime(
        2026,
        9,
        5,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    tests = [
        (
            "What did I decide yesterday?",
            "yesterday",
        ),
        (
            "What did we discuss last week?",
            "last week",
        ),
        (
            "What was my goal last month?",
            "last month",
        ),
        (
            "What did I decide 3 months ago?",
            "3 months ago",
        ),
        (
            "What did I work on 2 weeks ago?",
            "2 weeks ago",
        ),
        (
            "What did I decide recently?",
            "recently",
        ),
    ]

    for query, expected_expression in tests:

        result = parser.parse(
            query,
            now=fixed_now,
        )

        print()
        print(f"Query: {query}")
        print(
            f"Expression: {result['expression']}"
        )
        print(
            f"Start: {result['start']}"
        )
        print(
            f"End: {result['end']}"
        )

        assert result["has_time_filter"]
        assert (
            result["expression"]
            == expected_expression
        )
        assert result["start"] is not None
        assert result["end"] is not None
        assert result["start"] < result["end"]

    # Test query without time.
    result = parser.parse(
        "What vector database did I choose?",
        now=fixed_now,
    )

    assert not result["has_time_filter"]
    assert result["start"] is None
    assert result["end"] is None
    assert result["expression"] is None

    print()
    print("=" * 60)
    print("TIME PARSER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()