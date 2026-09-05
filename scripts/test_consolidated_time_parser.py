import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from memory.consolidated_time_parser import (
    ConsolidatedTimeQueryParser,
)


def main():

    print("=" * 60)
    print(
        "OMNIMIND CONSOLIDATED TIME QUERY PARSER TEST"
    )
    print("=" * 60)

    parser = ConsolidatedTimeQueryParser()

    now = datetime(
        2026,
        9,
        5,
        12,
        0,
        tzinfo=timezone.utc,
    )

    # --------------------------------------------------------------
    # Test 1
    # --------------------------------------------------------------

    print(
        "\nTEST 1: Three months ago"
    )

    result = parser.parse(
        "What did I decide about my project 3 months ago?",
        now=now,
    )

    assert result["has_time_filter"] is True

    assert (
        result["expression"]
        == "3 months ago"
    )

    assert result["start"] is not None
    assert result["end"] is not None

    print(
        "Expression:",
        result["expression"],
    )

    print(
        "Start:",
        result["start"],
    )

    print(
        "End:",
        result["end"],
    )

    # --------------------------------------------------------------
    # Test 2
    # --------------------------------------------------------------

    print(
        "\nTEST 2: Last month"
    )

    result = parser.parse(
        "What was my decision last month?",
        now=now,
    )

    assert result["has_time_filter"] is True

    assert (
        result["expression"]
        == "last month"
    )

    print(
        "Last month parsed correctly."
    )

    # --------------------------------------------------------------
    # Test 3
    # --------------------------------------------------------------

    print(
        "\nTEST 3: Last week"
    )

    result = parser.parse(
        "What did I choose last week?",
        now=now,
    )

    assert result["has_time_filter"] is True

    assert (
        result["expression"]
        == "last week"
    )

    print(
        "Last week parsed correctly."
    )

    # --------------------------------------------------------------
    # Test 4
    # --------------------------------------------------------------

    print(
        "\nTEST 4: Current query"
    )

    result = parser.parse(
        "What is my current database decision?",
        now=now,
    )

    assert result["is_current"] is True

    assert (
        result["expression"]
        == "current"
    )

    print(
        "Current intent detected correctly."
    )

    # --------------------------------------------------------------
    # Test 5
    # --------------------------------------------------------------

    print(
        "\nTEST 5: No temporal expression"
    )

    result = parser.parse(
        "What database did I use?",
        now=now,
    )

    assert (
        result["has_time_filter"]
        is False
    )

    assert (
        result["expression"]
        is None
    )

    print(
        "No temporal expression handled correctly."
    )

    # --------------------------------------------------------------
    # Test 6
    # --------------------------------------------------------------

    print(
        "\nTEST 6: Empty query"
    )

    result = parser.parse(
        "",
        now=now,
    )

    assert (
        result["has_time_filter"]
        is False
    )

    assert (
        result["is_current"]
        is False
    )

    print(
        "Empty query handled correctly."
    )

    print(
        "\nCONSOLIDATED TIME QUERY PARSER TEST PASSED"
    )


if __name__ == "__main__":
    main()