"""
OmniMind Evaluation Runner

Loads the OmniMind benchmark dataset and evaluates
answers against deterministic expectations.

This module is intentionally separated from the
production agent pipeline so that evaluation logic
remains reproducible and easy to debug.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from evaluation.metrics import (
    answer_fact_coverage,
    historical_current_separation,
    citation_coverage,
    keyword_coverage,
    mean_score,
)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_evaluation.json"
)


# ============================================================
# DATASET LOADING
# ============================================================


def load_dataset(
    path: Path = DATASET_PATH,
) -> dict[str, Any]:
    """
    Load the evaluation dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError(
            "Evaluation dataset must contain a JSON object."
        )

    cases = dataset.get(
        "cases",
        [],
    )

    if not isinstance(cases, list):
        raise ValueError(
            "Evaluation dataset 'cases' must be a list."
        )

    return dataset


# ============================================================
# CASE HELPERS
# ============================================================


def _contains_any(
    text: str,
    values: list[str],
) -> bool:
    """
    Check whether any value occurs in text.
    """

    text_lower = text.lower()

    return any(
        value.lower() in text_lower
        for value in values
    )


def _contains_all(
    text: str,
    values: list[str],
) -> bool:
    """
    Check whether every value occurs in text.
    """

    text_lower = text.lower()

    return all(
        value.lower() in text_lower
        for value in values
    )


def _indicates_insufficient_evidence(
    answer: str,
) -> bool:
    """
    Detect whether an answer appropriately indicates
    that available evidence is insufficient.
    """

    answer_lower = answer.lower()

    indicators = [
        "not enough evidence",
        "insufficient evidence",
        "don't have enough evidence",
        "do not have enough evidence",
        "cannot determine",
        "can't determine",
        "not enough information",
        "insufficient information",
        "cannot answer",
        "can't answer",
    ]

    return _contains_any(
        answer_lower,
        indicators,
    )


def _extract_expected_sources(
    case: dict[str, Any],
) -> tuple[int, int]:
    """
    Extract expected document and web citation counts.
    """

    expected = case.get(
        "expected",
        {},
    )

    document_count = int(
        expected.get(
            "expected_document_citations",
            0,
        )
    )

    web_count = int(
        expected.get(
            "expected_web_citations",
            0,
        )
    )

    return (
        document_count,
        web_count,
    )


# ============================================================
# CASE EVALUATION
# ============================================================


def evaluate_case(
    case: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    """
    Evaluate one benchmark case.

    Returns a dictionary containing:
    - case metadata
    - individual metric scores
    - aggregate score
    """

    expected = case.get(
        "expected",
        {},
    )

    required_facts = expected.get(
        "required_facts",
        [],
    )

    historical_facts = expected.get(
        "historical_facts",
        [],
    )

    current_facts = expected.get(
        "current_facts",
        [],
    )

    irrelevant_facts = expected.get(
        "irrelevant_facts",
        [],
    )

    # --------------------------------------------------------
    # Required fact coverage
    # --------------------------------------------------------

    fact_score = answer_fact_coverage(
        answer,
        required_facts,
    )

    # --------------------------------------------------------
    # Historical accuracy
    # --------------------------------------------------------

    historical_score = 1.0

    if historical_facts:
        historical_score = answer_fact_coverage(
            answer,
            historical_facts,
        )

    # --------------------------------------------------------
    # Current accuracy
    # --------------------------------------------------------

    current_score = 1.0

    if current_facts:
        current_score = answer_fact_coverage(
            answer,
            current_facts,
        )

    # --------------------------------------------------------
    # Historical/current separation
    # --------------------------------------------------------

    separation_score = 1.0

    if historical_facts:
        separation_score = historical_current_separation(
            answer,
            historical_facts,
            current_facts,
        )

    # --------------------------------------------------------
    # Irrelevant memory rejection
    # --------------------------------------------------------

    relevance_score = 1.0

    if irrelevant_facts:

        irrelevant_present = sum(
            1
            for fact in irrelevant_facts
            if fact.lower() in answer.lower()
        )

        if irrelevant_present:
            relevance_score = max(
                0.0,
                1.0
                - (
                    irrelevant_present
                    / len(irrelevant_facts)
                ),
            )

    # --------------------------------------------------------
    # Citation coverage
    # --------------------------------------------------------

    expected_document_citations, expected_web_citations = (
        _extract_expected_sources(case)
    )

    citation_score = citation_coverage(
        answer,
        expected_document_citations=(
            expected_document_citations
        ),
        expected_web_citations=(
            expected_web_citations
        ),
    )

    # --------------------------------------------------------
    # Insufficient evidence
    # --------------------------------------------------------

    insufficient_score = 1.0

    if expected.get(
        "should_indicate_insufficient_evidence",
        False,
    ):
        insufficient_score = (
            1.0
            if _indicates_insufficient_evidence(
                answer
            )
            else 0.0
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    scores = [
        fact_score,
        historical_score,
        current_score,
        separation_score,
        relevance_score,
        citation_score,
        insufficient_score,
    ]

    overall_score = mean_score(
        scores
    )

    return {
        "id": case.get(
            "id",
            "unknown",
        ),
        "category": case.get(
            "category",
            "unknown",
        ),
        "query": case.get(
            "query",
            "",
        ),
        "fact_score": fact_score,
        "historical_score": historical_score,
        "current_score": current_score,
        "separation_score": separation_score,
        "relevance_score": relevance_score,
        "citation_score": citation_score,
        "insufficient_evidence_score": (
            insufficient_score
        ),
        "overall_score": overall_score,
    }


# ============================================================
# CATEGORY AGGREGATION
# ============================================================


def aggregate_categories(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Calculate average score per benchmark category.
    """

    grouped: dict[str, list[float]] = {}

    for result in results:

        category = result[
            "category"
        ]

        grouped.setdefault(
            category,
            [],
        ).append(
            result[
                "overall_score"
            ]
        )

    return {
        category: mean_score(
            scores
        )
        for category, scores in grouped.items()
    }


# ============================================================
# REPORT
# ============================================================


def print_report(
    dataset: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """
    Print a human-readable evaluation report.
    """

    categories = aggregate_categories(
        results
    )

    overall_score = mean_score(
        result[
            "overall_score"
        ]
        for result in results
    )

    print("=" * 60)
    print("OMNIMIND MEMORY EVALUATION")
    print("=" * 60)

    print(
        f"\nDataset: "
        f"{dataset.get('name', 'Unknown')}"
    )

    print(
        f"Version: "
        f"{dataset.get('version', 'Unknown')}"
    )

    print(
        f"\nTotal cases: "
        f"{len(results)}"
    )

    print("\n" + "-" * 60)
    print("CATEGORY SCORES")
    print("-" * 60)

    for category, score in sorted(
        categories.items()
    ):
        print(
            f"{category:<35} "
            f"{score * 100:6.2f}%"
        )

    print("\n" + "-" * 60)
    print("OVERALL SCORE")
    print("-" * 60)

    print(
        f"{overall_score * 100:.2f}%"
    )

    print("\n" + "-" * 60)
    print("CASE RESULTS")
    print("-" * 60)

    for result in results:

        status = (
            "PASS"
            if result["overall_score"] >= 0.80
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{result['id']:<22} "
            f"{result['overall_score'] * 100:6.2f}%"
        )

    print("\n" + "=" * 60)

    if overall_score >= 0.80:
        print(
            "EVALUATION THRESHOLD PASSED"
        )
    else:
        print(
            "EVALUATION THRESHOLD FAILED"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================


def main() -> int:
    """
    Load the dataset and evaluate supplied benchmark answers.

    This initial evaluator uses deterministic placeholder
    answers generated from the expected facts so that the
    scoring layer can be validated independently before
    connecting it to the live OmniMind graph.
    """

    dataset = load_dataset()

    cases = dataset.get(
        "cases",
        [],
    )

    results = []

    for case in cases:

        expected = case.get(
            "expected",
            {},
        )

        required_facts = expected.get(
            "required_facts",
            [],
        )

        historical_facts = expected.get(
            "historical_facts",
            [],
        )

        current_facts = expected.get(
            "current_facts",
            [],
        )

        # ----------------------------------------------------
        # Temporary deterministic answer
        #
        # This is deliberately NOT the OmniMind answer.
        #
        # It allows us to verify the evaluation engine
        # before connecting the live graph.
        # ----------------------------------------------------

        answer_parts = []

        answer_parts.extend(
            required_facts
        )

        answer_parts.extend(
            historical_facts
        )

        answer_parts.extend(
            current_facts
        )

        answer = " ".join(
            str(item)
            for item in answer_parts
        )

        # Add temporal wording where applicable.
        if historical_facts:
            answer = (
                "Historically, "
                + answer
            )

        result = evaluate_case(
            case,
            answer,
        )

        results.append(
            result
        )

    print_report(
        dataset,
        results,
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    raise SystemExit(
        main()
    )