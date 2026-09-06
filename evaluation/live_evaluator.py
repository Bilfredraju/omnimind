from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ======================================================================
# PROJECT ROOT
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ======================================================================
# IMPORTS
# ======================================================================

from agents.graph import OmniMindGraph
from evaluation.fixtures import create_evaluation_memory_agent

from evaluation.metrics import (
    answer_contains_required_facts,
    answer_fact_coverage,
    citation_coverage,
    current_memory_accuracy,
    historical_current_separation,
    historical_memory_accuracy,
    keyword_coverage,
)


# ======================================================================
# PATHS
# ======================================================================

DATASET_PATH = (
    ROOT
    / "evaluation"
    / "datasets"
    / "memory_evaluation.json"
)

REPORT_PATH = (
    ROOT
    / "evaluation"
    / "live_evaluation_report.json"
)


# ======================================================================
# DATASET
# ======================================================================

def load_dataset() -> list[dict[str, Any]]:
    """
    Load the OmniMind evaluation dataset.

    Supported formats:

    1. Top-level list:

       [
           {...},
           {...}
       ]

    2. Object containing a 'cases' list:

       {
           "cases": [
               {...},
               {...}
           ]
       }
    """

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        cases = data

    elif isinstance(data, dict):
        cases = data.get("cases")

    else:
        cases = None

    if not isinstance(cases, list):
        raise ValueError(
            "Evaluation dataset must be either a list of cases "
            "or an object containing a 'cases' list."
        )

    for index, case in enumerate(
        cases,
        start=1,
    ):
        if not isinstance(case, dict):
            raise ValueError(
                f"Evaluation case {index} is not an object."
            )

    return cases


# ======================================================================
# PDF DISCOVERY
# ======================================================================

def find_pdf() -> Path | None:
    """
    Find the PDF used for live RAG evaluation.
    """

    search_directories = [
        ROOT / "data" / "documents",
        ROOT / "data" / "raw",
        ROOT / "data",
    ]

    for directory in search_directories:

        if not directory.exists():
            continue

        pdfs = sorted(
            directory.rglob("*.pdf")
        )

        if pdfs:
            return pdfs[0]

    return None


# ======================================================================
# GRAPH STATE
# ======================================================================

def build_state(
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a clean AgentState for one evaluation case.
    """

    return {
        "query": case.get(
            "query",
            "",
        ),
        "plan": [],
        "current_step": "",
        "route": "",
        "planning_memory_context": "",
        "memory_results": [],
        "memory_context": "",
        "memory_written": False,
        "memory_count": 0,
        "temporal_intent": {},
        "temporal_memory_results": [],
        "temporal_memory_context": "",
        "rag_results": [],
        "research_results": [],
        "analysis": "",
        "final_answer": "",
        "sources": [],
        "error": "",
    }


# ======================================================================
# HELPERS
# ======================================================================

def as_list(
    value: Any,
) -> list[Any]:
    """
    Normalize dataset fields to lists.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def safe_metric(
    value: Any,
) -> float:
    """
    Safely convert a metric result to float.
    """

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def expected_document_citations(
    case: dict[str, Any],
) -> int:
    """
    Read expected document citation count from a case.

    Supports several dataset field names for compatibility.
    """

    value = case.get(
        "expected_document_citations",
        case.get(
            "expected_doc_citations",
            0,
        ),
    )

    try:
        return max(
            0,
            int(value),
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0


def expected_web_citations(
    case: dict[str, Any],
) -> int:
    """
    Read expected web citation count from a case.
    """

    value = case.get(
        "expected_web_citations",
        0,
    )

    try:
        return max(
            0,
            int(value),
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0


# ======================================================================
# CASE EVALUATION
# ======================================================================

def evaluate_case(
    case: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate a live OmniMind graph result using the actual
    evaluation.metrics API.
    """

    answer = result.get(
        "final_answer",
        "",
    ) or ""

    # --------------------------------------------------------------
    # Expected values
    # --------------------------------------------------------------

    required_facts = as_list(
        case.get(
            "expected_facts",
            [],
        )
    )

    expected_keywords = as_list(
        case.get(
            "expected_keywords",
            [],
        )
    )

    historical_facts = as_list(
        case.get(
            "expected_historical",
            [],
        )
    )

    current_facts = as_list(
        case.get(
            "expected_current",
            [],
        )
    )

    # --------------------------------------------------------------
    # Answer correctness
    # --------------------------------------------------------------

    fact_score = answer_fact_coverage(
        answer,
        required_facts,
    )

    required_facts_pass = answer_contains_required_facts(
        answer,
        required_facts,
    )

    keyword_score = keyword_coverage(
        answer,
        expected_keywords,
    )

    # --------------------------------------------------------------
    # Temporal accuracy
    # --------------------------------------------------------------

    historical_score = historical_memory_accuracy(
        answer,
        historical_facts,
    )

    current_score = current_memory_accuracy(
        answer,
        current_facts,
    )

    separation_score = historical_current_separation(
        answer,
        historical_facts,
        current_facts,
    )

    # --------------------------------------------------------------
    # Citation coverage
    # --------------------------------------------------------------

    document_citations = expected_document_citations(
        case
    )

    web_citations = expected_web_citations(
        case
    )

    citation_score = citation_coverage(
        answer,
        document_citations,
        web_citations,
    )

    # --------------------------------------------------------------
    # Return detailed evaluation record
    # --------------------------------------------------------------

    return {
        "id": case.get(
            "id",
            "",
        ),
        "category": case.get(
            "category",
            "",
        ),
        "query": case.get(
            "query",
            "",
        ),

        # Answer
        "answer": answer,
        "required_facts_pass": required_facts_pass,

        # Metrics
        "fact_score": safe_metric(
            fact_score
        ),
        "keyword_score": safe_metric(
            keyword_score
        ),
        "historical_score": safe_metric(
            historical_score
        ),
        "current_score": safe_metric(
            current_score
        ),
        "separation_score": safe_metric(
            separation_score
        ),
        "citation_score": safe_metric(
            citation_score
        ),

        # Graph
        "route": result.get(
            "route",
            "",
        ),

        "plan": result.get(
            "plan",
            [],
        ),

        # Memory diagnostics
        "semantic_memory_count": len(
            result.get(
                "memory_results",
                [],
            )
        ),

        "temporal_memory_count": len(
            result.get(
                "temporal_memory_results",
                [],
            )
        ),

        "rag_count": len(
            result.get(
                "rag_results",
                [],
            )
        ),

        "research_count": len(
            result.get(
                "research_results",
                [],
            )
        ),

        "source_count": len(
            result.get(
                "sources",
                [],
            )
        ),

        # Temporal information
        "temporal_intent": result.get(
            "temporal_intent",
            {},
        ),

        # Memory-aware planning
        "planning_memory_context": result.get(
            "planning_memory_context",
            "",
        ),

        # Memory contexts
        "memory_context": result.get(
            "memory_context",
            "",
        ),

        "temporal_memory_context": result.get(
            "temporal_memory_context",
            "",
        ),

        # Errors
        "error": result.get(
            "error",
            "",
        ),
    }


# ======================================================================
# ERROR RESULT
# ======================================================================

def build_error_result(
    case: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    """
    Build a consistent evaluation result when a graph case fails.
    """

    return {
        "id": case.get(
            "id",
            "",
        ),
        "category": case.get(
            "category",
            "",
        ),
        "query": case.get(
            "query",
            "",
        ),
        "answer": "",
        "required_facts_pass": False,

        "fact_score": 0.0,
        "keyword_score": 0.0,
        "historical_score": 0.0,
        "current_score": 0.0,
        "separation_score": 0.0,
        "citation_score": 0.0,

        "route": "",
        "plan": [],

        "semantic_memory_count": 0,
        "temporal_memory_count": 0,
        "rag_count": 0,
        "research_count": 0,
        "source_count": 0,

        "temporal_intent": {},
        "planning_memory_context": "",
        "memory_context": "",
        "temporal_memory_context": "",

        "error": (
            f"{type(exc).__name__}: {exc}"
        ),
    }


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------

    dataset = load_dataset()

    if not dataset:
        raise ValueError(
            "Evaluation dataset contains no cases."
        )

    # --------------------------------------------------------------
    # Locate PDF
    # --------------------------------------------------------------

    pdf_path = find_pdf()

    if pdf_path is None:
        raise FileNotFoundError(
            "No PDF found under "
            "data/documents, data/raw, or data."
        )

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print("=" * 72)
    print("OMNIMIND LIVE EVALUATION")
    print("=" * 72)

    print(
        f"Dataset : {DATASET_PATH}"
    )

    print(
        f"PDF     : {pdf_path}"
    )

    print(
        f"Cases   : {len(dataset)}"
    )

    print()

    # --------------------------------------------------------------
    # Create isolated evaluation memory
    # --------------------------------------------------------------

    fixture, memory_agent = (
        create_evaluation_memory_agent()
    )

    print(
        "Evaluation memory initialized"
    )

    print(
        "Semantic memories     : "
        f"{memory_agent.semantic_store.count()}"
    )

    print(
        "Consolidated memories : "
        f"{fixture.consolidated_store.count()}"
    )

    print()

    # --------------------------------------------------------------
    # Create OmniMind graph
    # --------------------------------------------------------------

    graph = OmniMindGraph(
        pdf_path=str(pdf_path),
        memory_agent=memory_agent,
    )

    results: list[dict[str, Any]] = []

    try:

        # ==========================================================
        # CASE LOOP
        # ==========================================================

        for index, case in enumerate(
            dataset,
            start=1,
        ):

            case_id = case.get(
                "id",
                f"case_{index}",
            )

            query = case.get(
                "query",
                "",
            )

            print("-" * 72)

            print(
                f"CASE {index}/{len(dataset)}: "
                f"{case_id}"
            )

            print(
                f"Query: {query}"
            )

            # ------------------------------------------------------
            # Build clean state
            # ------------------------------------------------------

            state = build_state(
                case
            )

            try:

                # --------------------------------------------------
                # Execute real OmniMind graph
                # --------------------------------------------------

                output = graph.run(
                    state
                )

                # --------------------------------------------------
                # Evaluate result
                # --------------------------------------------------

                evaluated = evaluate_case(
                    case,
                    output,
                )

                results.append(
                    evaluated
                )

                # --------------------------------------------------
                # Diagnostics
                # --------------------------------------------------

                print(
                    f"Route: "
                    f"{evaluated['route']}"
                )

                print(
                    "Memory: "
                    f"semantic="
                    f"{evaluated['semantic_memory_count']} "
                    f"temporal="
                    f"{evaluated['temporal_memory_count']}"
                )

                print(
                    "Retrieval: "
                    f"RAG="
                    f"{evaluated['rag_count']} "
                    f"Research="
                    f"{evaluated['research_count']}"
                )

                print(
                    "Scores: "
                    f"facts="
                    f"{evaluated['fact_score']:.2%} "
                    f"keywords="
                    f"{evaluated['keyword_score']:.2%} "
                    f"historical="
                    f"{evaluated['historical_score']:.2%} "
                    f"current="
                    f"{evaluated['current_score']:.2%} "
                    f"separation="
                    f"{evaluated['separation_score']:.2%} "
                    f"citation="
                    f"{evaluated['citation_score']:.2%}"
                )

                if evaluated[
                    "required_facts_pass"
                ]:

                    print(
                        "[PASS]"
                    )

                else:

                    print(
                        "[FAIL]"
                    )

                # --------------------------------------------------
                # Graph error
                # --------------------------------------------------

                if evaluated[
                    "error"
                ]:

                    print(
                        "Graph error: "
                        f"{evaluated['error']}"
                    )

            except Exception as exc:

                # --------------------------------------------------
                # Keep evaluating remaining cases
                # --------------------------------------------------

                print(
                    "[ERROR] "
                    f"{type(exc).__name__}: {exc}"
                )

                results.append(
                    build_error_result(
                        case,
                        exc,
                    )
                )

    finally:

        # --------------------------------------------------------------
        # Release graph resources
        # --------------------------------------------------------------

        graph.close()

    # ==============================================================
    # SUMMARY
    # ==============================================================

    print()

    print("=" * 72)
    print("LIVE EVALUATION SUMMARY")
    print("=" * 72)

    total = len(
        results
    )

    passed = sum(
        1
        for result in results
        if result.get(
            "required_facts_pass",
            False,
        )
    )

    failed = (
        total
        - passed
    )

    print(
        f"Cases evaluated : {total}"
    )

    print(
        f"Cases passed    : {passed}"
    )

    print(
        f"Cases failed    : {failed}"
    )

    print(
        "Pass rate       : "
        f"{(passed / total if total else 0):.2%}"
    )

    # ==============================================================
    # AVERAGE METRICS
    # ==============================================================

    if results:

        avg_fact = sum(
            safe_metric(
                result.get(
                    "fact_score",
                    0.0,
                )
            )
            for result in results
        ) / total

        avg_keyword = sum(
            safe_metric(
                result.get(
                    "keyword_score",
                    0.0,
                )
            )
            for result in results
        ) / total

        avg_historical = sum(
            safe_metric(
                result.get(
                    "historical_score",
                    0.0,
                )
            )
            for result in results
        ) / total

        avg_current = sum(
            safe_metric(
                result.get(
                    "current_score",
                    0.0,
                )
            )
            for result in results
        ) / total

        avg_separation = sum(
            safe_metric(
                result.get(
                    "separation_score",
                    0.0,
                )
            )
            for result in results
        ) / total

        avg_citation = sum(
            safe_metric(
                result.get(
                    "citation_score",
                    0.0,
                )
            )
            for result in results
        ) / total

        print()

        print(
            "Average metrics"
        )

        print(
            f"Fact coverage      : "
            f"{avg_fact:.2%}"
        )

        print(
            f"Keyword coverage   : "
            f"{avg_keyword:.2%}"
        )

        print(
            f"Historical accuracy: "
            f"{avg_historical:.2%}"
        )

        print(
            f"Current accuracy   : "
            f"{avg_current:.2%}"
        )

        print(
            f"Temporal separation: "
            f"{avg_separation:.2%}"
        )

        print(
            f"Citation coverage  : "
            f"{avg_citation:.2%}"
        )

    # ==============================================================
    # MEMORY DIAGNOSTICS
    # ==============================================================

    print()

    print(
        "Memory diagnostics"
    )

    semantic_total = sum(
        result.get(
            "semantic_memory_count",
            0,
        )
        for result in results
    )

    temporal_total = sum(
        result.get(
            "temporal_memory_count",
            0,
        )
        for result in results
    )

    rag_total = sum(
        result.get(
            "rag_count",
            0,
        )
        for result in results
    )

    research_total = sum(
        result.get(
            "research_count",
            0,
        )
        for result in results
    )

    print(
        "Semantic memory retrievals : "
        f"{semantic_total}"
    )

    print(
        "Temporal memory retrievals : "
        f"{temporal_total}"
    )

    print(
        "RAG retrievals             : "
        f"{rag_total}"
    )

    print(
        "Research retrievals        : "
        f"{research_total}"
    )

    # ==============================================================
    # TEMPORAL CASE DIAGNOSTICS
    # ==============================================================

    temporal_cases = [
        result
        for result in results
        if isinstance(
            result.get(
                "temporal_intent",
                {},
            ),
            dict,
        )
        and result.get(
            "temporal_intent",
            {},
        ).get(
            "has_time_filter",
            False,
        )
    ]

    print()

    print(
        "Temporal evaluation cases : "
        f"{len(temporal_cases)}"
    )

    if temporal_cases:

        temporal_hits = sum(
            1
            for result in temporal_cases
            if result.get(
                "temporal_memory_count",
                0,
            ) > 0
        )

        print(
            "Temporal cases with memory "
            "retrievals              : "
            f"{temporal_hits}/"
            f"{len(temporal_cases)}"
        )

    # ==============================================================
    # ERROR DIAGNOSTICS
    # ==============================================================

    error_results = [
        result
        for result in results
        if result.get(
            "error",
            "",
        )
    ]

    print()

    print(
        "Cases with runtime errors  : "
        f"{len(error_results)}"
    )

    if error_results:

        rate_limit_errors = sum(
            1
            for result in error_results
            if "RateLimitError" in str(
                result.get(
                    "error",
                    "",
                )
            )
            or "rate_limit" in str(
                result.get(
                    "error",
                    "",
                )
            ).lower()
        )

        print(
            "Groq rate-limit errors    : "
            f"{rate_limit_errors}"
        )

    # ==============================================================
    # SAVE REPORT
    # ==============================================================

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print()

    print(
        "Detailed report saved to:"
    )

    print(
        REPORT_PATH
    )

    print()

    print("=" * 72)
    print("LIVE EVALUATION COMPLETE")
    print("=" * 72)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()