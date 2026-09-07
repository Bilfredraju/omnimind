"""
OmniMind Phase 19.8.5
Automated RAG Evaluation Report

Runs the deterministic unified RAG evaluator and produces:
- Human-readable console report
- Machine-readable JSON report
- Pass/fail threshold validation

No Groq/LLM calls are used by this reporting layer.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Existing deterministic evaluator
# ---------------------------------------------------------------------------

from evaluation.citation_evaluator import evaluate_real_retrieval


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"


DEFAULT_THRESHOLDS = {
    "citation_precision": 0.80,
    "citation_recall": 0.60,
    "citation_correctness": 0.80,
    "citation_f1": 0.70,
    "evidence_utilization": 0.30,
    "answer_grounding": 0.70,
    "citation_evidence_alignment": 0.80,
    "grounding_score": 0.70,
    "supported_claim_ratio": 0.80,
    "unsupported_claim_ratio": 0.20,
    "average_claim_support": 0.70,
    "claim_grounding_score": 0.75,
    "unified_rag_quality": 0.75,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metrics from the actual Phase 19.8.4 evaluator structure.

    Expected evaluator structure:

        {
            "citation": {...},
            "grounding": {...},
            "claim_grounding": {...},
            "unified_rag_quality": float
        }
    """

    citation = result.get("citation", {}) or {}
    grounding = result.get("grounding", {}) or {}
    claim_grounding = result.get("claim_grounding", {}) or {}

    unified_score = result.get(
        "unified_rag_quality",
        0.0,
    )

    return {
        # Citation quality
        "citation_precision": _safe_float(
            citation.get("citation_precision")
        ),
        "citation_recall": _safe_float(
            citation.get("citation_recall")
        ),
        "citation_correctness": _safe_float(
            citation.get("citation_correctness")
        ),
        "citation_completeness": _safe_float(
            citation.get("citation_completeness")
        ),
        "citation_f1": _safe_float(
            citation.get("citation_f1")
        ),

        # Grounding quality
        "evidence_utilization": _safe_float(
            grounding.get("evidence_utilization")
        ),
        "answer_grounding": _safe_float(
            grounding.get("answer_grounding")
        ),
        "citation_evidence_alignment": _safe_float(
            grounding.get("citation_evidence_alignment")
        ),
        "grounding_score": _safe_float(
            grounding.get("grounding_score")
        ),

        # Claim-level grounding
        "supported_claim_ratio": _safe_float(
            claim_grounding.get("supported_claim_ratio")
        ),
        "unsupported_claim_ratio": _safe_float(
            claim_grounding.get("unsupported_claim_ratio")
        ),
        "average_claim_support": _safe_float(
            claim_grounding.get("average_claim_support")
        ),
        "claim_grounding_score": _safe_float(
            claim_grounding.get("claim_grounding_score")
        ),

        # Unified score
        "unified_rag_quality": _safe_float(
            unified_score
        ),
    }


def _evaluate_thresholds(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate configured quality thresholds.

    Normal metrics:
        actual >= threshold

    Unsupported claim ratio:
        actual <= threshold
    """

    results: Dict[str, Dict[str, Any]] = {}

    for metric_name, threshold in thresholds.items():

        actual = metrics.get(metric_name, 0.0)

        if metric_name == "unsupported_claim_ratio":
            passed = actual <= threshold
            operator = "<="
        else:
            passed = actual >= threshold
            operator = ">="

        results[metric_name] = {
            "actual": round(actual, 6),
            "threshold": round(float(threshold), 6),
            "operator": operator,
            "passed": bool(passed),
        }

    return results


def _overall_status(
    threshold_results: Dict[str, Dict[str, Any]],
) -> str:
    """Return PASS when every configured threshold passes."""

    if not threshold_results:
        return "UNKNOWN"

    return (
        "PASS"
        if all(
            item["passed"]
            for item in threshold_results.values()
        )
        else "FAIL"
    )


def _build_report(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the complete machine-readable report."""

    metrics = _extract_metrics(result)

    threshold_results = _evaluate_thresholds(
        metrics,
        DEFAULT_THRESHOLDS,
    )

    status = _overall_status(
        threshold_results
    )

    retrieval = result.get(
        "retrieval",
        {},
    ) or {}

    citation = result.get(
        "citation",
        {},
    ) or {}

    claim_grounding = result.get(
        "claim_grounding",
        {},
    ) or {}

    report = {
        "report_version": "19.8.5",
        "project": "OmniMind",
        "evaluation_type": "deterministic_rag_quality",

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "query": result.get(
            "query",
            "",
        ),

        "status": status,

        # ---------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------

        "summary": {
            "unified_rag_quality": round(
                metrics["unified_rag_quality"],
                6,
            ),
            "citation_f1": round(
                metrics["citation_f1"],
                6,
            ),
            "grounding_score": round(
                metrics["grounding_score"],
                6,
            ),
            "claim_grounding_score": round(
                metrics["claim_grounding_score"],
                6,
            ),
        },

        # ---------------------------------------------------------------
        # Complete metrics
        # ---------------------------------------------------------------

        "metrics": metrics,

        # ---------------------------------------------------------------
        # Threshold configuration
        # ---------------------------------------------------------------

        "thresholds": DEFAULT_THRESHOLDS,

        "threshold_results": threshold_results,

        # ---------------------------------------------------------------
        # Retrieval
        # ---------------------------------------------------------------

        "retrieval": {
            "result_count": retrieval.get(
                "result_count",
                0,
            ),
            "evidence_count": retrieval.get(
                "evidence_count",
                0,
            ),
            "source_count": retrieval.get(
                "source_count",
                0,
            ),
            "expected_citation_ids": retrieval.get(
                "expected_citation_ids",
                [],
            ),
        },

        # ---------------------------------------------------------------
        # Answer
        # ---------------------------------------------------------------

        "answer": result.get(
            "answer",
            "",
        ),

        # ---------------------------------------------------------------
        # Citation information
        # ---------------------------------------------------------------

        "citations": {
            "coverage": _safe_float(
                citation.get("coverage")
            ),
            "document_citations": result.get(
                "document_citations",
                [],
            ),
            "expected_citation_ids": retrieval.get(
                "expected_citation_ids",
                [],
            ),
        },

        # ---------------------------------------------------------------
        # Claim information
        # ---------------------------------------------------------------

        "claim_grounding_summary": {
            "claim_count": claim_grounding.get(
                "claim_count",
                0,
            ),
            "supported_claims": claim_grounding.get(
                "supported_claims",
                0,
            ),
            "unsupported_claims": claim_grounding.get(
                "unsupported_claims",
                0,
            ),
        },

        # ---------------------------------------------------------------
        # Quality dimensions
        # ---------------------------------------------------------------

        "quality_dimensions": {

            "citation_quality": {
                "precision": metrics[
                    "citation_precision"
                ],
                "recall": metrics[
                    "citation_recall"
                ],
                "correctness": metrics[
                    "citation_correctness"
                ],
                "completeness": metrics[
                    "citation_completeness"
                ],
                "f1": metrics[
                    "citation_f1"
                ],
            },

            "grounding_quality": {
                "evidence_utilization": metrics[
                    "evidence_utilization"
                ],
                "answer_grounding": metrics[
                    "answer_grounding"
                ],
                "citation_evidence_alignment": metrics[
                    "citation_evidence_alignment"
                ],
                "grounding_score": metrics[
                    "grounding_score"
                ],
            },

            "claim_grounding": {
                "claim_count": claim_grounding.get(
                    "claim_count",
                    0,
                ),
                "supported_claims": claim_grounding.get(
                    "supported_claims",
                    0,
                ),
                "unsupported_claims": claim_grounding.get(
                    "unsupported_claims",
                    0,
                ),
                "supported_claim_ratio": metrics[
                    "supported_claim_ratio"
                ],
                "unsupported_claim_ratio": metrics[
                    "unsupported_claim_ratio"
                ],
                "average_claim_support": metrics[
                    "average_claim_support"
                ],
                "claim_grounding_score": metrics[
                    "claim_grounding_score"
                ],
            },
        },
    }

    return report


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_report(
    report: Dict[str, Any],
) -> None:
    """Print a human-readable evaluation report."""

    metrics = report["metrics"]

    print()
    print("=" * 72)
    print(
        "OMNIMIND — PHASE 19.8.5 "
        "RAG EVALUATION REPORT"
    )
    print("=" * 72)

    print()
    print(
        f"Status: {report['status']}"
    )

    print(
        f"Query:  {report['query']}"
    )

    # -----------------------------------------------------------------------
    # Citation
    # -----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("CITATION QUALITY")
    print("-" * 72)

    print(
        f"Precision:       "
        f"{metrics['citation_precision']:.3f}"
    )

    print(
        f"Recall:          "
        f"{metrics['citation_recall']:.3f}"
    )

    print(
        f"Correctness:     "
        f"{metrics['citation_correctness']:.3f}"
    )

    print(
        f"Completeness:    "
        f"{metrics['citation_completeness']:.3f}"
    )

    print(
        f"F1:              "
        f"{metrics['citation_f1']:.3f}"
    )

    # -----------------------------------------------------------------------
    # Grounding
    # -----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("GROUNDING QUALITY")
    print("-" * 72)

    print(
        f"Evidence Usage:  "
        f"{metrics['evidence_utilization']:.3f}"
    )

    print(
        f"Answer Grounding:"
        f"{metrics['answer_grounding']:.3f}"
    )

    print(
        f"Citation Align.: "
        f"{metrics['citation_evidence_alignment']:.3f}"
    )

    print(
        f"Grounding Score: "
        f"{metrics['grounding_score']:.3f}"
    )

    # -----------------------------------------------------------------------
    # Claims
    # -----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("CLAIM-LEVEL GROUNDING")
    print("-" * 72)

    claim_summary = report[
        "claim_grounding_summary"
    ]

    print(
        f"Claims:              "
        f"{claim_summary['claim_count']}"
    )

    print(
        f"Supported Claims:    "
        f"{claim_summary['supported_claims']}"
    )

    print(
        f"Unsupported Claims:  "
        f"{claim_summary['unsupported_claims']}"
    )

    print(
        f"Supported Ratio:     "
        f"{metrics['supported_claim_ratio']:.3f}"
    )

    print(
        f"Unsupported Ratio:   "
        f"{metrics['unsupported_claim_ratio']:.3f}"
    )

    print(
        f"Average Support:     "
        f"{metrics['average_claim_support']:.3f}"
    )

    print(
        f"Claim Grounding:     "
        f"{metrics['claim_grounding_score']:.3f}"
    )

    # -----------------------------------------------------------------------
    # Unified
    # -----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("UNIFIED RAG QUALITY")
    print("-" * 72)

    print(
        f"Unified RAG Score: "
        f"{metrics['unified_rag_quality']:.3f}"
    )

    # -----------------------------------------------------------------------
    # Thresholds
    # -----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("THRESHOLD VALIDATION")
    print("-" * 72)

    for metric_name, item in (
        report["threshold_results"].items()
    ):

        marker = (
            "PASS"
            if item["passed"]
            else "FAIL"
        )

        print(
            f"{marker:<6} "
            f"{metric_name:<32} "
            f"{item['actual']:.3f} "
            f"{item['operator']} "
            f"{item['threshold']:.3f}"
        )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("RETRIEVAL")
    print("-" * 72)

    retrieval = report["retrieval"]

    print(
        f"Results:            "
        f"{retrieval['result_count']}"
    )

    print(
        f"Evidence records:   "
        f"{retrieval['evidence_count']}"
    )

    print(
        f"Source records:     "
        f"{retrieval['source_count']}"
    )

    print()
    print("=" * 72)
    print(
        f"FINAL RESULT: {report['status']}"
    )
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_report() -> Dict[str, Any]:
    """Run deterministic evaluation and save a JSON report."""

    print()
    print(
        "Running deterministic RAG evaluation..."
    )

    print(
        "No Groq/LLM generation is required "
        "for this report."
    )

    result = evaluate_real_retrieval()

    if not isinstance(result, dict):
        raise TypeError(
            "evaluate_real_retrieval() "
            "must return a dictionary."
        )

    report = _build_report(result)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        REPORT_DIR
        / f"rag_evaluation_{timestamp}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_report(report)

    print(
        "JSON report saved to:"
    )

    print(report_path)

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_report()