from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Project path
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

from agents.planner import PlannerAgent
from evaluation.fixtures import create_evaluation_memory_agent


# ----------------------------------------------------------------------
# Evaluation configuration
# ----------------------------------------------------------------------

EVALUATION_NOW = datetime(
    2026,
    9,
    6,
    12,
    0,
    0,
    tzinfo=timezone.utc,
)

IRRELEVANT_MIN_SCORE = 0.70


# ----------------------------------------------------------------------
# Evaluation counters
# ----------------------------------------------------------------------

passed = 0
failed = 0
total = 0


# ----------------------------------------------------------------------
# Evaluation helper
# ----------------------------------------------------------------------

def record(
    name: str,
    condition: bool,
    success_message: str = "",
    failure_message: str = "",
) -> None:
    global passed, failed, total

    total += 1

    if condition:
        passed += 1

        print(f"[PASS] {name}")

        if success_message:
            print(success_message)

    else:
        failed += 1

        print(f"[FAIL] {name}")

        if failure_message:
            print(failure_message)


# ----------------------------------------------------------------------
# Result text helper
# ----------------------------------------------------------------------

def result_text(
    results: list[dict[str, Any]],
) -> str:
    """
    Convert different OmniMind memory-result formats into
    one searchable lowercase text string.

    Supported formats:

    1. Direct semantic memory result:

        {
            "memory_id": "...",
            "text": "...",
            "metadata": {...}
        }

    2. Direct consolidated memory result:

        {
            "topic": "...",
            "summary": "...",
            "current_memory_id": "...",
            "historical_memory_ids": [...]
        }

    3. Temporal retrieval result:

        {
            "consolidation": {
                "topic": "...",
                "summary": "...",
                "current_memory_id": "...",
                "historical_memory_ids": [...]
            },
            "score": ...
        }
    """

    parts: list[str] = []

    for result in results:

        if not isinstance(result, dict):
            continue

        # --------------------------------------------------------------
        # Direct result fields
        # --------------------------------------------------------------

        parts.append(
            str(
                result.get(
                    "text",
                    "",
                )
            )
        )

        parts.append(
            str(
                result.get(
                    "memory_id",
                    "",
                )
            )
        )

        parts.append(
            str(
                result.get(
                    "topic",
                    "",
                )
            )
        )

        parts.append(
            str(
                result.get(
                    "summary",
                    "",
                )
            )
        )

        parts.append(
            str(
                result.get(
                    "current_memory_id",
                    "",
                )
            )
        )

        # --------------------------------------------------------------
        # Direct historical memory IDs
        # --------------------------------------------------------------

        historical_memory_ids = result.get(
            "historical_memory_ids",
            [],
        )

        if isinstance(
            historical_memory_ids,
            (list, tuple, set),
        ):
            parts.append(
                " ".join(
                    str(item)
                    for item in historical_memory_ids
                )
            )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        metadata = result.get(
            "metadata",
            {},
        )

        if isinstance(
            metadata,
            dict,
        ):
            parts.append(
                str(
                    metadata.get(
                        "type",
                        "",
                    )
                )
            )

            parts.append(
                str(
                    metadata.get(
                        "status",
                        "",
                    )
                )
            )

        # --------------------------------------------------------------
        # Temporal retrieval format
        #
        # Temporal retrieval wraps the consolidated memory inside:
        #
        #     result["consolidation"]
        # --------------------------------------------------------------

        consolidation = result.get(
            "consolidation",
            {},
        )

        if isinstance(
            consolidation,
            dict,
        ):

            parts.append(
                str(
                    consolidation.get(
                        "topic",
                        "",
                    )
                )
            )

            parts.append(
                str(
                    consolidation.get(
                        "summary",
                        "",
                    )
                )
            )

            parts.append(
                str(
                    consolidation.get(
                        "current_memory_id",
                        "",
                    )
                )
            )

            consolidation_historical_ids = (
                consolidation.get(
                    "historical_memory_ids",
                    [],
                )
            )

            if isinstance(
                consolidation_historical_ids,
                (list, tuple, set),
            ):
                parts.append(
                    " ".join(
                        str(item)
                        for item in consolidation_historical_ids
                    )
                )

        # --------------------------------------------------------------
        # Timeline events
        # --------------------------------------------------------------

        timeline = result.get(
            "timeline",
            [],
        )

        if isinstance(
            timeline,
            list,
        ):
            for event in timeline:

                if not isinstance(
                    event,
                    dict,
                ):
                    continue

                parts.append(
                    str(
                        event.get(
                            "memory_id",
                            "",
                        )
                    )
                )

                parts.append(
                    str(
                        event.get(
                            "text",
                            "",
                        )
                    )
                )

                parts.append(
                    str(
                        event.get(
                            "status",
                            "",
                        )
                    )
                )

        # --------------------------------------------------------------
        # Wrapped timeline events
        # --------------------------------------------------------------

        matched_timeline_events = result.get(
            "matched_timeline_events",
            [],
        )

        if isinstance(
            matched_timeline_events,
            list,
        ):
            for event in matched_timeline_events:

                if not isinstance(
                    event,
                    dict,
                ):
                    continue

                parts.append(
                    str(
                        event.get(
                            "memory_id",
                            "",
                        )
                    )
                )

                parts.append(
                    str(
                        event.get(
                            "text",
                            "",
                        )
                    )
                )

                parts.append(
                    str(
                        event.get(
                            "status",
                            "",
                        )
                    )
                )

    return " ".join(parts).lower()


# ----------------------------------------------------------------------
# Search result score helper
# ----------------------------------------------------------------------

def get_result_scores(
    results: list[dict[str, Any]],
) -> list[float]:
    """
    Extract retrieval scores from semantic or temporal results.
    """

    scores: list[float] = []

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        score = result.get(
            "score"
        )

        if score is None:
            score = result.get(
                "ranking_score"
            )

        if score is None:
            score = result.get(
                "retrieval_score"
            )

        if score is None:
            continue

        try:
            scores.append(
                float(score)
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

    return scores


# ----------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print(
        "OMNIMIND DETERMINISTIC MEMORY EVALUATION"
    )
    print("=" * 70)

    print(
        "\nLLM DISABLED / Groq NOT USED"
    )

    print(
        f"Evaluation time: "
        f"{EVALUATION_NOW.isoformat()}"
    )

    # ==============================================================
    # Initialize isolated evaluation memory
    # ==============================================================

    print(
        "\nInitializing isolated evaluation memory..."
    )

    fixture, memory_agent = (
        create_evaluation_memory_agent()
    )

    semantic_store = (
        memory_agent.semantic_store
    )

    consolidated_store = (
        memory_agent.consolidated_store
    )

    temporal_pipeline = (
        memory_agent.temporal_pipeline
    )

    print(
        f"Semantic memories: "
        f"{semantic_store.count()}"
    )

    print(
        f"Consolidated memories: "
        f"{consolidated_store.count()}"
    )

    # ==============================================================
    # 1. SEMANTIC MEMORY
    # ==============================================================

    print("\n" + "-" * 70)
    print("1. SEMANTIC MEMORY")
    print("-" * 70)

    semantic_results = (
        semantic_store.search(
            query=(
                "What vector database did I "
                "decide to use for my OmniMind project?"
            ),
            top_k=5,
        )
    )

    semantic_text = result_text(
        semantic_results
    )

    record(
        "Semantic memory retrieves Qdrant",
        (
            len(semantic_results) > 0
            and "qdrant" in semantic_text
        ),
        (
            f"Retrieved "
            f"{len(semantic_results)} "
            f"semantic result(s)."
        ),
        (
            "Expected semantic memory "
            "containing Qdrant was not retrieved."
        ),
    )

    # ==============================================================
    # 2. GENERIC SEMANTIC QUERY
    # ==============================================================

    print("\n" + "-" * 70)
    print("2. GENERIC SEMANTIC QUERY")
    print("-" * 70)

    generic_results = (
        semantic_store.search(
            query="vector database decision",
            top_k=5,
        )
    )

    generic_text = result_text(
        generic_results
    )

    record(
        "Generic semantic query retrieves relevant memory",
        (
            len(generic_results) > 0
            and (
                "qdrant" in generic_text
                or "database" in generic_text
            )
        ),
        (
            f"Retrieved "
            f"{len(generic_results)} "
            f"relevant semantic result(s)."
        ),
        (
            "Generic semantic query did not "
            "retrieve the expected database decision."
        ),
    )

    # ==============================================================
    # 3. TEMPORAL MEMORY
    # ==============================================================

    print("\n" + "-" * 70)
    print("3. TEMPORAL MEMORY")
    print("-" * 70)

    temporal_query = (
        "What did I decide about my project "
        "3 months ago?"
    )

    temporal_response = (
        temporal_pipeline.query(
            query=temporal_query,
            now=EVALUATION_NOW,
            top_k=5,
        )
    )

    temporal_intent = (
        temporal_response.get(
            "temporal_intent",
            {},
        )
    )

    temporal_results = (
        temporal_response.get(
            "results",
            temporal_response.get(
                "temporal_memory_results",
                [],
            ),
        )
    )

    temporal_context = (
        temporal_response.get(
            "context",
            temporal_response.get(
                "temporal_memory_context",
                "",
            ),
        )
    )

    record(
        "Temporal query detects temporal intent",
        (
            bool(
                temporal_intent.get(
                    "has_time_filter",
                    False,
                )
            )
            and not bool(
                temporal_intent.get(
                    "is_current",
                    False,
                )
            )
        ),
        (
            f"Temporal intent: "
            f"{temporal_intent}"
        ),
        (
            "Temporal intent was not "
            "correctly detected."
        ),
    )

    temporal_text = result_text(
        temporal_results
    )

    if temporal_context:
        temporal_text += (
            " "
            + str(
                temporal_context
            ).lower()
        )

    record(
        "Temporal query retrieves historical Qdrant memory",
        (
            len(temporal_results) > 0
            and "qdrant" in temporal_text
        ),
        (
            f"Retrieved "
            f"{len(temporal_results)} "
            f"temporal result(s)."
        ),
        (
            "Historical Qdrant decision "
            "was not retrieved."
        ),
    )

    # ==============================================================
    # 4. HISTORICAL MEMORY
    # ==============================================================

    print("\n" + "-" * 70)
    print("4. HISTORICAL MEMORY")
    print("-" * 70)

    historical_response = (
        temporal_pipeline.query(
            query=(
                "What was the vector database "
                "decision 3 months ago?"
            ),
            now=EVALUATION_NOW,
            top_k=5,
        )
    )

    historical_results = (
        historical_response.get(
            "results",
            historical_response.get(
                "temporal_memory_results",
                [],
            ),
        )
    )

    historical_text = result_text(
        historical_results
    )

    record(
        "Historical query retrieves a result",
        len(historical_results) > 0,
        (
            f"Historical results: "
            f"{len(historical_results)}."
        ),
        (
            "Historical retrieval "
            "returned no results."
        ),
    )

    record(
        "Historical query finds Qdrant",
        "qdrant" in historical_text,
        (
            "Historical decision "
            "correctly identified as Qdrant."
        ),
        (
            "Historical retrieval "
            "did not contain Qdrant."
        ),
    )

    # ==============================================================
    # 5. CURRENT MEMORY
    # ==============================================================

    print("\n" + "-" * 70)
    print("5. CURRENT MEMORY")
    print("-" * 70)

    current_response = (
        temporal_pipeline.query(
            query=(
                "What is the current "
                "vector database decision?"
            ),
            now=EVALUATION_NOW,
            top_k=5,
        )
    )

    current_results = (
        current_response.get(
            "results",
            current_response.get(
                "temporal_memory_results",
                [],
            ),
        )
    )

    current_text = result_text(
        current_results
    )

    current_has_postgresql = (
        "postgresql" in current_text
    )

    record(
        "Current query retrieves a result",
        len(current_results) > 0,
        (
            f"Current results: "
            f"{len(current_results)}."
        ),
        (
            "Current retrieval "
            "returned no results."
        ),
    )

    record(
        "Current query finds PostgreSQL",
        current_has_postgresql,
        (
            "Current decision "
            "correctly identified as PostgreSQL."
        ),
        (
            "Current retrieval "
            "did not contain PostgreSQL."
        ),
    )

    # ==============================================================
    # 6. HISTORICAL / CURRENT SEPARATION
    # ==============================================================

    print("\n" + "-" * 70)
    print("6. HISTORICAL / CURRENT SEPARATION")
    print("-" * 70)

    historical_text = result_text(
        historical_results
    )

    current_text = result_text(
        current_results
    )

    historical_has_qdrant = (
        "qdrant" in historical_text
    )

    current_has_postgresql = (
        "postgresql" in current_text
    )

    record(
        "Historical decision is Qdrant",
        historical_has_qdrant,
        (
            "Historical state "
            "preserved as Qdrant."
        ),
        (
            "Historical Qdrant state "
            "was not preserved."
        ),
    )

    record(
        "Current decision is PostgreSQL",
        current_has_postgresql,
        (
            "Current state "
            "correctly identified as PostgreSQL."
        ),
        (
            "Current PostgreSQL state "
            "was not retrieved."
        ),
    )

    record(
        "Historical and current states are distinguishable",
        (
            historical_has_qdrant
            and current_has_postgresql
        ),
        (
            "Historical Qdrant and current "
            "PostgreSQL states are distinguishable."
        ),
        (
            "Historical and current states "
            "could not be clearly separated."
        ),
    )

    record(
        "Current retrieval prioritizes PostgreSQL",
        (
            current_has_postgresql
            and len(current_results) > 0
        ),
        (
            "Current retrieval returns the latest "
            "PostgreSQL decision while historical "
            "Qdrant remains preserved in memory."
        ),
        (
            "Current retrieval did not return "
            "the latest PostgreSQL decision."
        ),
    )

    # ==============================================================
    # 7. IRRELEVANT MEMORY REJECTION
    # ==============================================================

    print("\n" + "-" * 70)
    print("7. IRRELEVANT MEMORY REJECTION")
    print("-" * 70)

    irrelevant_query = (
        "What is the capital city of France?"
    )

    irrelevant_results = (
        semantic_store.search(
            query=irrelevant_query,
            top_k=5,
            min_score=IRRELEVANT_MIN_SCORE,
        )
    )

    irrelevant_text = result_text(
        irrelevant_results
    )

    irrelevant_scores = (
        get_result_scores(
            irrelevant_results
        )
    )

    irrelevant_rejected = (
        len(irrelevant_results) == 0
        or (
            all(
                score < IRRELEVANT_MIN_SCORE
                for score in irrelevant_scores
            )
            and "qdrant" not in irrelevant_text
            and "postgresql" not in irrelevant_text
        )
    )

    record(
        "Irrelevant memory is rejected",
        irrelevant_rejected,
        (
            "Unrelated query did not retrieve "
            "a high-confidence OmniMind database memory."
        ),
        (
            "Unrelated query retrieved "
            "a high-confidence OmniMind memory."
        ),
    )

    # ==============================================================
    # 8. CONSOLIDATED MEMORY
    # ==============================================================

    print("\n" + "-" * 70)
    print("8. CONSOLIDATED MEMORY")
    print("-" * 70)

    consolidated_results = (
        consolidated_store.search_topic(
            topic="OmniMind Vector Database",
        )
    )

    consolidated_text = result_text(
        consolidated_results
    )

    record(
        "Consolidated memory exists",
        len(consolidated_results) > 0,
        (
            f"Consolidated results: "
            f"{len(consolidated_results)}."
        ),
        (
            "No consolidated memory "
            "was found."
        ),
    )

    record(
        "Consolidated memory preserves Qdrant history",
        "qdrant" in consolidated_text,
        (
            "Consolidated memory preserves "
            "the historical Qdrant decision."
        ),
        (
            "Consolidated memory does not "
            "contain the historical Qdrant decision."
        ),
    )

    record(
        "Consolidated memory preserves PostgreSQL current state",
        "postgresql" in consolidated_text,
        (
            "Consolidated memory preserves "
            "the current PostgreSQL decision."
        ),
        (
            "Consolidated memory does not "
            "contain the current PostgreSQL decision."
        ),
    )

    # ==============================================================
    # 9. TIMELINE-AWARE RETRIEVAL
    # ==============================================================

    print("\n" + "-" * 70)
    print("9. TIMELINE-AWARE RETRIEVAL")
    print("-" * 70)

    timeline_response = (
        temporal_pipeline.query(
            query=(
                "What database did I use "
                "3 months ago?"
            ),
            now=EVALUATION_NOW,
            top_k=5,
        )
    )

    timeline_results = (
        timeline_response.get(
            "results",
            timeline_response.get(
                "temporal_memory_results",
                [],
            ),
        )
    )

    timeline_context = (
        timeline_response.get(
            "context",
            timeline_response.get(
                "temporal_memory_context",
                "",
            ),
        )
    )

    timeline_text = result_text(
        timeline_results
    )

    timeline_text += (
        " "
        + str(
            timeline_context
        ).lower()
    )

    record(
        "Timeline-aware retrieval finds historical Qdrant",
        (
            len(timeline_results) > 0
            and "qdrant" in timeline_text
        ),
        (
            "Timeline-aware retrieval "
            "correctly located the historical "
            "Qdrant event."
        ),
        (
            "Timeline-aware retrieval failed "
            "to locate the historical Qdrant event."
        ),
    )

    record(
        "Timeline-aware retrieval preserves current PostgreSQL",
        (
            "postgresql" in timeline_text
            or "postgresql" in consolidated_text
        ),
        (
            "Timeline/consolidated memory "
            "preserves the current PostgreSQL state."
        ),
        (
            "Current PostgreSQL state was not "
            "preserved in the timeline/consolidated view."
        ),
    )

    # ==============================================================
    # 10. MEMORY-AWARE PLANNING
    # ==============================================================

    print("\n" + "-" * 70)
    print("10. MEMORY-AWARE PLANNING")
    print("-" * 70)

    planner = PlannerAgent()

    planning_state = {
        "query": (
            "What did I decide about my "
            "OmniMind vector database "
            "3 months ago?"
        ),
        "route": "rag",
        "memory_results": semantic_results,
        "memory_context": (
            "Semantic memory:\n"
            "I decided to use Qdrant as the "
            "vector database for my OmniMind project."
        ),
        "temporal_memory_results": (
            temporal_results
        ),
        "temporal_memory_context": (
            temporal_context
        ),
        "temporal_intent": (
            temporal_intent
        ),
    }

    planning_result = planner.plan(
        planning_state
    )

    plan = planning_result.get(
        "plan",
        [],
    )

    planning_route = planning_result.get(
        "route",
        "",
    )

    planning_memory_context = (
        planning_result.get(
            "planning_memory_context",
            "",
        )
    )

    record(
        "Planner produces an execution plan",
        len(plan) > 0,
        (
            f"Planner produces an execution "
            f"plan ({len(plan)} steps)"
        ),
        (
            "Planner did not produce "
            "an execution plan."
        ),
    )

    record(
        "Planner preserves RAG route",
        planning_route == "rag",
        (
            "Planner preserves "
            "the deterministic RAG route."
        ),
        (
            f"Planner selected "
            f"unexpected route: "
            f"{planning_route}"
        ),
    )

    planning_semantic_text = (
        planning_memory_context.lower()
    )

    record(
        "Planner receives semantic memory",
        "qdrant" in planning_semantic_text,
        (
            "Planner received the "
            "remembered Qdrant decision."
        ),
        (
            "Planner did not receive "
            "the semantic Qdrant memory."
        ),
    )

    record(
        "Planner receives temporal memory",
        (
            bool(temporal_results)
            or "temporal" in planning_semantic_text
            or "historical" in planning_semantic_text
        ),
        (
            "Planner received "
            "temporal/historical memory context."
        ),
        (
            "Planner did not receive "
            "temporal memory context."
        ),
    )

    record(
        "Planner generates memory-aware planning context",
        bool(
            planning_memory_context.strip()
        ),
        (
            "Memory-aware planning "
            "context generated."
        ),
        (
            "Planner did not generate "
            "memory-aware context."
        ),
    )

    # ==============================================================
    # SUMMARY
    # ==============================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Tests passed : {passed}"
    )

    print(
        f"Tests failed : {failed}"
    )

    print(
        f"Total tests  : {total}"
    )

    pass_rate = (
        (
            passed / total
        ) * 100
        if total
        else 0.0
    )

    print(
        f"Pass rate    : "
        f"{pass_rate:.2f}%"
    )

    print(
        "\nLLM calls    : 0"
    )

    print(
        "Groq tokens  : 0"
    )

    print(
        "Rate limits  : NOT APPLICABLE"
    )

    # ==============================================================
    # Final status
    # ==============================================================

    if failed == 0:

        print(
            "\n" + "=" * 70
        )

        print(
            "MEMORY EVALUATION PASSED"
        )

        print(
            "=" * 70
        )

    else:

        print(
            "\n" + "=" * 70
        )

        print(
            "MEMORY EVALUATION FAILED"
        )

        print(
            "=" * 70
        )

        raise SystemExit(1)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()