"""
OmniMind Evaluation Metrics

Reusable evaluation metrics for:
- memory retrieval
- temporal memory
- answer correctness
- relevance
- citation coverage
- historical/current distinction

All metrics are deterministic and lightweight.
No LLM judge is used here.
"""

from __future__ import annotations

import re
from typing import Iterable, Any


# ============================================================
# BASIC HELPERS
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparisons.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Normalize common Unicode whitespace.
    text = (
        text
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2009", " ")
        .replace("\u2007", " ")
    )

    # Keep words and numbers.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def _tokens(text: str) -> set[str]:
    """
    Return normalized word tokens.
    """

    normalized = _normalize_text(
        text
    )

    if not normalized:
        return set()

    return set(
        normalized.split()
    )


def _to_list(
    value: Any,
) -> list[Any]:
    """
    Normalize a value into a list.

    Useful for evaluation datasets where a field may
    contain either one value or multiple values.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    return [value]


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def retrieval_precision(
    retrieved_ids: Iterable[str],
    relevant_ids: Iterable[str],
) -> float:
    """
    Calculate retrieval precision.

    Precision =
        relevant retrieved items /
        total retrieved items
    """

    retrieved = list(
        retrieved_ids
    )

    relevant = set(
        relevant_ids
    )

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        1
        for item in retrieved
        if item in relevant
    )

    return (
        relevant_retrieved
        / len(retrieved)
    )


def retrieval_recall(
    retrieved_ids: Iterable[str],
    relevant_ids: Iterable[str],
) -> float:
    """
    Calculate retrieval recall.

    Recall =
        relevant retrieved items /
        total relevant items
    """

    retrieved = set(
        retrieved_ids
    )

    relevant = set(
        relevant_ids
    )

    if not relevant:
        return 0.0

    relevant_retrieved = len(
        retrieved.intersection(
            relevant
        )
    )

    return (
        relevant_retrieved
        / len(relevant)
    )


def retrieval_f1(
    precision: float,
    recall: float,
) -> float:
    """
    Calculate F1 from precision and recall.
    """

    precision = float(
        precision
    )

    recall = float(
        recall
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


# ============================================================
# TEXT RELEVANCE
# ============================================================

def token_overlap_score(
    answer: str,
    reference: str,
) -> float:
    """
    Calculate token overlap between an answer
    and a reference answer.

    This is deterministic and lightweight.
    It is not an LLM judge.
    """

    answer_tokens = _tokens(
        answer
    )

    reference_tokens = _tokens(
        reference
    )

    if not reference_tokens:
        return 0.0

    overlap = answer_tokens.intersection(
        reference_tokens
    )

    return (
        len(overlap)
        / len(reference_tokens)
    )


def keyword_coverage(
    text: str,
    keywords: Iterable[str],
) -> float:
    """
    Measure how many required keywords
    appear in the text.
    """

    normalized = _normalize_text(
        text
    )

    keywords = [
        _normalize_text(keyword)
        for keyword in keywords
        if keyword
    ]

    if not keywords:
        return 1.0

    matched = sum(
        1
        for keyword in keywords
        if keyword in normalized
    )

    return (
        matched
        / len(keywords)
    )


# ============================================================
# ANSWER CORRECTNESS
# ============================================================

def answer_contains_required_facts(
    answer: str,
    required_facts: Iterable[str] | None = None,
    *,
    expected_facts: Iterable[str] | None = None,
) -> bool:
    """
    Return True when every required fact appears
    in the answer.

    `required_facts` is the canonical parameter.

    `expected_facts` is supported as a compatibility
    alias so older evaluation code does not break.
    """

    if required_facts is None:
        required_facts = expected_facts

    facts = _to_list(
        required_facts
    )

    normalized = _normalize_text(
        answer
    )

    if not facts:
        return True

    for fact in facts:

        fact_normalized = _normalize_text(
            fact
        )

        if (
            not fact_normalized
            or fact_normalized
            not in normalized
        ):
            return False

    return True


def answer_fact_coverage(
    answer: str,
    required_facts: Iterable[str] | None = None,
    *,
    expected_facts: Iterable[str] | None = None,
) -> float:
    """
    Calculate the proportion of required facts
    represented in the answer.

    `required_facts` is the canonical parameter.

    `expected_facts` is supported as a compatibility
    alias.
    """

    if required_facts is None:
        required_facts = expected_facts

    facts = _to_list(
        required_facts
    )

    normalized = _normalize_text(
        answer
    )

    normalized_facts = [
        _normalize_text(fact)
        for fact in facts
        if fact
    ]

    if not normalized_facts:
        return 1.0

    matched = sum(
        1
        for fact in normalized_facts
        if fact in normalized
    )

    return (
        matched
        / len(normalized_facts)
    )


# ============================================================
# TEMPORAL MEMORY METRICS
# ============================================================

def historical_memory_accuracy(
    answer: str,
    historical_facts: Iterable[str] | None = None,
    *,
    expected: Iterable[str] | None = None,
) -> float:
    """
    Measure whether historical facts are preserved
    in the final answer.

    `expected` is supported as a compatibility alias.
    """

    if historical_facts is None:
        historical_facts = expected

    return answer_fact_coverage(
        answer,
        historical_facts,
    )


def current_memory_accuracy(
    answer: str,
    current_facts: Iterable[str] | None = None,
    *,
    expected: Iterable[str] | None = None,
) -> float:
    """
    Measure whether explicitly required current
    facts are represented in the answer.

    `expected` is supported as a compatibility alias.
    """

    if current_facts is None:
        current_facts = expected

    return answer_fact_coverage(
        answer,
        current_facts,
    )


def historical_current_separation(
    answer: str,
    historical_facts: Iterable[str] | None = None,
    current_facts: Iterable[str] | None = None,
    *,
    historical: Iterable[str] | None = None,
    current: Iterable[str] | None = None,
) -> float:
    """
    Evaluate whether an answer preserves the distinction
    between historical and current facts.

    Scoring:

    1.0
        Historical facts are present and the answer
        explicitly distinguishes the current state.

    0.5
        Historical facts are present but current and
        historical states are insufficiently separated.

    0.0
        Historical facts are missing.

    Compatibility aliases:
        historical
        current
    """

    if historical_facts is None:
        historical_facts = historical

    if current_facts is None:
        current_facts = current

    normalized = _normalize_text(
        answer
    )

    historical_list = _to_list(
        historical_facts
    )

    current_list = _to_list(
        current_facts
    )

    historical_normalized = [
        _normalize_text(fact)
        for fact in historical_list
        if fact
    ]

    current_normalized = [
        _normalize_text(fact)
        for fact in current_list
        if fact
    ]

    # ------------------------------------------------------------
    # Historical facts must be present.
    # ------------------------------------------------------------

    if not historical_normalized:
        return 1.0 if not current_normalized else 0.0

    historical_present = all(
        fact in normalized
        for fact in historical_normalized
    )

    if not historical_present:
        return 0.0

    # ------------------------------------------------------------
    # No current facts required.
    # ------------------------------------------------------------

    if not current_normalized:
        return 1.0

    # ------------------------------------------------------------
    # Look for explicit temporal separation.
    # ------------------------------------------------------------

    temporal_markers = [
        "later",
        "later changed",
        "previously",
        "historical",
        "historically",
        "at that time",
        "at the time",
        "three months ago",
        "3 months ago",
        "earlier",
        "initially",
        "originally",
        "current",
        "currently",
        "now",
        "today",
        "changed",
        "changed to",
        "switched",
        "replaced",
        "eventually",
    ]

    has_temporal_marker = any(
        marker in normalized
        for marker in temporal_markers
    )

    if has_temporal_marker:
        return 1.0

    return 0.5


# ============================================================
# CITATION METRICS
# ============================================================

# ============================================================
# CITATION METRICS
# ============================================================

def extract_document_citations(
    answer: str,
) -> list[str]:
    """
    Extract document citations from an OmniMind answer.

    Current citation format:

        [1]
        [2]
        [3]

    Legacy format is also supported:

        [Source 1, Page 2]

    The current numeric citation format is intentionally kept
    separate from web citations such as:

        [Web Source 1]
    """

    if not answer:
        return []

    citations: list[str] = []

    # ------------------------------------------------------------
    # Current structured citation format
    # ------------------------------------------------------------

    current_pattern = r"(?<![A-Za-z])\[(\d+)\]"

    current_citations = re.findall(
        current_pattern,
        answer,
        flags=re.IGNORECASE,
    )

    for citation_number in current_citations:
        citation = f"[{citation_number}]"

        if citation not in citations:
            citations.append(citation)

    # ------------------------------------------------------------
    # Legacy citation format
    # ------------------------------------------------------------

    legacy_pattern = (
        r"\[Source\s+\d+,\s*Page\s+\d+\]"
    )

    legacy_citations = re.findall(
        legacy_pattern,
        answer,
        flags=re.IGNORECASE,
    )

    for citation in legacy_citations:
        if citation not in citations:
            citations.append(citation)

    return citations


def extract_web_citations(
    answer: str,
) -> list[str]:
    """
    Extract web citations in the expected format:

        [Web Source N]
    """

    if not answer:
        return []

    pattern = (
        r"\[Web\s+Source\s+\d+\]"
    )

    return re.findall(
        pattern,
        answer,
        flags=re.IGNORECASE,
    )


def citation_coverage(
    answer: str,
    expected_document_citations: int = 0,
    expected_web_citations: int = 0,
    *,
    expected_source_types: Iterable[str] | None = None,
) -> float:
    """
    Measure citation coverage.

    Supports current OmniMind document citations:

        [1]
        [2]
        [3]

    Legacy document citations are also supported:

        [Source 1, Page 2]

    Web citations:

        [Web Source 1]

    The metric checks whether the answer contains at least
    the expected number of document and/or web citation markers.

    If no citations are expected, returns 1.0.

    Compatibility:
        expected_source_types can be supplied as:
            ["document"]
            ["web"]
            ["document", "web"]
    """

    # ------------------------------------------------------------
    # Compatibility handling for expected_source_types
    # ------------------------------------------------------------

    if expected_source_types is not None:

        source_types = {
            str(source).strip().lower()
            for source in expected_source_types
            if source
        }

        if (
            expected_document_citations == 0
            and "document" in source_types
        ):
            expected_document_citations = 1

        if (
            expected_web_citations == 0
            and "web" in source_types
        ):
            expected_web_citations = 1

    # ------------------------------------------------------------
    # Normalize expected counts
    # ------------------------------------------------------------

    expected_document_citations = max(
        0,
        int(expected_document_citations),
    )

    expected_web_citations = max(
        0,
        int(expected_web_citations),
    )

    # ------------------------------------------------------------
    # Extract actual citations
    # ------------------------------------------------------------

    document_citations = len(
        extract_document_citations(
            answer
        )
    )

    web_citations = len(
        extract_web_citations(
            answer
        )
    )

    # ------------------------------------------------------------
    # Expected total
    # ------------------------------------------------------------

    expected_total = (
        expected_document_citations
        + expected_web_citations
    )

    if expected_total == 0:
        return 1.0

    # ------------------------------------------------------------
    # Count only up to expected amount.
    # ------------------------------------------------------------

    actual_total = (
        min(
            document_citations,
            expected_document_citations,
        )
        +
        min(
            web_citations,
            expected_web_citations,
        )
    )

    return (
        actual_total
        / expected_total
    )

# ============================================================
# SOURCE COVERAGE
# ============================================================

def source_coverage(
    answer: str,
    sources: Iterable[str],
) -> float:
    """
    Measure whether expected source names
    are represented in the answer.
    """

    normalized = _normalize_text(
        answer
    )

    source_list = [
        _normalize_text(source)
        for source in sources
        if source
    ]

    if not source_list:
        return 1.0

    matched = sum(
        1
        for source in source_list
        if source in normalized
    )

    return (
        matched
        / len(source_list)
    )


# ============================================================
# AGGREGATE SCORE
# ============================================================

def mean_score(
    scores: Iterable[float],
) -> float:
    """
    Calculate the arithmetic mean of scores.
    """

    values = [
        float(score)
        for score in scores
    ]

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def bounded_score(
    score: float,
) -> float:
    """
    Clamp a score to [0.0, 1.0].
    """

    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )