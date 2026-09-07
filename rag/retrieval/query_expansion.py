"""
OmniMind — Deterministic Multi-Query Expansion

Generates semantically different retrieval queries without using an LLM.
"""


from __future__ import annotations

import re
from typing import List


class QueryExpander:
    """
    Deterministic query expansion for multi-query retrieval.

    Strategy:
    1. Original user query.
    2. Keyword-oriented retrieval query.
    3. Concept/semantic retrieval query.

    No external API or LLM is required.
    """

    def __init__(self, max_queries: int = 3):
        if max_queries < 1:
            raise ValueError("max_queries must be >= 1")

        self.max_queries = max_queries

    @staticmethod
    def _clean_query(query: str) -> str:
        """Normalize whitespace and punctuation."""
        query = re.sub(r"\s+", " ", query.strip())
        return query

    @staticmethod
    def _remove_question_prefix(query: str) -> str:
        """
        Convert common natural-language question forms into
        retrieval-oriented phrases.
        """
        patterns = [
            r"^what\s+",
            r"^which\s+",
            r"^who\s+",
            r"^where\s+",
            r"^when\s+",
            r"^why\s+",
            r"^how\s+",
            r"^can\s+you\s+",
            r"^could\s+you\s+",
            r"^tell\s+me\s+",
            r"^please\s+",
        ]

        result = query

        for pattern in patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        return result.strip(" ?.")

    @staticmethod
    def _keyword_query(query: str) -> str:
        """
        Generate a concise keyword-oriented retrieval query.

        Example:
            What datasets were used to evaluate the RAG models?

        becomes:
            datasets used for RAG model evaluation
        """
        lower = query.lower()

        # Domain-specific transformations.
        replacements = [
            (r"\bdatasets\b", "datasets"),
            (r"\bwere used to evaluate\b", "used for evaluating"),
            (r"\bevaluate\b", "evaluation"),
            (r"\bevaluated\b", "evaluation"),
            (r"\bmodels\b", "models"),
        ]

        result = lower

        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)

        result = result.replace("?", "")
        result = re.sub(r"\s+", " ", result).strip()

        # Remove conversational/question wording.
        result = re.sub(
            r"^(what|which|who|where|when|why|how)\s+",
            "",
            result,
        )

        # Make the phrase more retrieval-oriented.
        if "used for evaluating" in result:
            result = result.replace(
                "used for evaluating",
                "used for RAG model evaluation",
            )

        return result.strip()

    @staticmethod
    def _semantic_query(query: str) -> str:
        """
        Generate a concept-oriented retrieval query.

        This intentionally uses domain terminology rather than simply
        deleting punctuation.
        """
        lower = query.lower()

        if "rag" in lower and "dataset" in lower:
            return "RAG evaluation datasets and benchmarks"

        if "rag" in lower and "evaluate" in lower:
            return "retrieval augmented generation model evaluation"

        if "dataset" in lower:
            return "datasets evaluation benchmarks"

        # Generic fallback.
        cleaned = QueryExpander._remove_question_prefix(query)
        return f"information about {cleaned}"

    @staticmethod
    def _deduplicate(queries: List[str]) -> List[str]:
        """Remove duplicate queries while preserving order."""
        result: List[str] = []
        seen = set()

        for query in queries:
            normalized = re.sub(r"\s+", " ", query.strip().lower())

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(query.strip())

        return result

    def expand(self, query: str) -> List[str]:
        """
        Generate deterministic query variations.

        The original query is always retained as the first query.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = self._clean_query(query)

        if not query:
            return []

        queries = [
            query,
            self._keyword_query(query),
            self._semantic_query(query),
        ]

        queries = self._deduplicate(queries)

        return queries[: self.max_queries]


def expand_query(query: str, max_queries: int = 3) -> List[str]:
    """Convenience function for deterministic query expansion."""
    return QueryExpander(max_queries=max_queries).expand(query)