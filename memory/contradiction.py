from __future__ import annotations

import re
from typing import Any


class MemoryContradictionEngine:
    """
    Detects likely contradictions between a new memory and an existing memory.

    This engine is intentionally conservative:
    - It does not replace or delete memories.
    - It does not treat every different statement as a contradiction.
    - It looks for conflicting values within the same subject/context.
    """

    CONTRADICTION_PATTERNS = [
        # Technology / architecture decisions
        (
            r"\bqdrant\b",
            r"\b(postgresql|postgres|pinecone|weaviate|chroma|milvus)\b",
        ),
        (
            r"\b(postgresql|postgres)\b",
            r"\b(qdrant|pinecone|weaviate|chroma|milvus)\b",
        ),
        (
            r"\bpinecone\b",
            r"\b(qdrant|postgresql|postgres|weaviate|chroma|milvus)\b",
        ),
        (
            r"\bweaviate\b",
            r"\b(qdrant|postgresql|postgres|pinecone|chroma|milvus)\b",
        ),
        (
            r"\bchroma\b",
            r"\b(qdrant|postgresql|postgres|pinecone|weaviate|milvus)\b",
        ),
        (
            r"\bmilvus\b",
            r"\b(qdrant|postgresql|postgres|pinecone|weaviate|chroma)\b",
        ),
    ]

    NEGATION_PAIRS = [
        ("use", "not use"),
        ("using", "not using"),
        ("will use", "will not use"),
        ("is", "is not"),
        ("are", "are not"),
        ("can", "cannot"),
        ("should", "should not"),
        ("prefer", "do not prefer"),
        ("like", "do not like"),
        ("want", "do not want"),
    ]

    def detect(
        self,
        new_text: str,
        existing_memory: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Determine whether new_text conflicts with an existing memory.

        Returns:
            {
                "contradiction": bool,
                "confidence": float,
                "reason": str,
            }
        """

        new_text = self._normalize(new_text)
        existing_text = self._normalize(
            str(existing_memory.get("text", ""))
        )

        if not new_text or not existing_text:
            return self._result(
                False,
                0.0,
                "Insufficient text for contradiction detection.",
            )

        # Do not flag exact/similar statements as contradictions.
        if new_text == existing_text:
            return self._result(
                False,
                0.0,
                "Statements are identical.",
            )

        # Contradiction should normally occur within the same memory type.
        existing_metadata = existing_memory.get("metadata", {})
        existing_type = str(
            existing_metadata.get("type", "general")
        ).lower().strip()

        # Strong contradiction for decision/project memories.
        if existing_type in {"decision", "project", "preference", "goal"}:
            technology_conflict = self._detect_technology_conflict(
                new_text,
                existing_text,
            )

            if technology_conflict:
                return self._result(
                    True,
                    0.95,
                    technology_conflict,
                )

        # Explicit negation / reversal.
        negation_conflict = self._detect_negation_conflict(
            new_text,
            existing_text,
        )

        if negation_conflict:
            return self._result(
                True,
                0.90,
                negation_conflict,
            )

        # Detect direct opposing statements using common decision language.
        direct_conflict = self._detect_direct_conflict(
            new_text,
            existing_text,
        )

        if direct_conflict:
            return self._result(
                True,
                0.88,
                direct_conflict,
            )

        return self._result(
            False,
            0.0,
            "No contradiction detected.",
        )

    def _detect_technology_conflict(
        self,
        new_text: str,
        existing_text: str,
    ) -> str | None:
        """
        Detect conflicting technology choices.

        Example:
            Existing: "I decided to use Qdrant."
            New:      "I decided to use PostgreSQL."
        """

        technologies = [
            "qdrant",
            "postgresql",
            "postgres",
            "pinecone",
            "weaviate",
            "chroma",
            "milvus",
            "mongodb",
            "mysql",
            "sqlite",
        ]

        existing_techs = [
            tech for tech in technologies
            if re.search(rf"\b{re.escape(tech)}\b", existing_text)
        ]

        new_techs = [
            tech for tech in technologies
            if re.search(rf"\b{re.escape(tech)}\b", new_text)
        ]

        if not existing_techs or not new_techs:
            return None

        if set(existing_techs) == set(new_techs):
            return None

        # Require overlapping decision/context language.
        context_terms = [
            "use",
            "using",
            "database",
            "vector database",
            "decided",
            "chose",
            "chosen",
            "selected",
            "architecture",
            "project",
        ]

        shared_context = sum(
            1
            for term in context_terms
            if term in existing_text and term in new_text
        )

        if shared_context >= 1:
            return (
                f"Conflicting technology choices: "
                f"existing memory mentions "
                f"{', '.join(existing_techs)}, while the new memory "
                f"mentions {', '.join(new_techs)}."
            )

        return None

    def _detect_negation_conflict(
        self,
        new_text: str,
        existing_text: str,
    ) -> str | None:
        """
        Detect simple positive/negative conflicts.

        Example:
            Existing: "I will use Qdrant."
            New:      "I will not use Qdrant."
        """

        for positive, negative in self.NEGATION_PAIRS:
            positive_pattern = rf"\b{re.escape(positive)}\b"
            negative_pattern = rf"\b{re.escape(negative)}\b"

            if (
                re.search(positive_pattern, existing_text)
                and re.search(negative_pattern, new_text)
            ):
                return (
                    f"Existing memory contains '{positive}' "
                    f"while the new memory contains '{negative}'."
                )

            if (
                re.search(negative_pattern, existing_text)
                and re.search(positive_pattern, new_text)
            ):
                return (
                    f"Existing memory contains '{negative}' "
                    f"while the new memory contains '{positive}'."
                )

        return None

    def _detect_direct_conflict(
        self,
        new_text: str,
        existing_text: str,
    ) -> str | None:
        """
        Detect a small set of explicit opposing statements.
        """

        opposing_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("enabled", "disabled"),
            ("active", "inactive"),
            ("current", "obsolete"),
            ("recommended", "not recommended"),
            ("required", "not required"),
            ("works", "does not work"),
            ("supported", "unsupported"),
        ]

        for first, second in opposing_pairs:
            if (
                re.search(rf"\b{re.escape(first)}\b", existing_text)
                and re.search(rf"\b{re.escape(second)}\b", new_text)
            ):
                return (
                    f"Existing memory contains '{first}' "
                    f"while the new memory contains '{second}'."
                )

            if (
                re.search(rf"\b{re.escape(second)}\b", existing_text)
                and re.search(rf"\b{re.escape(first)}\b", new_text)
            ):
                return (
                    f"Existing memory contains '{second}' "
                    f"while the new memory contains '{first}'."
                )

        return None

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _result(
        contradiction: bool,
        confidence: float,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "contradiction": contradiction,
            "confidence": round(confidence, 4),
            "reason": reason,
        }