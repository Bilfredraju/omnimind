from __future__ import annotations

from typing import Any


class MemoryImportanceEngine:
    """
    Calculates the importance of a memory.

    Importance is based primarily on memory type, with optional
    explicit importance and lightweight content signals.
    """

    TYPE_WEIGHTS = {
        "decision": 1.00,
        "goal": 0.90,
        "preference": 0.80,
        "project": 0.85,
        "fact": 0.65,
        "technical": 0.60,
        "general": 0.40,
    }

    def calculate(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        metadata = metadata or {}

        memory_type = str(
            metadata.get("type", "general")
        ).lower().strip()

        base_score = self.TYPE_WEIGHTS.get(memory_type, 0.40)

        # Explicit importance supplied by the caller takes priority.
        explicit_importance = metadata.get("importance")

        if explicit_importance is not None:
            try:
                explicit = float(explicit_importance)
                explicit = max(0.0, min(1.0, explicit))

                # Blend explicit importance with the semantic/type score.
                score = (base_score * 0.4) + (explicit * 0.6)
            except (TypeError, ValueError):
                score = base_score
        else:
            score = base_score

        text_lower = text.lower()

        # Strong signals that a memory represents a durable decision.
        decision_signals = [
            "i decided",
            "we decided",
            "i chose",
            "we chose",
            "i will use",
            "we will use",
            "i'm going to use",
            "i am going to use",
            "we're going to use",
            "we are going to use",
        ]

        if any(signal in text_lower for signal in decision_signals):
            score += 0.05

        # Project-related information tends to remain useful longer.
        project_signals = [
            "my project",
            "our project",
            "omnimind",
            "architecture",
            "database",
            "vector database",
        ]

        if any(signal in text_lower for signal in project_signals):
            score += 0.03

        return round(min(1.0, max(0.0, score)), 4)