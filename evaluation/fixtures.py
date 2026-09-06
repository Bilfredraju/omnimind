from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agents.memory_agent import MemoryAgent
from memory.consolidated_store import ConsolidatedMemoryStore
from memory.semantic_store import SemanticMemoryStore


class EvaluationMemoryFixture:
    """
    Isolated benchmark memory environment for OmniMind evaluation.

    The fixture never touches the production memory files.
    """

    def __init__(
        self,
        root: str | Path = "data/evaluation_memory",
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.semantic_path = self.root / "semantic_memories.json"
        self.consolidated_path = self.root / "consolidated_memories.json"

        self.semantic_store = SemanticMemoryStore(
            path=str(self.semantic_path)
        )

        self.consolidated_store = ConsolidatedMemoryStore(
            path=str(self.consolidated_path)
        )

        self._clear()

    def _clear(self) -> None:
        self.semantic_store.clear()
        self.consolidated_store.clear()

    def seed(self) -> None:
        """
        Seed deterministic memories required by the evaluation dataset.
        """

        # --------------------------------------------------------------
        # Historical decision
        # --------------------------------------------------------------

        self.semantic_store.add(
            text=(
                "I decided to use Qdrant as the vector database "
                "for my OmniMind project."
            ),
            metadata={
                "memory_id": "eval-qdrant-decision",
                "created_at": "2026-06-06T10:00:00+00:00",
                "source": "evaluation_fixture",
                "type": "decision",
                "status": "historical",
                "importance": 1.0,
            },
        )

        # --------------------------------------------------------------
        # Current decision
        # --------------------------------------------------------------

        self.semantic_store.add(
            text=(
                "I decided to use PostgreSQL as the current database "
                "for my OmniMind project."
            ),
            metadata={
                "memory_id": "eval-postgres-decision",
                "created_at": "2026-09-01T10:00:00+00:00",
                "source": "evaluation_fixture",
                "type": "decision",
                "status": "current",
                "importance": 1.0,
            },
        )

        # --------------------------------------------------------------
        # Historical consolidated memory
        # --------------------------------------------------------------

        self.consolidated_store.add(
            {
                "consolidation_id": "eval-omnimind-vector-db",
                "topic": "OmniMind Vector Database",
                "summary": (
                    "The project initially used Qdrant. "
                    "The current decision is PostgreSQL."
                ),
                "created_at": "2026-09-01T10:00:00+00:00",
                "current_memory_id": "eval-postgres-decision",
                "historical_memory_ids": [
                    "eval-qdrant-decision"
                ],
                "timeline": [
                    {
                        "memory_id": "eval-qdrant-decision",
                        "timestamp": "2026-06-06T10:00:00+00:00",
                        "text": (
                            "I decided to use Qdrant as the vector database "
                            "for my OmniMind project."
                        ),
                        "type": "decision",
                        "status": "historical",
                        "importance": 1.0,
                    },
                    {
                        "memory_id": "eval-postgres-decision",
                        "timestamp": "2026-09-01T10:00:00+00:00",
                        "text": (
                            "I decided to use PostgreSQL as the current "
                            "database for my OmniMind project."
                        ),
                        "type": "decision",
                        "status": "current",
                        "importance": 1.0,
                    },
                ],
            }
        )

    def create_agent(self) -> MemoryAgent:
        """
        Create a MemoryAgent connected only to evaluation memory.
        """

        agent = MemoryAgent(
            semantic_store=self.semantic_store,
            consolidated_store=self.consolidated_store,
        )

        return agent


def create_evaluation_memory_agent() -> tuple[
    EvaluationMemoryFixture,
    MemoryAgent,
]:
    """
    Convenience factory used by the live evaluator.
    """

    fixture = EvaluationMemoryFixture()
    fixture.seed()

    agent = fixture.create_agent()

    return fixture, agent