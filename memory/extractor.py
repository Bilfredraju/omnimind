import re
from datetime import datetime


class MemoryExtractor:
    """
    Extract potentially useful long-term memories from
    user/assistant conversations.

    This first version intentionally uses deterministic
    rules so memory creation is predictable and testable.
    """

    DECISION_PATTERNS = [
        r"\bi decided to\b",
        r"\bwe decided to\b",
        r"\bi chose\b",
        r"\bwe chose\b",
        r"\bi will use\b",
        r"\bwe will use\b",
        r"\bi'm going to use\b",
        r"\bi am going to use\b",
    ]

    PREFERENCE_PATTERNS = [
        r"\bi prefer\b",
        r"\bi like\b",
        r"\bi don't like\b",
        r"\bi want\b",
        r"\bmy preference is\b",
    ]

    GOAL_PATTERNS = [
        r"\bi want to build\b",
        r"\bi want to create\b",
        r"\bmy goal is\b",
        r"\bi plan to\b",
        r"\bi'm planning to\b",
        r"\bi am planning to\b",
    ]

    PROJECT_PATTERNS = [
        r"\bmy project\b",
        r"\bfor my project\b",
        r"\bthe project\b",
    ]

    def extract(
        self,
        user_message: str,
        assistant_message: str,
    ) -> list[dict]:

        user_message = user_message.strip()
        assistant_message = assistant_message.strip()

        if not user_message:
            return []

        memories = []

        # --------------------------------------------------
        # Decision
        # --------------------------------------------------

        if self._matches(
            user_message,
            self.DECISION_PATTERNS,
        ):

            text = self._clean_decision(
                user_message
            )

            memories.append(
                self._build_memory(
                    text=text,
                    memory_type="decision",
                    importance=0.9,
                    user_message=user_message,
                )
            )

        # --------------------------------------------------
        # Preference
        # --------------------------------------------------

        if self._matches(
            user_message,
            self.PREFERENCE_PATTERNS,
        ):

            memories.append(
                self._build_memory(
                    text=user_message,
                    memory_type="preference",
                    importance=0.8,
                    user_message=user_message,
                )
            )

        # --------------------------------------------------
        # Goal
        # --------------------------------------------------

        if self._matches(
            user_message,
            self.GOAL_PATTERNS,
        ):

            memories.append(
                self._build_memory(
                    text=user_message,
                    memory_type="goal",
                    importance=0.8,
                    user_message=user_message,
                )
            )

        # --------------------------------------------------
        # Project context
        # --------------------------------------------------

        if (
            not memories
            and self._matches(
                user_message,
                self.PROJECT_PATTERNS,
            )
        ):

            memories.append(
                self._build_memory(
                    text=(
                        f"Project context: {user_message}"
                    ),
                    memory_type="project_context",
                    importance=0.6,
                    user_message=user_message,
                )
            )

        return memories

    # ==================================================
    # Helpers
    # ==================================================

    @staticmethod
    def _matches(
        text: str,
        patterns: list[str],
    ) -> bool:

        text = text.lower()

        return any(
            re.search(
                pattern,
                text,
            )
            for pattern in patterns
        )

    @staticmethod
    def _clean_decision(
        text: str,
    ) -> str:

        text = text.strip()

        if text.endswith("."):
            return (
                "Decision: "
                + text[0].lower()
                + text[1:]
            )

        return (
            "Decision: "
            + text[0].lower()
            + text[1:]
            + "."
        )

    @staticmethod
    def _build_memory(
        text: str,
        memory_type: str,
        importance: float,
        user_message: str,
    ) -> dict:

        return {
            "text": text,
            "metadata": {
                "type": memory_type,
                "importance": importance,
                "created_at": datetime.utcnow().isoformat(),
                "source": "conversation",
                "query": user_message,
            },
        }