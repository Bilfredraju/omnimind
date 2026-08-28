import json
import re

from models.llm.groq_provider import GroqProvider
from agents.state import AgentState


class PlannerAgent:
    """Create an execution plan and route for a user query."""

    def __init__(self):
        self.llm = GroqProvider()

    # ======================================================
    # Deterministic routing
    # ======================================================

    def _rule_based_route(
        self,
        query: str,
    ):
        """
        Determine the route using explicit query signals.

        Returns:
            "rag", "research", "both", or None
        """

        query_lower = query.lower().strip()

        # --------------------------------------------------
        # BOTH
        # --------------------------------------------------
        # These queries explicitly combine the user's
        # document with current/external information.
        # --------------------------------------------------

        both_patterns = [
            r"\bcompare\b.*\b(recent|latest|current|external)\b",
            r"\bcompare\b.*\b(document|pdf|paper|file)\b.*\b(recent|latest|current)\b",
            r"\b(document|pdf|paper|file)\b.*\bcompare\b.*\b(recent|latest|current)\b",
            r"\bmy document\b.*\b(research|web)\b",
            r"\bmy pdf\b.*\b(research|web)\b",
            r"\buploaded document\b.*\b(research|web)\b",
            r"\buploaded pdf\b.*\b(research|web)\b",
        ]

        for pattern in both_patterns:
            if re.search(
                pattern,
                query_lower,
            ):
                return "both"

        # --------------------------------------------------
        # RESEARCH
        # --------------------------------------------------

        research_keywords = [
            "latest",
            "recent",
            "current",
            "today",
            "now",
            "news",
            "developments",
            "web search",
            "search the web",
            "external research",
            "current state",
        ]

        for keyword in research_keywords:
            if keyword in query_lower:
                return "research"

        # --------------------------------------------------
        # RAG
        # --------------------------------------------------

        document_keywords = [
            "my document",
            "my pdf",
            "my paper",
            "uploaded document",
            "uploaded pdf",
            "uploaded paper",
            "the document",
            "the pdf",
            "the paper",
            "this document",
            "this pdf",
            "this paper",
        ]

        for keyword in document_keywords:
            if keyword in query_lower:
                return "rag"

        # No strong signal.
        return None

    # ======================================================
    # Create execution plan
    # ======================================================

    def _create_steps(
        self,
        route: str,
    ) -> list[str]:
        """Create a plan based on the selected route."""

        if route == "rag":
            return [
                "Search the uploaded documents for relevant information.",
                "Analyze the retrieved information.",
                "Generate the final answer.",
            ]

        if route == "research":
            return [
                "Perform external web research.",
                "Analyze the retrieved information.",
                "Generate the final answer.",
            ]

        if route == "both":
            return [
                "Retrieve relevant information from the uploaded documents.",
                "Perform external web research.",
                "Analyze and compare both evidence sources.",
                "Generate the final answer.",
            ]

        return [
            "Search relevant documents.",
            "Analyze the retrieved information.",
            "Generate the final answer.",
        ]

    # ======================================================
    # Planner
    # ======================================================

    def plan(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state["query"]

        # --------------------------------------------------
        # First use deterministic routing.
        # --------------------------------------------------

        rule_route = self._rule_based_route(
            query
        )

        if rule_route is not None:

            steps = self._create_steps(
                rule_route
            )

            return {
                **state,
                "plan": steps,
                "route": rule_route,
                "current_step": "planning_complete",
            }

        # --------------------------------------------------
        # LLM fallback for ambiguous questions.
        # --------------------------------------------------

        prompt = f"""
You are the Planner Agent in an autonomous AI system
called OmniMind.

Determine where the answer should come from.

Available routes:

"rag"
Use when the answer should primarily come from the
user's uploaded documents.

"research"
Use when the user needs external, current, recent,
or web-based information.

"both"
Use when the user needs both the uploaded documents
and external/current information.

Rules:

1. Return ONLY valid JSON.
2. Include "route".
3. Include "steps".
4. route must be exactly:
   rag, research, or both.
5. steps must be a list.
6. Do not answer the user's question.
7. Do not include markdown.
8. Do not include ```json.

User request:
{query}

Return:

{{
    "route": "rag",
    "steps": [
        "..."
    ]
}}
""".strip()

        response = self.llm.generate(
            prompt
        )

        try:

            # --------------------------------------------------
            # Remove accidental markdown fences.
            # --------------------------------------------------

            cleaned_response = response.strip()

            if cleaned_response.startswith(
                "```"
            ):
                cleaned_response = (
                    cleaned_response
                    .replace(
                        "```json",
                        "",
                    )
                    .replace(
                        "```",
                        "",
                    )
                    .strip()
                )

            parsed = json.loads(
                cleaned_response
            )

            route = parsed.get(
                "route",
                "rag",
            )

            steps = parsed.get(
                "steps",
                [],
            )

            # --------------------------------------------------
            # Validate route.
            # --------------------------------------------------

            if route not in {
                "rag",
                "research",
                "both",
            }:
                raise ValueError(
                    "Invalid planner route."
                )

            # --------------------------------------------------
            # Validate steps.
            # --------------------------------------------------

            if not isinstance(
                steps,
                list,
            ):
                raise ValueError(
                    "Planner steps must be a list."
                )

        except (
            json.JSONDecodeError,
            ValueError,
        ):

            route = "rag"

            steps = self._create_steps(
                "rag"
            )

        return {
            **state,
            "plan": steps,
            "route": route,
            "current_step": "planning_complete",
        }