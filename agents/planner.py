import json
import re

from models.llm.groq_provider import GroqProvider
from agents.state import AgentState


class PlannerAgent:
    """Create an execution plan and route for a user query."""

    def __init__(self):
        self.llm = GroqProvider()

    # ======================================================
    # Rule-Based Routing
    # ======================================================

    def _rule_based_route(
        self,
        query: str,
    ):
        """
        Determine the route using explicit query signals.

        Priority:
            1. both
            2. research
            3. rag

        Ambiguous questions default to RAG because the
        uploaded documents are OmniMind's primary knowledge
        source.
        """

        query_lower = query.lower().strip()

        # --------------------------------------------------
        # BOTH
        # --------------------------------------------------
        # The user wants information from the document AND
        # recent/current/external information.
        # --------------------------------------------------

        both_patterns = [
            r"\bcompare\b.*\b(recent|latest|current|external)\b",

            r"\bcompare\b.*\b(document|pdf|paper|file)\b"
            r".*\b(recent|latest|current|external)\b",

            r"\b(document|pdf|paper|file)\b"
            r".*\bcompare\b"
            r".*\b(recent|latest|current|external)\b",

            r"\bmy document\b.*\b(research|web)\b",
            r"\bmy pdf\b.*\b(research|web)\b",
            r"\bmy paper\b.*\b(research|web)\b",

            r"\buploaded document\b.*\b(research|web)\b",
            r"\buploaded pdf\b.*\b(research|web)\b",
            r"\buploaded paper\b.*\b(research|web)\b",

            r"\bcompare\b.*\bmy document\b",
            r"\bcompare\b.*\bmy pdf\b",
            r"\bcompare\b.*\bmy paper\b",
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
        # Explicit request for current/external information.
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
        # Explicit document-related questions.
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
            "document",
            "pdf",
            "paper",
            "file",
        ]

        for keyword in document_keywords:
            if keyword in query_lower:
                return "rag"

        # --------------------------------------------------
        # DEFAULT → RAG
        # --------------------------------------------------
        # If the user did not explicitly request external
        # or current information, use the uploaded documents.
        # --------------------------------------------------

        return "rag"

    # ======================================================
    # Create Plan
    # ======================================================

    def _create_steps(
        self,
        route: str,
    ) -> list[str]:
        """Create execution steps for the selected route."""

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
            "Search the uploaded documents for relevant information.",
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
        """
        Analyze the user query and determine the execution
        route and plan.
        """

        query = state["query"]

        # --------------------------------------------------
        # First: deterministic routing
        # --------------------------------------------------

        route = self._rule_based_route(
            query
        )

        steps = self._create_steps(
            route
        )

        # --------------------------------------------------
        # Return deterministic result
        # --------------------------------------------------
        #
        # We intentionally do not ask the LLM to choose the
        # route when the routing rules already determine it.
        #
        # This prevents inconsistent routing between runs.
        # --------------------------------------------------

        return {
            **state,
            "plan": steps,
            "route": route,
            "current_step": "planning_complete",
        }