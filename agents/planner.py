import re

from models.llm.groq_provider import GroqProvider
from agents.state import AgentState


class PlannerAgent:
    """Create an execution plan and route for a user query.

    Phase 17.2:
    The planner is memory-aware. Previously retrieved semantic and
    temporal memory can influence the execution plan while the
    deterministic routing rules remain unchanged.
    """

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

        return "rag"

    # ======================================================
    # Memory Helpers
    # ======================================================

    def _get_memory_evidence(
        self,
        state: AgentState,
    ) -> dict:
        """
        Extract memory evidence from AgentState.

        The planner receives memory from the memory-recall
        stage. We keep semantic and temporal memory separate
        so that historical decisions are not confused with
        current knowledge.
        """

        semantic_results = state.get(
            "memory_results",
            [],
        ) or []

        temporal_results = state.get(
            "temporal_memory_results",
            [],
        ) or []

        semantic_context = state.get(
            "memory_context",
            "",
        ) or ""

        temporal_context = state.get(
            "temporal_memory_context",
            "",
        ) or ""

        temporal_intent = state.get(
            "temporal_intent",
            {},
        ) or {}

        return {
            "semantic_results": semantic_results,
            "temporal_results": temporal_results,
            "semantic_context": semantic_context,
            "temporal_context": temporal_context,
            "temporal_intent": temporal_intent,
        }

    def _memory_is_relevant(
        self,
        evidence: dict,
    ) -> bool:
        """
        Determine whether retrieved memory contains usable
        planning context.
        """

        if evidence["semantic_results"]:
            return True

        if evidence["temporal_results"]:
            return True

        if evidence["semantic_context"].strip():
            return True

        if evidence["temporal_context"].strip():
            return True

        return False

    def _memory_summary(
        self,
        evidence: dict,
    ) -> str:
        """
        Build a compact memory summary for planning.

        This is intentionally deterministic. The planner does
        not ask the LLM to interpret memory before routing.
        """

        sections = []

        if evidence["semantic_context"].strip():
            sections.append(
                "Semantic memory:\n"
                + evidence["semantic_context"].strip()
            )

        if evidence["temporal_context"].strip():
            sections.append(
                "Temporal memory:\n"
                + evidence["temporal_context"].strip()
            )

        if evidence["temporal_intent"]:
            temporal_expression = evidence[
                "temporal_intent"
            ].get("expression")

            if temporal_expression:
                sections.append(
                    "Temporal intent: "
                    + str(temporal_expression)
                )

        return "\n\n".join(sections)

    # ======================================================
    # Create Plan
    # ======================================================

    def _create_steps(
        self,
        route: str,
        memory_evidence: dict | None = None,
    ) -> list[str]:
        """
        Create execution steps for the selected route.

        Memory-aware planning adds a memory/context step when
        relevant memory was successfully retrieved.
        """

        memory_evidence = memory_evidence or {}

        memory_relevant = self._memory_is_relevant(
            memory_evidence
        )

        steps = []

        # --------------------------------------------------
        # Memory-aware planning step
        # --------------------------------------------------

        if memory_relevant:
            temporal_results = memory_evidence.get(
                "temporal_results",
                [],
            ) or []

            temporal_intent = memory_evidence.get(
                "temporal_intent",
                {},
            ) or {}

            if temporal_results or temporal_intent:
                steps.append(
                    "Use relevant temporal memory and historical "
                    "context when planning the response."
                )
            else:
                steps.append(
                    "Use relevant remembered context when "
                    "planning the response."
                )

        # --------------------------------------------------
        # Route-specific steps
        # --------------------------------------------------

        if route == "rag":
            steps.extend(
                [
                    "Search the uploaded documents for relevant information.",
                    "Analyze the retrieved information.",
                    "Generate the final answer.",
                ]
            )
            return steps

        if route == "research":
            steps.extend(
                [
                    "Perform external web research.",
                    "Analyze the retrieved information.",
                    "Generate the final answer.",
                ]
            )
            return steps

        if route == "both":
            steps.extend(
                [
                    "Retrieve relevant information from the uploaded documents.",
                    "Perform external web research.",
                    "Analyze and compare both evidence sources.",
                    "Generate the final answer.",
                ]
            )
            return steps

        steps.extend(
            [
                "Search the uploaded documents for relevant information.",
                "Analyze the retrieved information.",
                "Generate the final answer.",
            ]
        )

        return steps

    # ======================================================
    # Planner
    # ======================================================

    def plan(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Analyze the user query and determine the execution
        route and memory-aware execution plan.

        Phase 17.2:
            Memory is available to the planner before the
            downstream RAG/research stages.
        """

        query = state["query"]

        # --------------------------------------------------
        # Retrieve memory already placed into state
        # --------------------------------------------------

        memory_evidence = self._get_memory_evidence(
            state
        )

        # --------------------------------------------------
        # Deterministic routing remains unchanged
        # --------------------------------------------------

        route = self._rule_based_route(
            query
        )

        # --------------------------------------------------
        # Create memory-aware plan
        # --------------------------------------------------

        steps = self._create_steps(
            route=route,
            memory_evidence=memory_evidence,
        )

        # --------------------------------------------------
        # Store planning context
        # --------------------------------------------------

        memory_summary = self._memory_summary(
            memory_evidence
        )

        # --------------------------------------------------
        # Return updated state
        # --------------------------------------------------

        return {
            **state,
            "plan": steps,
            "route": route,
            "current_step": "planning_complete",
            "planning_memory_context": memory_summary,
        }