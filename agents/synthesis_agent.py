from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class SynthesisAgent:
    """
    Generate the final user-facing answer from:

    1. Semantic long-term memory
    2. Temporal / consolidated memory
    3. Uploaded document / RAG evidence
    4. External web research
    5. Memory-aware analysis

    Historical memory must remain distinct from current knowledge.
    """

    def __init__(self):
        self.llm = GroqProvider()

    # ======================================================
    # MEMORY HELPERS
    # ======================================================

    def _build_semantic_memory_evidence(
        self,
        memory_results: list[dict],
    ) -> str:

        if not memory_results:
            return ""

        evidence = []

        for index, result in enumerate(
            memory_results,
            start=1,
        ):
            score = result.get(
                "score",
                result.get("ranking_score", 0.0),
            )

            text = result.get(
                "text",
                result.get(
                    "content",
                    result.get("memory", ""),
                ),
            )

            metadata = result.get(
                "metadata",
                {},
            )

            memory_type = metadata.get(
                "type",
                "unknown",
            )

            created_at = metadata.get(
                "created_at",
                "unknown",
            )

            evidence.append(
                f"""
[Semantic Memory {index}]
Memory Type: {memory_type}
Created At: {created_at}
Relevance Score: {score:.4f}

{text}
""".strip()
            )

        return "\n\n".join(evidence)

    def _build_temporal_memory_evidence(
        self,
        temporal_results: list[dict],
    ) -> str:

        if not temporal_results:
            return ""

        evidence = []

        for index, result in enumerate(
            temporal_results,
            start=1,
        ):
            topic = result.get(
                "topic",
                "Unknown topic",
            )

            summary = result.get(
                "summary",
                "",
            )

            score = result.get(
                "score",
                result.get(
                    "retrieval_score",
                    0.0,
                ),
            )

            current_memory_id = result.get(
                "current_memory_id",
                "",
            )

            historical_memory_ids = result.get(
                "historical_memory_ids",
                [],
            )

            timeline = result.get(
                "timeline",
                [],
            )

            timeline_text = []

            for event in timeline:
                memory_id = event.get(
                    "memory_id",
                    "",
                )

                text = event.get(
                    "text",
                    "",
                )

                status = event.get(
                    "status",
                    "unknown",
                )

                timestamp = event.get(
                    "timestamp",
                    event.get(
                        "created_at",
                        "unknown",
                    ),
                )

                timeline_text.append(
                    f"- Memory ID: {memory_id}\n"
                    f"  Status: {status}\n"
                    f"  Time: {timestamp}\n"
                    f"  Text: {text}"
                )

            timeline_section = (
                "\n".join(timeline_text)
                if timeline_text
                else "No timeline events supplied."
            )

            evidence.append(
                f"""
[Temporal Memory {index}]
Topic: {topic}
Retrieval Score: {score:.4f}

Summary:
{summary}

Current Memory ID:
{current_memory_id}

Historical Memory IDs:
{historical_memory_ids}

Timeline:
{timeline_section}
""".strip()
            )

        return "\n\n".join(evidence)

    # ======================================================
    # DOCUMENT / RAG
    # ======================================================

    def _build_rag_evidence(
        self,
        rag_results: list[dict],
    ) -> tuple[str, list[dict]]:

        if not rag_results:
            return "", []

        evidence = []
        sources = []

        for index, result in enumerate(
            rag_results,
            start=1,
        ):
            source = result.get(
                "source",
                "Unknown document",
            )

            page = result.get(
                "page",
                "Unknown",
            )

            chunk = result.get(
                "chunk",
                "Unknown",
            )

            score = result.get(
                "score",
                0.0,
            )

            text = result.get(
                "text",
                "",
            )

            sources.append(
                {
                    "source": source,
                    "page": page,
                    "chunk": chunk,
                    "type": "document",
                }
            )

            evidence.append(
                f"""
[RAG Source {index}]
Type: Document
Document: {source}
Page: {page}
Chunk: {chunk}
Retrieval Score: {score:.4f}

{text}
""".strip()
            )

        return "\n\n".join(evidence), sources

    # ======================================================
    # WEB RESEARCH
    # ======================================================

    def _build_web_evidence(
        self,
        research_results: list[dict],
    ) -> tuple[str, list[dict]]:

        if not research_results:
            return "", []

        evidence = []
        sources = []

        for index, result in enumerate(
            research_results,
            start=1,
        ):
            title = result.get(
                "title",
                "Unknown web source",
            )

            url = result.get(
                "url",
                "",
            )

            snippet = result.get(
                "snippet",
                "",
            )

            sources.append(
                {
                    "source": title,
                    "url": url,
                    "type": "web",
                }
            )

            evidence.append(
                f"""
[Web Source {index}]
Type: External Web Source
Title: {title}
URL: {url}

Snippet:
{snippet}
""".strip()
            )

        return "\n\n".join(evidence), sources

    # ======================================================
    # MAIN
    # ======================================================

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state["query"]

        analysis = state.get(
            "analysis",
            "",
        )

        memory_results = state.get(
            "memory_results",
            [],
        )

        memory_context = state.get(
            "memory_context",
            "",
        )

        temporal_memory_results = state.get(
            "temporal_memory_results",
            [],
        )

        temporal_memory_context = state.get(
            "temporal_memory_context",
            "",
        )

        temporal_intent = state.get(
            "temporal_intent",
            {},
        )

        planning_memory_context = state.get(
            "planning_memory_context",
            "",
        )

        research_memory_context = state.get(
            "research_memory_context",
            "",
        )

        rag_results = state.get(
            "rag_results",
            [],
        )

        research_results = state.get(
            "research_results",
            [],
        )

        route = state.get(
            "route",
            "rag",
        )

        # --------------------------------------------------
        # Build memory evidence
        # --------------------------------------------------

        semantic_memory_evidence = (
            self._build_semantic_memory_evidence(
                memory_results
            )
        )

        temporal_memory_evidence = (
            self._build_temporal_memory_evidence(
                temporal_memory_results
            )
        )

        # --------------------------------------------------
        # Build RAG and web evidence
        # --------------------------------------------------

        rag_evidence, rag_sources = (
            self._build_rag_evidence(
                rag_results
            )
        )

        web_evidence, web_sources = (
            self._build_web_evidence(
                research_results
            )
        )

        sources = (
            rag_sources
            + web_sources
        )

        # --------------------------------------------------
        # Fallback context
        # --------------------------------------------------

        if (
            not analysis
            and not semantic_memory_evidence
            and not temporal_memory_evidence
            and not rag_evidence
            and not web_evidence
        ):
            return {
                **state,
                "final_answer": (
                    "I don't have enough evidence "
                    "to answer this question."
                ),
                "sources": [],
                "current_step": "synthesis_complete",
            }

        # --------------------------------------------------
        # Build combined evidence
        # --------------------------------------------------

        evidence_sections = []

        if semantic_memory_evidence:
            evidence_sections.append(
                "==================================================\n"
                "SEMANTIC LONG-TERM MEMORY\n"
                "==================================================\n\n"
                + semantic_memory_evidence
            )

        if memory_context:
            evidence_sections.append(
                "==================================================\n"
                "SEMANTIC MEMORY CONTEXT\n"
                "==================================================\n\n"
                + memory_context
            )

        if temporal_memory_evidence:
            evidence_sections.append(
                "==================================================\n"
                "TEMPORAL / CONSOLIDATED MEMORY\n"
                "==================================================\n\n"
                + temporal_memory_evidence
            )

        if temporal_memory_context:
            evidence_sections.append(
                "==================================================\n"
                "TEMPORAL MEMORY CONTEXT\n"
                "==================================================\n\n"
                + temporal_memory_context
            )

        if rag_evidence:
            evidence_sections.append(
                "==================================================\n"
                "DOCUMENT / RAG EVIDENCE\n"
                "==================================================\n\n"
                + rag_evidence
            )

        if web_evidence:
            evidence_sections.append(
                "==================================================\n"
                "EXTERNAL WEB EVIDENCE\n"
                "==================================================\n\n"
                + web_evidence
            )

        evidence = "\n\n".join(
            evidence_sections
        )

        # --------------------------------------------------
        # Synthesis prompt
        # --------------------------------------------------

        prompt = f"""
You are the final answer generation component in OmniMind.

Your task is to answer the user's question using ONLY the
analysis and evidence supplied below.

USER QUESTION:
{query}

SELECTED ROUTE:
{route}

==================================================
TEMPORAL INTENT
==================================================

{temporal_intent}

==================================================
ANALYSIS
==================================================

{analysis}

==================================================
PLANNING MEMORY CONTEXT
==================================================

{planning_memory_context}

==================================================
RESEARCH MEMORY CONTEXT
==================================================

{research_memory_context}

==================================================
EVIDENCE
==================================================

{evidence}

==================================================
SYNTHESIS RULES
==================================================

1. Answer the user's question directly.

2. Use ONLY the supplied analysis and evidence.

3. Do NOT use outside knowledge.

4. Do NOT invent facts, dates, sources, URLs,
   citations, technical details, or conclusions.

5. Semantic long-term memory represents information
   remembered from previous conversations.

6. Temporal / consolidated memory represents information
   organized around topics, timelines, historical states,
   and current states.

7. Historical memory and current memory are NOT the same.

8. If the user asks about a past time such as:
   - yesterday
   - last week
   - last month
   - 3 months ago
   - previously
   - earlier
   - at that time

   prioritize evidence belonging to that historical period.

9. If a historical decision was later changed, preserve
   the historical decision when answering a historical
   question.

10. Example:
    If historical memory says Qdrant was selected and
    current memory says PostgreSQL is now selected:

    Question:
    "What did I decide 3 months ago?"

    Correct:
    "You decided to use Qdrant 3 months ago."

    You may additionally explain that the decision later
    changed to PostgreSQL if that is explicitly supported.

11. Never replace a historical fact with a newer fact
    merely because the newer fact exists.

12. If the user asks what is CURRENTLY true, prefer
    explicitly supported current evidence.

13. If memory contains conflicting information:
    - preserve the historical information
    - identify the current information when explicitly
      supported
    - explain the change clearly
    - never silently rewrite history

14. When document evidence is available, treat it as
    evidence from the user's uploaded material.

15. When web evidence is available, use only information
    contained in the supplied web results and snippets.

16. Do not claim a web source proves something that is
    not actually present in the supplied evidence.

17. For route "rag":
    primarily answer from document evidence.

18. For route "research":
    primarily answer from external web evidence.

19. For route "both":
    use both document and external web evidence and
    distinguish their roles.

20. When comparing sources, clearly identify agreements,
    differences, and limitations supported by evidence.

21. Document citations MUST use:
    [Source N, Page X]

22. Web citations MUST use:
    [Web Source N]

23. Only cite sources actually present in the evidence.

24. Do not fabricate citation numbers.

25. Memory does NOT require fabricated citation numbers.
    Refer to previous conversation context naturally.

26. If evidence is insufficient, say so clearly.

27. Do not turn uncertainty into certainty.

28. Keep the answer concise but informative.

29. Use bullets, tables, or short paragraphs when
    appropriate.

30. Do not mention internal implementation details or
    orchestration.

31. Do not mention:
    - Planner Agent
    - RAG Agent
    - Research Agent
    - Analysis Agent
    - Synthesis Agent
    - LangGraph
    - MCP
    - internal orchestration

32. Return ONLY the final user-facing answer.

FINAL ANSWER:
""".strip()

        final_answer = self.llm.generate(
            prompt
        )

        return {
            **state,
            "final_answer": final_answer,
            "sources": sources,
            "current_step": "synthesis_complete",
        }

    def close(self):
        close = getattr(
            self.llm,
            "close",
            None,
        )

        if callable(close):
            close()