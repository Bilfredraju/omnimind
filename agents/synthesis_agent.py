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

            if not isinstance(metadata, dict):
                metadata = {}

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
            # --------------------------------------------------
            # Safely extract metadata
            # --------------------------------------------------

            metadata = result.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            # --------------------------------------------------
            # Safely extract structured citation
            # --------------------------------------------------

            citation = result.get(
                "citation",
                {},
            )

            if not isinstance(citation, dict):
                citation = {}

            # --------------------------------------------------
            # Citation ID
            #
            # Prefer the structured citation generated by the
            # retrieval layer. Fall back to deterministic numbering
            # for backward compatibility.
            # --------------------------------------------------

            citation_id = citation.get(
                "citation_id",
                f"[{index}]",
            )

            if not citation_id:
                citation_id = f"[{index}]"

            # --------------------------------------------------
            # Document source
            # --------------------------------------------------

            source = citation.get(
                "source",
                result.get(
                    "source",
                    metadata.get(
                        "source",
                        "Unknown document",
                    ),
                ),
            )

            # --------------------------------------------------
            # Document name
            # --------------------------------------------------

            document_name = citation.get(
                "document_name",
                result.get(
                    "document_name",
                    metadata.get(
                        "document_name",
                        source,
                    ),
                ),
            )

            # --------------------------------------------------
            # Page
            # --------------------------------------------------

            page = citation.get(
                "page",
                result.get(
                    "page",
                    metadata.get(
                        "page",
                        result.get(
                            "page_number",
                            metadata.get(
                                "page_number",
                                "Unknown",
                            ),
                        ),
                    ),
                ),
            )

            # --------------------------------------------------
            # Chunk ID
            # --------------------------------------------------

            chunk_id = citation.get(
                "chunk_id",
                result.get(
                    "chunk_id",
                    metadata.get(
                        "chunk_id",
                        "",
                    ),
                ),
            )

            # --------------------------------------------------
            # Scores
            # --------------------------------------------------

            score = result.get(
                "score",
                0.0,
            )

            rerank_score = result.get(
                "rerank_score",
            )

            hybrid_score = result.get(
                "hybrid_score",
            )

            # --------------------------------------------------
            # Evidence text
            # --------------------------------------------------

            text = result.get(
                "text",
                "",
            )

            # --------------------------------------------------
            # Structured source record
            # --------------------------------------------------

            source_record = {
                "citation_id": citation_id,
                "source": source,
                "document_name": document_name,
                "page": page,
                "chunk_id": chunk_id,
                "type": "document",
            }

            sources.append(source_record)

            # --------------------------------------------------
            # Evidence block for LLM
            # --------------------------------------------------

            rerank_score_text = (
                f"{rerank_score:.4f}"
                if isinstance(
                    rerank_score,
                    (int, float),
                )
                else "N/A"
            )

            hybrid_score_text = (
                f"{hybrid_score:.6f}"
                if isinstance(
                    hybrid_score,
                    (int, float),
                )
                else "N/A"
            )

            evidence.append(
                f"""
[RAG Source {citation_id}]
Type: Document
Document: {document_name}
Source: {source}
Page: {page}
Chunk ID: {chunk_id}
Retrieval Score: {score:.4f}
Rerank Score: {rerank_score_text}
Hybrid Score: {hybrid_score_text}

Evidence:
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

21. Document citations MUST use the exact citation
    identifier supplied with the document evidence.

    Examples:
    [1]
    [2]
    [3]

    The citation identifier refers to the corresponding
    supplied document/source/page metadata.

22. Web citations MUST use:
    [Web Source N]

23. Only cite sources actually present in the evidence.

24. Never invent citation identifiers, citation numbers,
    document names, source names, or page numbers.

25. When citing document evidence, place the exact
    citation identifier immediately after the relevant
    factual statement.

26. A document citation such as [1] must correspond to
    the document evidence block containing the same
    citation identifier.

27. Memory does NOT require fabricated citation numbers.
    Refer to previous conversation context naturally.

28. If evidence is insufficient, say so clearly.

29. Do not turn uncertainty into certainty.

30. Keep the answer concise but informative.

31. Use bullets, tables, or short paragraphs when
    appropriate.

32. Do not mention internal implementation details or
    orchestration.

33. Do not mention:
    - Planner Agent
    - RAG Agent
    - Research Agent
    - Analysis Agent
    - Synthesis Agent
    - LangGraph
    - MCP
    - internal orchestration

34. Return ONLY the final user-facing answer.

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