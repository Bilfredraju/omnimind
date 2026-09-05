from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class SynthesisAgent:
    """Generate the final answer from memory, document, and web evidence."""

    def __init__(self):
        self.llm = GroqProvider()

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
        # Check for available evidence
        # --------------------------------------------------

        if (
            not analysis
            and not memory_results
            and not rag_results
            and not research_results
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
        # Build evidence
        # --------------------------------------------------

        evidence_sections = []

        sources = []

        # ==================================================
        # LONG-TERM MEMORY EVIDENCE
        # ==================================================

        if memory_results:

            memory_evidence = []

            for index, result in enumerate(
                memory_results,
                start=1,
            ):

                score = result.get(
                    "score",
                    0.0,
                )

                text = result.get(
                    "text",
                    "",
                )

                memory_evidence.append(
                    f"""
[Memory Source {index}]
Type: Previous Conversation
Relevance Score: {score:.4f}

{text}
""".strip()
                )

            evidence_sections.append(
                "==================================================\n"
                "LONG-TERM MEMORY EVIDENCE\n"
                "==================================================\n\n"
                + "\n\n".join(
                    memory_evidence
                )
            )

        # ==================================================
        # DOCUMENT / RAG EVIDENCE
        # ==================================================

        if rag_results:

            rag_evidence = []

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

                rag_evidence.append(
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

            evidence_sections.append(
                "==================================================\n"
                "DOCUMENT / RAG EVIDENCE\n"
                "==================================================\n\n"
                + "\n\n".join(
                    rag_evidence
                )
            )

        # ==================================================
        # WEB RESEARCH EVIDENCE
        # ==================================================

        if research_results:

            web_evidence = []

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

                web_evidence.append(
                    f"""
[Web Source {index}]
Type: External Web Source
Title: {title}
URL: {url}

Snippet:
{snippet}
""".strip()
                )

            evidence_sections.append(
                "==================================================\n"
                "EXTERNAL WEB EVIDENCE\n"
                "==================================================\n\n"
                + "\n\n".join(
                    web_evidence
                )
            )

        evidence = "\n\n".join(
            evidence_sections
        )

        # --------------------------------------------------
        # Synthesis prompt
        # --------------------------------------------------

        prompt = f"""
You are the Synthesis Agent in OmniMind.

Your task is to produce the final answer to the user's
question using ONLY the analysis and evidence supplied
below.

USER QUESTION:
{query}

SELECTED ROUTE:
{route}

==================================================
ANALYSIS
==================================================

{analysis}

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

4. Do NOT invent facts, sources, URLs, citations,
   dataset names, dates, or technical details.

5. Previous conversation memory represents historical
   context from earlier OmniMind interactions.

6. When memory evidence is available:
   - use it to answer questions about previous
     conversations, decisions, plans, or statements
   - make clear that the information comes from
     previous conversation context when appropriate
   - do not assume that an old statement is still
     current if newer evidence contradicts it

7. When document evidence is available, treat it as
   evidence from the user's uploaded material.

8. When web evidence is available, treat it only as
   information contained in the supplied search results
   and snippets.

9. Do NOT claim that a web snippet proves information
   that is not actually contained in that snippet.

10. For route "rag":
    - Answer primarily from the document evidence.

11. For route "research":
    - Answer primarily from the external web evidence.
    - Make clear when the supplied web evidence is
      limited.

12. For route "both":
    - Use both document and web evidence.
    - Explicitly distinguish what comes from the
      document and what comes from external research.
    - Compare them when the question asks for a
      comparison.

13. If memory and current evidence disagree:
    - identify the disagreement
    - prefer newer/current evidence when it is explicitly
      supported
    - do not silently rewrite historical memory

14. If the evidence is insufficient, say so clearly.

15. If the evidence sources disagree, explain the
    disagreement instead of choosing one without evidence.

16. Keep the final answer concise but informative.

17. Use readable formatting such as short paragraphs,
    bullet points, or tables when appropriate.

18. Document citations MUST use:
    [Source N, Page X]

19. Web citations MUST use:
    [Web Source N]

20. Memory does NOT require fabricated citation numbers.
    Refer to it naturally as previous conversation context.

21. Only cite sources that actually appear in the
    supplied evidence.

22. Do not fabricate citation numbers.

23. Do not mention:
    - Planner Agent
    - RAG Agent
    - Research Agent
    - Analysis Agent
    - Synthesis Agent
    - LangGraph
    - MCP
    - internal orchestration

24. Return ONLY the final user-facing answer.

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