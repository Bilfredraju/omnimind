from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class SynthesisAgent:
    """Generate the final answer from multi-source evidence."""

    def __init__(self):
        self.llm = GroqProvider()

    def run(self, state: AgentState) -> AgentState:
        query = state["query"]

        analysis = state.get(
            "analysis",
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
        # Check whether enough information exists
        # --------------------------------------------------

        if (
            not analysis
            and not rag_results
            and not research_results
        ):
            return {
                **state,
                "final_answer": (
                    "I don't have enough evidence "
                    "to answer this question."
                ),
                "current_step": "synthesis_complete",
            }

        # --------------------------------------------------
        # Build evidence and sources
        # --------------------------------------------------

        sources = []
        evidence_sections = []

        source_number = 1

        # --------------------------------------------------
        # Document / RAG evidence
        # --------------------------------------------------

        if rag_results:

            rag_evidence = []

            for result in rag_results:

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
[Source {source_number}]
Type: Document
Document: {source}
Page: {page}
Chunk: {chunk}

{text}
""".strip()
                )

                source_number += 1

            evidence_sections.append(
                "DOCUMENT EVIDENCE\n\n"
                + "\n\n".join(
                    rag_evidence
                )
            )

        # --------------------------------------------------
        # Web research evidence
        # --------------------------------------------------

        if research_results:

            web_evidence = []

            for result in research_results:

                title = result.get(
                    "title",
                    "Web source",
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
[Source {source_number}]
Type: Web
Title: {title}
URL: {url}

{snippet}
""".strip()
                )

                source_number += 1

            evidence_sections.append(
                "WEB RESEARCH EVIDENCE\n\n"
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

Generate the final answer to the user's question using
ONLY the analysis and evidence provided below.

USER QUESTION:
{query}

SELECTED ROUTE:
{route}

ANALYSIS:
{analysis}

EVIDENCE:
{evidence}

Rules:

1. Answer the user's question directly.
2. Use only the provided analysis and evidence.
3. Do not invent facts.
4. Do not introduce outside knowledge.
5. Clearly distinguish document evidence from web evidence.
6. If the route is "both", compare the two evidence sources
   when the user's question requires comparison.
7. If the evidence is insufficient, say so clearly.
8. Keep the answer concise but informative.
9. For document evidence, cite claims using:
   [Source N, Page X]
10. For web evidence, cite claims using:
    [Source N]
11. Do not fabricate citations.
12. Do not mention internal agents or the orchestration
    process.
13. Do not claim that a web search result proves something
    beyond the information contained in its provided snippet.
14. Prefer factual, evidence-grounded language.

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