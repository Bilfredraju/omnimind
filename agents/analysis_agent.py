from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class AnalysisAgent:
    """Analyze evidence collected by other agents."""

    def __init__(self):
        self.llm = GroqProvider()

    def run(self, state: AgentState) -> AgentState:
        query = state["query"]

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
        # Check whether any evidence is available
        # --------------------------------------------------

        if not rag_results and not research_results:
            return {
                **state,
                "analysis": (
                    "No evidence was available "
                    "for analysis."
                ),
                "current_step": "analysis_complete",
            }

        evidence_sections = []

        # --------------------------------------------------
        # RAG evidence
        # --------------------------------------------------

        if rag_results:

            rag_evidence = []

            for index, result in enumerate(
                rag_results,
                start=1,
            ):

                rag_evidence.append(
                    f"""
Document Evidence {index}
Source: {result.get("source", "Unknown")}
Page: {result.get("page", "Unknown")}
Chunk: {result.get("chunk", "Unknown")}
Score: {result.get("score", 0.0):.4f}

{result.get("text", "")}
""".strip()
                )

            evidence_sections.append(
                "DOCUMENT EVIDENCE\n\n"
                + "\n\n".join(rag_evidence)
            )

        # --------------------------------------------------
        # Web research evidence
        # --------------------------------------------------

        if research_results:

            web_evidence = []

            for index, result in enumerate(
                research_results,
                start=1,
            ):

                web_evidence.append(
                    f"""
Web Evidence {index}
Title: {result.get("title", "Unknown")}
URL: {result.get("url", "")}

{result.get("snippet", "")}
""".strip()
                )

            evidence_sections.append(
                "WEB RESEARCH EVIDENCE\n\n"
                + "\n\n".join(web_evidence)
            )

        evidence = "\n\n" + (
            "\n\n".join(evidence_sections)
        )

        # --------------------------------------------------
        # Analysis prompt
        # --------------------------------------------------

        prompt = f"""
You are the Analysis Agent in OmniMind.

Analyze the evidence provided below and determine what
information is relevant to the user's question.

User question:
{query}

Selected route:
{route}

Evidence:
{evidence}

Rules:

1. Use only the provided evidence.
2. Do not invent information.
3. Identify the key facts relevant to the question.
4. Distinguish between document evidence and web evidence.
5. If multiple pieces of evidence agree, combine them.
6. If evidence conflicts, explicitly mention the conflict.
7. For web evidence, do not treat the search snippet as
   proof beyond what it actually states.
8. If the route is "both", compare the document evidence
   with the web research evidence.
9. Do not produce a polished final answer.
10. Produce concise analytical notes for the
    Synthesis Agent.

Analysis:
""".strip()

        analysis = self.llm.generate(
            prompt
        )

        return {
            **state,
            "analysis": analysis,
            "current_step": "analysis_complete",
        }