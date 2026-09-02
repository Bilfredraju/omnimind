from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class AnalysisAgent:
    """Analyze evidence collected from RAG and web research."""

    def __init__(self):
        self.llm = GroqProvider()

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state["query"]

        rag_results = state.get(
            "rag_results",
            [],
        )

        research_results = state.get(
            "research_results",
            [],
        )

        # --------------------------------------------------
        # Check whether any evidence exists
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

        # --------------------------------------------------
        # Build RAG evidence
        # --------------------------------------------------

        rag_evidence = []

        for index, result in enumerate(
            rag_results,
            start=1,
        ):

            rag_evidence.append(
                f"""
[RAG Evidence {index}]
Source: {result.get("source", "Unknown")}
Page: {result.get("page", "Unknown")}
Chunk: {result.get("chunk", "Unknown")}
Score: {result.get("score", 0.0):.4f}

{result.get("text", "")}
""".strip()
            )

        rag_text = "\n\n".join(
            rag_evidence
        )

        if not rag_text:
            rag_text = "No document evidence available."

        # --------------------------------------------------
        # Build research evidence
        # --------------------------------------------------

        web_evidence = []

        for index, result in enumerate(
            research_results,
            start=1,
        ):

            web_evidence.append(
                f"""
[Web Evidence {index}]
Title: {result.get("title", "Unknown")}
URL: {result.get("url", "")}

{result.get("snippet", "")}
""".strip()
            )

        web_text = "\n\n".join(
            web_evidence
        )

        if not web_text:
            web_text = "No external web evidence available."

        # --------------------------------------------------
        # Analysis prompt
        # --------------------------------------------------

        prompt = f"""
You are the Analysis Agent in OmniMind.

Analyze the evidence collected from the user's documents
and/or external web research.

USER QUESTION:
{query}

==================================================
DOCUMENT / RAG EVIDENCE
==================================================

{rag_text}

==================================================
EXTERNAL WEB EVIDENCE
==================================================

{web_text}

==================================================
ANALYSIS RULES
==================================================

1. Use ONLY the evidence provided above.

2. Do NOT invent information.

3. Identify the facts that directly answer the
   user's question.

4. Clearly distinguish between:
   - information from the user's documents
   - information from external web research

5. When both sources are available:
   - compare them when relevant
   - identify similarities
   - identify differences
   - identify meaningful extensions or gaps

6. If the evidence conflicts, explicitly mention
   the conflict.

7. Do not treat a web source as proof merely because
   it appears in the search results.

8. Do not produce a polished final answer.

9. Produce concise analytical notes that the
   Synthesis Agent can use.

10. If one evidence source is unavailable, do not
    pretend that it exists.

ANALYSIS:
""".strip()

        analysis = self.llm.generate(
            prompt
        )

        return {
            **state,
            "analysis": analysis,
            "current_step": "analysis_complete",
        }