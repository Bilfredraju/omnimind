from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class AnalysisAgent:
    """Analyze evidence collected from memory, RAG, and web research."""

    def __init__(self):
        self.llm = GroqProvider()

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state["query"]

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

        # --------------------------------------------------
        # Check whether any evidence exists
        # --------------------------------------------------

        if (
            not memory_results
            and not rag_results
            and not research_results
        ):

            return {
                **state,
                "analysis": (
                    "No evidence was available "
                    "for analysis."
                ),
                "current_step": "analysis_complete",
            }

        # --------------------------------------------------
        # Build memory evidence
        # --------------------------------------------------

        memory_evidence = []

        for index, result in enumerate(
            memory_results,
            start=1,
        ):

            memory_evidence.append(
                f"""
[Memory Evidence {index}]
Relevance Score: {result.get("score", 0.0):.4f}

{result.get("text", "")}
""".strip()
            )

        memory_text = "\n\n".join(
            memory_evidence
        )

        if not memory_text:
            memory_text = "No relevant memory evidence available."

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

Analyze the evidence collected from the user's long-term
memory, documents, and/or external web research.

USER QUESTION:
{query}

==================================================
LONG-TERM MEMORY EVIDENCE
==================================================

{memory_text}

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
   - information from long-term memory
   - information from the user's documents
   - information from external web research

5. Memory evidence represents information from
   previous OmniMind conversations. Treat it as
   historical context, not automatically as current truth.

6. When memory and current evidence are both available:
   - compare them when relevant
   - identify whether previous decisions or statements
     are consistent with current evidence
   - identify meaningful changes or conflicts

7. When both document and web evidence are available:
   - compare them when relevant
   - identify similarities
   - identify differences
   - identify meaningful extensions or gaps

8. If the evidence conflicts, explicitly mention
   the conflict.

9. Do not treat a web source as proof merely because
   it appears in the search results.

10. Do not produce a polished final answer.

11. Produce concise analytical notes that the
    Synthesis Agent can use.

12. If one evidence source is unavailable, do not
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