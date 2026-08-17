from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class AnalysisAgent:
    """Analyze evidence collected by other agents."""

    def __init__(self):
        self.llm = GroqProvider()

    def run(self, state: AgentState) -> AgentState:
        query = state["query"]
        rag_results = state.get("rag_results", [])

        if not rag_results:
            return {
                **state,
                "analysis": (
                    "No RAG evidence was available "
                    "for analysis."
                ),
                "current_step": "analysis_complete",
            }

        evidence_text = []

        for index, result in enumerate(
            rag_results,
            start=1,
        ):
            evidence_text.append(
                f"""
Evidence {index}
Source: {result["source"]}
Page: {result["page"]}
Chunk: {result["chunk"]}
Score: {result["score"]:.4f}

{result["text"]}
""".strip()
            )

        evidence = "\n\n".join(evidence_text)

        prompt = f"""
You are the Analysis Agent in OmniMind.

Analyze the evidence provided below and determine what
information is relevant to the user's question.

User question:
{query}

Evidence:
{evidence}

Rules:
1. Use only the provided evidence.
2. Do not invent information.
3. Identify the key facts relevant to the question.
4. If multiple pieces of evidence agree, combine them.
5. If evidence conflicts, explicitly mention the conflict.
6. Do not produce a polished final answer.
7. Produce concise analytical notes for the Synthesis Agent.

Analysis:
""".strip()

        analysis = self.llm.generate(prompt)

        return {
            **state,
            "analysis": analysis,
            "current_step": "analysis_complete",
        }