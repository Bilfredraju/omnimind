from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class SynthesisAgent:
    """Generate the final answer from agent-produced evidence."""

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

        if not analysis and not rag_results:
            return {
                **state,
                "final_answer": (
                    "I don't have enough evidence "
                    "to answer this question."
                ),
                "current_step": "synthesis_complete",
            }

        sources = []

        evidence_text = []

        for index, result in enumerate(
            rag_results,
            start=1,
        ):
            source = result["source"]
            page = result["page"]
            chunk = result["chunk"]

            sources.append(
                {
                    "source": source,
                    "page": page,
                    "chunk": chunk,
                }
            )

            evidence_text.append(
                f"""
[Source {index}]
Document: {source}
Page: {page}
Chunk: {chunk}

{result["text"]}
""".strip()
            )

        evidence = "\n\n".join(
            evidence_text
        )

        prompt = f"""
You are the Synthesis Agent in OmniMind.

Generate the final answer to the user's question using
ONLY the analysis and evidence provided below.

USER QUESTION:
{query}

ANALYSIS:
{analysis}

EVIDENCE:
{evidence}

Rules:
1. Answer the user's question directly.
2. Use only the provided evidence.
3. Do not invent facts.
4. Do not introduce outside knowledge.
5. If the evidence is insufficient, say so clearly.
6. Keep the answer concise but informative.
7. Cite claims using [Source N, Page X] format.
8. Do not mention internal agents or the orchestration process.

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