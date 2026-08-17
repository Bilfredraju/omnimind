import json

from models.llm.groq_provider import GroqProvider
from agents.state import AgentState


class PlannerAgent:
    """Create an execution plan for a user query."""

    def __init__(self):
        self.llm = GroqProvider()

    def plan(self, state: AgentState) -> AgentState:
        query = state["query"]

        prompt = f"""
You are the Planner Agent in an autonomous AI system
called OmniMind.

Analyze the user's request and create a short execution plan.

Available capabilities:
1. RAG Agent - searches the user's documents.
2. Research Agent - performs external research.
3. Analysis Agent - analyzes retrieved information.
4. Synthesis Agent - creates the final answer.

Rules:
- Return ONLY valid JSON.
- The JSON must contain a key called "steps".
- "steps" must be a list of short action descriptions.
- Use only the capabilities that are actually needed.
- Do not answer the user's question.
- Create between 1 and 5 steps.

User request:
{query}

Return:
{{
    "steps": [
        "..."
    ]
}}
""".strip()

        response = self.llm.generate(prompt)

        try:
            parsed = json.loads(response)

            steps = parsed.get("steps", [])

            if not isinstance(steps, list):
                raise ValueError(
                    "Planner steps must be a list."
                )

        except (
            json.JSONDecodeError,
            ValueError,
        ):
            # Safe fallback if the LLM doesn't return valid JSON.
            steps = [
                "Search relevant documents",
                "Analyze the retrieved information",
                "Generate the final answer",
            ]

        return {
            **state,
            "plan": steps,
            "current_step": "planning_complete",
        }