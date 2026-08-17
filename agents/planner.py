import json

from models.llm.groq_provider import GroqProvider
from agents.state import AgentState


class PlannerAgent:
    """Create an execution plan and route for a user query."""

    def __init__(self):
        self.llm = GroqProvider()

    def plan(self, state: AgentState) -> AgentState:
        """Analyze the query and determine the required route."""

        query = state["query"]

        prompt = f"""
You are the Planner Agent in an autonomous AI system
called OmniMind.

Analyze the user's request and determine which capabilities
are required.

Available capabilities:

1. RAG Agent
   Searches the user's uploaded documents.

2. Research Agent
   Performs external web research.

3. Analysis Agent
   Analyzes retrieved information.

4. Synthesis Agent
   Produces the final answer.

Choose exactly one route:

"rag"
Use when the answer should come primarily from the
user's documents.

"research"
Use when the user explicitly needs external or recent
information.

"both"
Use when the user needs information from the user's
documents AND external research.

Rules:
- Return ONLY valid JSON.
- Include "route".
- Include "steps".
- route must be exactly one of:
  rag, research, both.
- steps must be a list.
- Do not answer the user's question.

User request:
{query}

Return:

{{
    "route": "rag",
    "steps": [
        "..."
    ]
}}
""".strip()

        response = self.llm.generate(prompt)

        try:
            parsed = json.loads(response)

            steps = parsed.get(
                "steps",
                [],
            )

            route = parsed.get(
                "route",
                "rag",
            )

            # Validate route
            if route not in {
                "rag",
                "research",
                "both",
            }:
                route = "rag"

            # Validate steps
            if not isinstance(steps, list):
                raise ValueError(
                    "Planner steps must be a list."
                )

        except (
            json.JSONDecodeError,
            ValueError,
        ):
            # Safe fallback if the LLM doesn't return
            # valid JSON.
            route = "rag"

            steps = [
                "Search relevant documents",
                "Analyze the retrieved information",
                "Generate the final answer",
            ]

        return {
            **state,
            "plan": steps,
            "route": route,
            "current_step": "planning_complete",
        }