from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between OmniMind agents.
    """

    query: str

    plan: list[str]

    current_step: str

    research_results: list[dict]

    rag_results: list[dict]

    analysis: str

    final_answer: str

    sources: list[dict]

    error: str