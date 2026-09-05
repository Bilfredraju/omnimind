from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between OmniMind agents.
    """

    # User query
    query: str

    # Planning
    plan: list[str]
    current_step: str
    route: str

    # Memory
    memory_results: list[dict]
    memory_context: str
    memory_written: bool
    memory_count: int

    # Retrieval / research
    research_results: list[dict]
    rag_results: list[dict]

    # Reasoning
    analysis: str

    # Final response
    final_answer: str
    sources: list[dict]

    # Errors
    error: str