from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.planner import PlannerAgent
from agents.rag_agent import RAGAgent
from agents.analysis_agent import AnalysisAgent
from agents.synthesis_agent import SynthesisAgent


class OmniMindGraph:
    """LangGraph orchestration for OmniMind agents."""

    def __init__(self, pdf_path: str):

        self.planner = PlannerAgent()

        self.rag_agent = RAGAgent(
            pdf_path=pdf_path
        )

        self.analysis_agent = AnalysisAgent()

        self.synthesis_agent = SynthesisAgent()

        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node(
            "planner",
            self.planner_node,
        )

        graph.add_node(
            "rag",
            self.rag_node,
        )

        graph.add_node(
            "analysis",
            self.analysis_node,
        )

        graph.add_node(
            "synthesis",
            self.synthesis_node,
        )

        # Define workflow
        graph.add_edge(
            START,
            "planner",
        )

        graph.add_edge(
            "planner",
            "rag",
        )

        graph.add_edge(
            "rag",
            "analysis",
        )

        graph.add_edge(
            "analysis",
            "synthesis",
        )

        graph.add_edge(
            "synthesis",
            END,
        )

        self.graph = graph.compile()

    def planner_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.planner.plan(state)

    def rag_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.rag_agent.run(state)

    def analysis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.analysis_agent.run(state)

    def synthesis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.synthesis_agent.run(state)

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.graph.invoke(state)

    def close(self):
        """Release resources."""

        self.rag_agent.close()