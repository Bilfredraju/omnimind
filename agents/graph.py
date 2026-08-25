from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.planner import PlannerAgent
from agents.rag_agent import RAGAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.synthesis_agent import SynthesisAgent


class OmniMindGraph:
    """LangGraph orchestration for OmniMind agents."""

    def __init__(self, pdf_path: str):

        self.planner = PlannerAgent()

        self.rag_agent = RAGAgent(
            pdf_path=pdf_path
        )

        self.research_agent = ResearchAgent()

        self.analysis_agent = AnalysisAgent()

        self.synthesis_agent = SynthesisAgent()

        graph = StateGraph(AgentState)

        # --------------------------------------------------
        # Nodes
        # --------------------------------------------------

        graph.add_node(
            "planner",
            self.planner_node,
        )

        graph.add_node(
            "rag",
            self.rag_node,
        )

        graph.add_node(
            "research",
            self.research_node,
        )

        graph.add_node(
            "analysis",
            self.analysis_node,
        )

        graph.add_node(
            "synthesis",
            self.synthesis_node,
        )

        # --------------------------------------------------
        # Initial route
        # --------------------------------------------------

        graph.add_edge(
            START,
            "planner",
        )

        graph.add_conditional_edges(
            "planner",
            self.route_from_planner,
            {
                "rag": "rag",
                "research": "research",
                "both": "rag",
            },
        )

        # --------------------------------------------------
        # RAG route
        # --------------------------------------------------

        graph.add_conditional_edges(
            "rag",
            self.route_after_rag,
            {
                "analysis": "analysis",
                "research": "research",
            },
        )

        # --------------------------------------------------
        # Research route
        # --------------------------------------------------

        graph.add_edge(
            "research",
            "analysis",
        )

        # --------------------------------------------------
        # Final pipeline
        # --------------------------------------------------

        graph.add_edge(
            "analysis",
            "synthesis",
        )

        graph.add_edge(
            "synthesis",
            END,
        )

        self.graph = graph.compile()

    # ======================================================
    # Planner
    # ======================================================

    def planner_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.planner.plan(state)

    # ======================================================
    # Routing
    # ======================================================

    def route_from_planner(
        self,
        state: AgentState,
    ) -> str:

        route = state.get(
            "route",
            "rag",
        )

        if route not in {
            "rag",
            "research",
            "both",
        }:
            return "rag"

        return route

    def route_after_rag(
        self,
        state: AgentState,
    ) -> str:

        route = state.get(
            "route",
            "rag",
        )

        if route == "both":
            return "research"

        return "analysis"

    # ======================================================
    # RAG
    # ======================================================

    def rag_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.rag_agent.run(state)

    # ======================================================
    # Research
    # ======================================================

    def research_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.research_agent.run(state)

    # ======================================================
    # Analysis
    # ======================================================

    def analysis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.analysis_agent.run(state)

    # ======================================================
    # Synthesis
    # ======================================================

    def synthesis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.synthesis_agent.run(state)

    # ======================================================
    # Run
    # ======================================================

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.graph.invoke(state)

    # ======================================================
    # Cleanup
    # ======================================================

    def close(self):

        self.rag_agent.close()