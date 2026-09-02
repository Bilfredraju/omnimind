from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.planner import PlannerAgent
from agents.rag_agent import RAGAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.synthesis_agent import SynthesisAgent


class OmniMindGraph:
    """
    LangGraph orchestration for OmniMind agents.

    Routes:
        rag      -> RAG -> Analysis -> Synthesis
        research -> Research -> Analysis -> Synthesis
        both     -> RAG -> Research -> Analysis -> Synthesis
    """

    def __init__(
        self,
        pdf_path: str,
    ):

        self.planner = PlannerAgent()

        self.rag_agent = RAGAgent(
            pdf_path=pdf_path
        )

        self.research_agent = ResearchAgent()

        self.analysis_agent = AnalysisAgent()

        self.synthesis_agent = SynthesisAgent()

        # ----------------------------------------------------
        # Build graph
        # ----------------------------------------------------

        graph = StateGraph(AgentState)

        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # START -> Planner
        # ----------------------------------------------------

        graph.add_edge(
            START,
            "planner",
        )

        # ----------------------------------------------------
        # Planner -> RAG / Research
        # ----------------------------------------------------

        graph.add_conditional_edges(
            "planner",
            self.route_from_planner,
            {
                "rag": "rag",
                "research": "research",
                "both": "rag",
            },
        )

        # ----------------------------------------------------
        # RAG -> Analysis / Research
        # ----------------------------------------------------

        graph.add_conditional_edges(
            "rag",
            self.route_after_rag,
            {
                "analysis": "analysis",
                "research": "research",
            },
        )

        # ----------------------------------------------------
        # Research -> Analysis
        # ----------------------------------------------------

        graph.add_edge(
            "research",
            "analysis",
        )

        # ----------------------------------------------------
        # Analysis -> Synthesis
        # ----------------------------------------------------

        graph.add_edge(
            "analysis",
            "synthesis",
        )

        # ----------------------------------------------------
        # Synthesis -> END
        # ----------------------------------------------------

        graph.add_edge(
            "synthesis",
            END,
        )

        # ----------------------------------------------------
        # Compile
        # ----------------------------------------------------

        self.graph = graph.compile()

    # ========================================================
    # PLANNER
    # ========================================================

    def planner_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.planner.plan(
            state
        )

    # ========================================================
    # PLANNER ROUTING
    # ========================================================

    def route_from_planner(
        self,
        state: AgentState,
    ) -> str:

        route = state.get(
            "route",
            "rag",
        )

        if route == "research":
            return "research"

        if route == "both":
            return "both"

        return "rag"

    # ========================================================
    # ROUTE AFTER RAG
    # ========================================================

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

    # ========================================================
    # RAG NODE
    # ========================================================

    def rag_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.rag_agent.run(
            state
        )

    # ========================================================
    # RESEARCH NODE
    # ========================================================

    def research_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.research_agent.run(
            state
        )

    # ========================================================
    # ANALYSIS NODE
    # ========================================================

    def analysis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.analysis_agent.run(
            state
        )

    # ========================================================
    # SYNTHESIS NODE
    # ========================================================

    def synthesis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.synthesis_agent.run(
            state
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.graph.invoke(
            state
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self):
        """
        Close resources owned by the graph.

        The RAG agent currently delegates retrieval
        to the Document MCP server, so its close()
        method is a compatibility no-op.
        """

        self.rag_agent.close()