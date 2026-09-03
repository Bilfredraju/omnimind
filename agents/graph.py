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

        result = self.planner.plan(state)

        return {
            "plan": result.get("plan", []),
            "route": result.get("route", "rag"),
            "current_step": result.get(
                "current_step",
                "planning_complete",
            ),
        }

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

        result = self.rag_agent.run(state)

        return {
            "rag_results": result.get(
                "rag_results",
                [],
            ),
            "current_step": result.get(
                "current_step",
                "rag_complete",
            ),
        }

    # ========================================================
    # RESEARCH NODE
    # ========================================================

    def research_node(
        self,
        state: AgentState,
    ) -> AgentState:

        result = self.research_agent.run(state)

        return {
            "research_results": result.get(
                "research_results",
                [],
            ),
            "sources": result.get(
                "sources",
                state.get("sources", []),
            ),
            "current_step": result.get(
                "current_step",
                "research_complete",
            ),
            "error": result.get(
                "error",
                "",
            ),
        }

    # ========================================================
    # ANALYSIS NODE
    # ========================================================

    def analysis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        result = self.analysis_agent.run(state)

        return {
            "analysis": result.get(
                "analysis",
                "",
            ),
            "current_step": result.get(
                "current_step",
                "analysis_complete",
            ),
        }

    # ========================================================
    # SYNTHESIS NODE
    # ========================================================

    def synthesis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        result = self.synthesis_agent.run(state)

        return {
            "final_answer": result.get(
                "final_answer",
                "",
            ),
            "sources": result.get(
                "sources",
                state.get("sources", []),
            ),
            "current_step": result.get(
                "current_step",
                "synthesis_complete",
            ),
        }

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
        """

        self.rag_agent.close()