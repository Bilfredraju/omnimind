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
        # START → Planner
        # --------------------------------------------------

        graph.add_edge(
            START,
            "planner",
        )

        # --------------------------------------------------
        # Planner → RAG / Research
        # --------------------------------------------------

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
        # RAG → Analysis / Research
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
        # Research → Analysis
        # --------------------------------------------------

        graph.add_edge(
            "research",
            "analysis",
        )

        # --------------------------------------------------
        # Analysis → Synthesis
        # --------------------------------------------------

        graph.add_edge(
            "analysis",
            "synthesis",
        )

        # --------------------------------------------------
        # Synthesis → END
        # --------------------------------------------------

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
    # Planner Routing
    # ======================================================

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

    # ======================================================
    # Route after RAG
    # ======================================================

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

        result = self.rag_agent.run(
            state
        )

        return {
            **result,
            "route": state.get(
                "route",
                "rag",
            ),
        }

    # ======================================================
    # Research
    # ======================================================

    def research_node(
        self,
        state: AgentState,
    ) -> AgentState:

        print(
            "\n[DEBUG] Research node entered"
        )

        print(
            "[DEBUG] Route:",
            state.get("route"),
        )

        print(
            "[DEBUG] RAG results before research:",
            len(
                state.get(
                    "rag_results",
                    [],
                )
            ),
        )

        result = self.research_agent.run(
            state
        )

        research_results = result.get(
            "research_results",
            [],
        )

        rag_results = state.get(
            "rag_results",
            [],
        )

        print(
            "[DEBUG] Research results returned:",
            len(research_results),
        )

        print(
            "[DEBUG] RAG results preserved:",
            len(rag_results),
        )

        return {
            **state,
            "research_results": research_results,
            "rag_results": rag_results,
            "route": state.get(
                "route",
                "research",
            ),
            "sources": result.get(
                "sources",
                state.get(
                    "sources",
                    [],
                ),
            ),
            "current_step": result.get(
                "current_step",
                "research_complete",
            ),
        }

    # ======================================================
    # Analysis
    # ======================================================

    def analysis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.analysis_agent.run(
            state
        )

    # ======================================================
    # Synthesis
    # ======================================================

    def synthesis_node(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.synthesis_agent.run(
            state
        )

    # ======================================================
    # Run
    # ======================================================

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.graph.invoke(
            state
        )

    # ======================================================
    # Cleanup
    # ======================================================

    def close(self):

        self.rag_agent.close()