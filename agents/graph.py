from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.planner import PlannerAgent
from agents.rag_agent import RAGAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.synthesis_agent import SynthesisAgent
from agents.memory_agent import MemoryAgent


class OmniMindGraph:
    """
    LangGraph orchestration for OmniMind.

    Routes:

        rag
            -> Memory Recall
            -> RAG
            -> Analysis
            -> Synthesis
            -> Memory Write

        research
            -> Memory Recall
            -> Research
            -> Analysis
            -> Synthesis
            -> Memory Write

        both
            -> Memory Recall
            -> RAG
            -> Research
            -> Analysis
            -> Synthesis
            -> Memory Write
    """

    def __init__(self, pdf_path: str):

        # --------------------------------------------------
        # Agents
        # --------------------------------------------------

        self.planner = PlannerAgent()

        self.memory_agent = MemoryAgent()

        self.rag_agent = RAGAgent(
            pdf_path=pdf_path
        )

        self.research_agent = ResearchAgent()

        self.analysis_agent = AnalysisAgent()

        self.synthesis_agent = SynthesisAgent()

        # --------------------------------------------------
        # Build graph
        # --------------------------------------------------

        graph = StateGraph(AgentState)

        # --------------------------------------------------
        # Nodes
        # --------------------------------------------------

        graph.add_node(
            "planner",
            self.planner_node,
        )

        graph.add_node(
            "memory_recall",
            self.memory_recall_node,
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

        graph.add_node(
            "memory_write",
            self.memory_write_node,
        )

        # --------------------------------------------------
        # START -> Planner
        # --------------------------------------------------

        graph.add_edge(
            START,
            "planner",
        )

        # --------------------------------------------------
        # Planner -> Memory Recall
        # --------------------------------------------------

        graph.add_edge(
            "planner",
            "memory_recall",
        )

        # --------------------------------------------------
        # Memory Recall -> RAG / Research
        # --------------------------------------------------

        graph.add_conditional_edges(
            "memory_recall",
            self.route_from_memory,
            {
                "rag": "rag",
                "research": "research",
                "both": "rag",
            },
        )

        # --------------------------------------------------
        # RAG -> Analysis / Research
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
        # Research -> Analysis
        # --------------------------------------------------

        graph.add_edge(
            "research",
            "analysis",
        )

        # --------------------------------------------------
        # Analysis -> Synthesis
        # --------------------------------------------------

        graph.add_edge(
            "analysis",
            "synthesis",
        )

        # --------------------------------------------------
        # Synthesis -> Memory Write
        # --------------------------------------------------

        graph.add_edge(
            "synthesis",
            "memory_write",
        )

        # --------------------------------------------------
        # Memory Write -> END
        # --------------------------------------------------

        graph.add_edge(
            "memory_write",
            END,
        )

        # --------------------------------------------------
        # Compile
        # --------------------------------------------------

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
    # MEMORY RECALL
    # ========================================================

    def memory_recall_node(
        self,
        state: AgentState,
    ) -> AgentState:

        result = self.memory_agent.recall(state)

        return {
            "memory_results": result.get(
                "memory_results",
                [],
            ),
            "memory_context": result.get(
                "memory_context",
                "",
            ),
            "current_step": result.get(
                "current_step",
                "memory_recall_complete",
            ),
            "error": result.get(
                "error",
                state.get("error", ""),
            ),
        }

    # ========================================================
    # MEMORY ROUTING
    # ========================================================

    def route_from_memory(
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
    # RAG ROUTING
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
    # MEMORY WRITE NODE
    # ========================================================

    def memory_write_node(
        self,
        state: AgentState,
    ) -> AgentState:

        result = self.memory_agent.write(state)

        return {
            "memory_written": result.get(
                "memory_written",
                False,
            ),
            "current_step": result.get(
                "current_step",
                "memory_write_complete",
            ),
            "error": result.get(
                "error",
                state.get("error", ""),
            ),
        }

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        return self.graph.invoke(state)

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.rag_agent.close()
        self.memory_agent.close()