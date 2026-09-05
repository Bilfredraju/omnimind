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

    Phase 17.2:
    Memory is recalled BEFORE planning so previous semantic and
    temporal knowledge can influence the execution plan.

    Routes:

        rag
            -> Memory Recall
            -> Memory-Augmented Planner
            -> RAG
            -> Analysis
            -> Synthesis
            -> Memory Write

        research
            -> Memory Recall
            -> Memory-Augmented Planner
            -> Research
            -> Analysis
            -> Synthesis
            -> Memory Write

        both
            -> Memory Recall
            -> Memory-Augmented Planner
            -> RAG
            -> Research
            -> Analysis
            -> Synthesis
            -> Memory Write

    Memory Recall provides:

        1. Semantic long-term memory
        2. Temporal/consolidated memory

    Both are passed to the planner before route-specific
    retrieval begins and remain available to downstream
    reasoning and synthesis stages.
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
        # START -> Memory Recall
        # --------------------------------------------------
        #
        # Phase 17.2:
        # Memory must be available before planning.
        # --------------------------------------------------

        graph.add_edge(
            START,
            "memory_recall",
        )

        # --------------------------------------------------
        # Memory Recall -> Planner
        # --------------------------------------------------

        graph.add_edge(
            "memory_recall",
            "planner",
        )

        # --------------------------------------------------
        # Planner -> RAG / Research
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
    # MEMORY RECALL
    # ========================================================

    def memory_recall_node(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Retrieve both semantic and temporal memory.

        Phase 17.2:
        Memory is retrieved before the planner runs.

        All memory outputs are explicitly propagated into
        LangGraph state so the planner, Analysis, and
        Synthesis stages can use them.
        """

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

            "temporal_intent": result.get(
                "temporal_intent",
                {},
            ),

            "temporal_memory_results": result.get(
                "temporal_memory_results",
                [],
            ),

            "temporal_memory_context": result.get(
                "temporal_memory_context",
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
    # PLANNER
    # ========================================================

    def planner_node(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Run the memory-aware planner.

        The planner now receives semantic and temporal memory
        because Memory Recall executes before this node.
        """

        result = self.planner.plan(state)

        return {
            "plan": result.get(
                "plan",
                [],
            ),

            "route": result.get(
                "route",
                "rag",
            ),

            "planning_memory_context": result.get(
                "planning_memory_context",
                "",
            ),

            "current_step": result.get(
                "current_step",
                "planning_complete",
            ),

            "error": result.get(
                "error",
                state.get("error", ""),
            ),
        }

    # ========================================================
    # PLANNER ROUTING
    # ========================================================

    def route_from_planner(
        self,
        state: AgentState,
    ) -> str:
        """
        Route according to the planner's deterministic route.

        Memory influences the plan, while explicit routing
        rules continue to determine whether the request uses
        RAG, research, or both.
        """

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
        """
        Determine whether RAG should be followed by research.
        """

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
        """
        Execute document retrieval.
        """

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

            "error": result.get(
                "error",
                state.get("error", ""),
            ),
        }

    # ========================================================
    # RESEARCH NODE
    # ========================================================

    def research_node(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Execute external research.
        """

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
                state.get("error", ""),
            ),
        }

    # ========================================================
    # ANALYSIS NODE
    # ========================================================

    def analysis_node(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Analyze RAG, research, semantic memory and temporal
        memory evidence.
        """

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

            "error": result.get(
                "error",
                state.get("error", ""),
            ),
        }

    # ========================================================
    # SYNTHESIS NODE
    # ========================================================

    def synthesis_node(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Generate the final memory-aware response.
        """

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

            "error": result.get(
                "error",
                state.get("error", ""),
            ),
        }

    # ========================================================
    # MEMORY WRITE NODE
    # ========================================================

    def memory_write_node(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Persist important information from the completed
        interaction into long-term memory.
        """

        result = self.memory_agent.write(state)

        return {
            "memory_written": result.get(
                "memory_written",
                False,
            ),

            "memory_count": result.get(
                "memory_count",
                state.get("memory_count", 0),
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
        """
        Execute the complete OmniMind graph.
        """

        return self.graph.invoke(state)

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """
        Release agent resources.
        """

        self.rag_agent.close()
        self.memory_agent.close()