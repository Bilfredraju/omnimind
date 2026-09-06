from agents.state import AgentState
from models.llm.groq_provider import GroqProvider


class AnalysisAgent:
    """
    Analyze evidence collected by OmniMind.

    Evidence sources:
        1. Semantic long-term memory
        2. Temporal/consolidated memory
        3. Uploaded documents / RAG
        4. External web research

    The Analysis Agent does not generate the final user-facing answer.
    It produces concise reasoning notes for the Synthesis Agent.
    """

    def __init__(self):
        self.llm = GroqProvider()

    def run(self, state: AgentState) -> AgentState:
        query = state.get("query", "")

        # --------------------------------------------------
        # Memory evidence
        # --------------------------------------------------

        memory_results = state.get(
            "memory_results",
            [],
        )

        temporal_memory_results = state.get(
            "temporal_memory_results",
            [],
        )

        memory_context = state.get(
            "memory_context",
            "",
        )

        temporal_memory_context = state.get(
            "temporal_memory_context",
            "",
        )

        temporal_intent = state.get(
            "temporal_intent",
            {},
        )

        # --------------------------------------------------
        # Build semantic memory evidence
        # --------------------------------------------------

        semantic_evidence = []

        for index, result in enumerate(
            memory_results,
            start=1,
        ):
            if not isinstance(result, dict):
                continue

            text = (
                result.get("text")
                or result.get("content")
                or result.get("memory")
                or ""
            )

            if not text:
                continue

            score = result.get(
                "score",
                result.get("ranking_score", 0.0),
            )

            metadata = result.get(
                "metadata",
                {},
            )

            memory_type = (
                metadata.get("type", "unknown")
                if isinstance(metadata, dict)
                else "unknown"
            )

            semantic_evidence.append(
                f"""
[Semantic Memory {index}]
Type: {memory_type}
Relevance Score: {float(score):.4f}

{text}
""".strip()
            )

        semantic_memory_text = "\n\n".join(
            semantic_evidence
        )

        if not semantic_memory_text:
            semantic_memory_text = (
                "No semantic long-term memory evidence available."
            )

        # --------------------------------------------------
        # Build temporal memory evidence
        # --------------------------------------------------

        temporal_evidence = []

        for index, result in enumerate(
            temporal_memory_results,
            start=1,
        ):
            if not isinstance(result, dict):
                continue

            # Consolidated memory results may expose their
            # information under different fields.
            topic = result.get(
                "topic",
                "",
            )

            summary = result.get(
                "summary",
                "",
            )

            text = result.get(
                "text",
                "",
            )

            score = result.get(
                "score",
                result.get("retrieval_score", 0.0),
            )

            current_memory_id = result.get(
                "current_memory_id",
                "",
            )

            historical_memory_ids = result.get(
                "historical_memory_ids",
                [],
            )

            timeline = result.get(
                "timeline",
                [],
            )

            parts = []

            if topic:
                parts.append(f"Topic: {topic}")

            if summary:
                parts.append(f"Summary: {summary}")

            if text:
                parts.append(f"Memory: {text}")

            if current_memory_id:
                parts.append(
                    f"Current Memory ID: {current_memory_id}"
                )

            if historical_memory_ids:
                parts.append(
                    "Historical Memory IDs: "
                    + ", ".join(
                        str(item)
                        for item in historical_memory_ids
                    )
                )

            if timeline:
                parts.append(
                    f"Timeline Events: {timeline}"
                )

            parts.append(
                f"Retrieval Score: {float(score):.4f}"
            )

            if parts:
                temporal_evidence.append(
                    f"""
[Temporal Memory {index}]
{"\n".join(parts)}
""".strip()
                )

        temporal_memory_text = "\n\n".join(
            temporal_evidence
        )

        # If structured temporal results are unavailable,
        # preserve the generated temporal context.
        if not temporal_memory_text:
            temporal_memory_text = (
                temporal_memory_context.strip()
                if temporal_memory_context
                else "No temporal memory evidence available."
            )

        # --------------------------------------------------
        # Combined memory context
        # --------------------------------------------------

        combined_memory_context_parts = []

        if memory_context.strip():
            combined_memory_context_parts.append(
                "SEMANTIC MEMORY CONTEXT:\n"
                + memory_context.strip()
            )

        if temporal_memory_context.strip():
            combined_memory_context_parts.append(
                "TEMPORAL MEMORY CONTEXT:\n"
                + temporal_memory_context.strip()
            )

        combined_memory_context = "\n\n".join(
            combined_memory_context_parts
        )

        if not combined_memory_context:
            combined_memory_context = (
                "No combined memory context available."
            )

        # --------------------------------------------------
        # RAG evidence
        # --------------------------------------------------

        rag_results = state.get(
            "rag_results",
            [],
        )

        rag_evidence = []

        for index, result in enumerate(
            rag_results,
            start=1,
        ):
            if not isinstance(result, dict):
                continue

            rag_evidence.append(
                f"""
[RAG Evidence {index}]
Source: {result.get("source", "Unknown")}
Page: {result.get("page", "Unknown")}
Chunk: {result.get("chunk", "Unknown")}
Score: {float(result.get("score", 0.0)):.4f}

{result.get("text", "")}
""".strip()
            )

        rag_text = "\n\n".join(
            rag_evidence
        )

        if not rag_text:
            rag_text = (
                "No document evidence available."
            )

        # --------------------------------------------------
        # Web research evidence
        # --------------------------------------------------

        research_results = state.get(
            "research_results",
            [],
        )

        web_evidence = []

        for index, result in enumerate(
            research_results,
            start=1,
        ):
            if not isinstance(result, dict):
                continue

            web_evidence.append(
                f"""
[Web Evidence {index}]
Title: {result.get("title", "Unknown")}
URL: {result.get("url", "")}

{result.get("snippet", "")}
""".strip()
            )

        web_text = "\n\n".join(
            web_evidence
        )

        if not web_text:
            web_text = (
                "No external web evidence available."
            )

        # --------------------------------------------------
        # Temporal intent
        # --------------------------------------------------

        temporal_intent_text = (
            str(temporal_intent)
            if temporal_intent
            else "No explicit temporal intent detected."
        )

        # --------------------------------------------------
        # Analysis prompt
        # --------------------------------------------------

        prompt = f"""
You are the Analysis Agent in OmniMind.

Analyze the evidence collected from the user's long-term
memory, temporal memory, documents, and/or external web research.

USER QUESTION:
{query}

==================================================
TEMPORAL INTENT
==================================================

{temporal_intent_text}

==================================================
SEMANTIC LONG-TERM MEMORY
==================================================

{semantic_memory_text}

==================================================
TEMPORAL / CONSOLIDATED MEMORY
==================================================

{temporal_memory_text}

==================================================
COMBINED MEMORY CONTEXT
==================================================

{combined_memory_context}

==================================================
DOCUMENT / RAG EVIDENCE
==================================================

{rag_text}

==================================================
EXTERNAL WEB EVIDENCE
==================================================

{web_text}

==================================================
ANALYSIS RULES
==================================================

1. Use ONLY the evidence provided above.

2. Do NOT invent information.

3. Identify the facts that directly answer the
   user's question.

4. Clearly distinguish between:
   - semantic long-term memory
   - temporal/consolidated memory
   - user's documents
   - external web research

5. Treat memory as historical context unless the
   evidence explicitly establishes that it is current.

6. Temporal memory is especially important for questions
   involving:
   - past decisions
   - previous project states
   - historical choices
   - "last month"
   - "3 months ago"
   - "previously"
   - "before"
   - "when did I decide"
   - similar time-dependent requests

7. When semantic and temporal memory overlap, compare them
   rather than blindly treating them as separate facts.

8. When current and historical memories disagree:
   - identify the disagreement
   - determine which memory is current when the evidence
     explicitly indicates this
   - preserve historical context

9. When memory and document/web evidence disagree:
   explicitly identify the conflict.

10. When both document and web evidence are available:
    identify similarities, differences, extensions,
    and meaningful gaps.

11. Do not treat a web search result as proof merely because
    it appears in the search results.

12. Do not produce a polished final answer.

13. Produce concise analytical notes that the
    Synthesis Agent can use.

14. If an evidence source is unavailable, do not
    pretend that it exists.

ANALYSIS:
""".strip()

        analysis = self.llm.generate(
            prompt
        )

        return {
            **state,
            "analysis": analysis,
            "current_step": "analysis_complete",
        }