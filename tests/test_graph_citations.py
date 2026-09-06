import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.graph import OmniMindGraph


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = str(
    PROJECT_ROOT / "data" / "raw" / "sample.pdf"
)


# ============================================================
# TEST
# ============================================================

def test_full_graph_citations():
    """
    Validate citation propagation through the complete
    OmniMind graph.

    Flow:

        Query
          ↓
        Memory
          ↓
        Planner
          ↓
        RAG
          ↓
        Analysis
          ↓
        Synthesis
          ↓
        Final Answer + Sources
    """

    graph = OmniMindGraph(
        pdf_path=PDF_PATH
    )

    initial_state = {
        "query": "What datasets were used to evaluate the RAG models?",
        "current_step": "start",
        "error": "",
    }

    try:
        result = graph.run(initial_state)

        # --------------------------------------------------
        # Basic graph validation
        # --------------------------------------------------

        assert isinstance(result, dict)

        print("Graph execution: PASSED")

        # --------------------------------------------------
        # Final state validation
        # --------------------------------------------------

        current_step = result.get(
            "current_step",
            "",
        )

        print(
            f"Final step: {current_step}"
        )

        assert current_step in {
            "memory_write_complete",
            "synthesis_complete",
        }

        # --------------------------------------------------
        # Error validation
        # --------------------------------------------------

        error = result.get(
            "error",
            "",
        )

        print(
            f"Error: {error}"
        )

        assert not error

        # --------------------------------------------------
        # RAG validation
        # --------------------------------------------------

        rag_results = result.get(
            "rag_results",
            [],
        )

        print(
            f"RAG results: {len(rag_results)}"
        )

        assert rag_results

        # --------------------------------------------------
        # Citation validation
        # --------------------------------------------------

        citation_count = 0

        for index, rag_result in enumerate(
            rag_results,
            start=1,
        ):
            citation = rag_result.get(
                "citation",
                {},
            )

            if not isinstance(citation, dict):
                citation = {}

            metadata = rag_result.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            citation_id = citation.get(
                "citation_id"
            )

            source = citation.get(
                "source",
                rag_result.get(
                    "source",
                    metadata.get("source"),
                ),
            )

            document_name = citation.get(
                "document_name",
                rag_result.get(
                    "document_name",
                    metadata.get("document_name"),
                ),
            )

            page = citation.get(
                "page",
                rag_result.get(
                    "page",
                    metadata.get(
                        "page",
                        rag_result.get(
                            "page_number",
                            metadata.get("page_number"),
                        ),
                    ),
                ),
            )

            chunk_id = citation.get(
                "chunk_id",
                rag_result.get(
                    "chunk_id",
                    metadata.get("chunk_id"),
                ),
            )

            if citation_id:
                citation_count += 1

            print(
                f"\nRAG Result {index}"
            )
            print(
                f"  Citation ID: {citation_id}"
            )
            print(
                f"  Document: {document_name}"
            )
            print(
                f"  Source: {source}"
            )
            print(
                f"  Page: {page}"
            )
            print(
                f"  Chunk ID: {chunk_id}"
            )

            assert citation_id
            assert source
            assert document_name
            assert page is not None
            assert chunk_id

        assert citation_count > 0

        print(
            f"\nStructured citations: {citation_count}"
        )

        # --------------------------------------------------
        # Final answer validation
        # --------------------------------------------------

        final_answer = result.get(
            "final_answer",
            "",
        )

        print(
            "\nFinal Answer:"
        )
        print(
            "------------------------------------------------------------"
        )
        print(final_answer)
        print(
            "------------------------------------------------------------"
        )

        assert isinstance(
            final_answer,
            str,
        )

        assert final_answer.strip()

        # --------------------------------------------------
        # Source output validation
        # --------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        print(
            f"\nFinal source records: {len(sources)}"
        )

        assert sources

        document_sources = [
            source
            for source in sources
            if source.get("type") == "document"
        ]

        assert document_sources

        for source in document_sources:
            print(
                "\nDocument Source:"
            )
            print(
                f"  Citation ID: {source.get('citation_id')}"
            )
            print(
                f"  Document: {source.get('document_name')}"
            )
            print(
                f"  Source: {source.get('source')}"
            )
            print(
                f"  Page: {source.get('page')}"
            )
            print(
                f"  Chunk ID: {source.get('chunk_id')}"
            )

            assert source.get("citation_id")
            assert source.get("document_name")
            assert source.get("source")
            assert source.get("page") is not None
            assert source.get("chunk_id")

        # --------------------------------------------------
        # Citation consistency
        # --------------------------------------------------

        rag_citation_ids = {
            rag_result.get(
                "citation",
                {},
            ).get(
                "citation_id"
            )
            for rag_result in rag_results
            if isinstance(
                rag_result.get("citation", {}),
                dict,
            )
        }

        source_citation_ids = {
            source.get("citation_id")
            for source in document_sources
        }

        rag_citation_ids.discard(None)
        source_citation_ids.discard(None)

        print(
            f"\nRAG citation IDs: {sorted(rag_citation_ids)}"
        )

        print(
            f"Source citation IDs: {sorted(source_citation_ids)}"
        )

        assert rag_citation_ids
        assert source_citation_ids

        assert rag_citation_ids.intersection(
            source_citation_ids
        )

        print()
        print("=" * 60)
        print(
            "PHASE 19.6.2 FULL GRAPH CITATION VALIDATION: PASSED"
        )
        print("=" * 60)

    finally:
        graph.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_full_graph_citations()