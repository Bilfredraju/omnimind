import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.memory_agent import MemoryAgent


print("=" * 60)
print("OMNIMIND LONG-TERM MEMORY INTEGRATION TEST")
print("=" * 60)


# --------------------------------------------------
# Create memory agent
# --------------------------------------------------

print("\nInitializing Memory Agent...")

memory_agent = MemoryAgent()


# --------------------------------------------------
# Clear existing semantic memory
# --------------------------------------------------

memory_agent.semantic_store.clear()

print("Existing semantic memory cleared.")


# --------------------------------------------------
# Conversation 1
# --------------------------------------------------

first_query = (
    "I decided to use Qdrant as the vector database "
    "for my OmniMind project."
)

first_answer = (
    "The project will use Qdrant for vector storage "
    "and semantic retrieval."
)

print("\n" + "-" * 60)
print("CONVERSATION 1")
print("-" * 60)

print("\nUser:")
print(first_query)

print("\nAssistant:")
print(first_answer)


state_1 = {
    "query": first_query,
    "final_answer": first_answer,
}

write_result = memory_agent.write(state_1)

print("\nMemory written:")
print(write_result["memory_written"])


if not write_result["memory_written"]:
    raise RuntimeError("Memory write failed.")


# --------------------------------------------------
# Conversation 2
# --------------------------------------------------

second_query = (
    "What vector database did I decide to use "
    "for OmniMind?"
)

print("\n" + "-" * 60)
print("CONVERSATION 2")
print("-" * 60)

print("\nUser:")
print(second_query)


state_2 = {
    "query": second_query,
}

recall_result = memory_agent.recall(state_2)

memory_results = recall_result.get(
    "memory_results",
    [],
)


# --------------------------------------------------
# Display recalled memories
# --------------------------------------------------

print("\n" + "-" * 60)
print("RECALLED MEMORIES")
print("-" * 60)

print(
    f"\nMemories retrieved: {len(memory_results)}"
)

for index, memory in enumerate(
    memory_results,
    start=1,
):
    print(
        f"\nMemory {index}"
    )

    print(
        f"Score: {memory.get('score', 0.0):.4f}"
    )

    print(
        memory.get("text", "")
    )


# --------------------------------------------------
# Verify recall
# --------------------------------------------------

combined_memory = " ".join(
    memory.get("text", "").lower()
    for memory in memory_results
)

if "qdrant" not in combined_memory:
    raise AssertionError(
        "Expected memory containing 'Qdrant' "
        "was not retrieved."
    )


print("\n" + "=" * 60)
print("LONG-TERM MEMORY TEST PASSED")
print("=" * 60)