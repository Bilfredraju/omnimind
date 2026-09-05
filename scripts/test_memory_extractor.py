import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from memory.extractor import MemoryExtractor


extractor = MemoryExtractor()


# --------------------------------------------------
# Decision
# --------------------------------------------------

memories = extractor.extract(
    user_message=(
        "I decided to use Qdrant as the vector "
        "database for my OmniMind project."
    ),
    assistant_message=(
        "Qdrant will be used for vector storage."
    ),
)

assert len(memories) == 1

assert memories[0]["metadata"]["type"] == "decision"

assert "Qdrant" in memories[0]["text"]


# --------------------------------------------------
# Preference
# --------------------------------------------------

memories = extractor.extract(
    user_message=(
        "I prefer Python for my AI projects."
    ),
    assistant_message="Understood.",
)

assert len(memories) == 1

assert memories[0]["metadata"]["type"] == "preference"


# --------------------------------------------------
# Ordinary conversation
# --------------------------------------------------

memories = extractor.extract(
    user_message="Hello, how are you?",
    assistant_message="I'm doing well.",
)

assert len(memories) == 0


print("=" * 60)
print("MEMORY EXTRACTOR TEST PASSED")
print("=" * 60)