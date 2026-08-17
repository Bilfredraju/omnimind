import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.llm.groq_provider import GroqProvider


print("=" * 60)
print("OMNIMIND LLM TEST")
print("=" * 60)


llm = GroqProvider()


prompt = """
Explain Retrieval-Augmented Generation in three simple sentences.
"""


print("\nSending request to Groq...\n")


response = llm.generate(prompt)


print("LLM RESPONSE")
print("-" * 60)
print(response)


print("\n" + "=" * 60)
print("LLM TEST SUCCESSFUL")
print("=" * 60)