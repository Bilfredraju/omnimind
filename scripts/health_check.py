import sys

print("=" * 50)
print("OMNIMIND HEALTH CHECK")
print("=" * 50)

print(f"Python version: {sys.version}")

try:
    import fastapi
    print(f"FastAPI: {fastapi.__version__}")
except ImportError:
    print("FastAPI: NOT INSTALLED")

try:
    import langgraph
    print("LangGraph: OK")
except ImportError:
    print("LangGraph: NOT INSTALLED")

try:
    import sentence_transformers
    print("Sentence Transformers: OK")
except ImportError:
    print("Sentence Transformers: NOT INSTALLED")

try:
    import qdrant_client
    print("Qdrant Client: OK")
except ImportError:
    print("Qdrant Client: NOT INSTALLED")

print("=" * 50)
print("Health check completed.")
print("=" * 50)