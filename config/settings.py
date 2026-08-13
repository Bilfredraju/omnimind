import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    QDRANT_URL = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333"
    )

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    LLM_PROVIDER = os.getenv(
        "LLM_PROVIDER",
        "groq"
    )

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    )


settings = Settings()