from groq import Groq

from config.settings import settings
from models.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    """Groq-based LLM provider."""

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
    ):
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Add it to your .env file."
            )

        self.model_name = model_name

        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a response using Groq."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content