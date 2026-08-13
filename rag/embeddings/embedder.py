from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate vector embeddings for text."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully.")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Convert text into embedding vectors.
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()