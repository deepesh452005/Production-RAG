"""
A minimal, RAGAS-compatible wrapper around FastEmbed.

RAGAS's internal usage-tracking expects the embeddings object to expose
a plain string `.model` attribute. langchain_community's FastEmbedEmbeddings
instead exposes the raw internal model object, which crashes RAGAS's
telemetry with a pydantic validation error. This wrapper embeds using
fastembed directly and exposes `.model` as a proper string, sidestepping
the bug entirely.
"""
from fastembed import TextEmbedding


class SafeFastEmbedEmbeddings:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = model_name  # plain string — this is what RAGAS's tracker needs
        self._engine = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._engine.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._engine.embed([text]))[0].tolist()