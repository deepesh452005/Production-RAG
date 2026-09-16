"""
Central config. Reads from .env so no keys are ever hardcoded.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-20b"

    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "documents"

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    min_rerank_score: float = 0.15  # below this, refuse to answer instead of guessing
    class Config:
        env_file = ".env"


settings = Settings()
