from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # LLM Provider
    anthropic_api_key: str = ""

    # Models
    llm_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "policy_documents"

    # Retrieval Parameters
    hybrid_search_alpha: float = 0.5  # 0=BM25 only, 1=Dense only
    retrieval_top_k: int = 10
    rrf_k: int = 60

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
