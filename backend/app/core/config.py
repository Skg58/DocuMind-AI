"""Application configuration settings for DocuMind AI."""

from pathlib import Path
from typing import List

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """DocuMind AI application settings loaded from environment variables or defaults."""

    # Application Information
    APP_NAME: str = "DocuMind AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # CORS Configuration
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # File Constraints
    MAX_FILE_SIZE_MB: int = Field(default=15, ge=1)

    # LLM Configuration
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=1)

    # Embedding and Reranking Models
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking Configuration
    CHUNK_SIZE: int = Field(default=512, ge=1)
    CHUNK_OVERLAP: int = Field(default=64, ge=0)

    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = Field(default=20, ge=1)
    TOP_K_RERANK: int = Field(default=5, ge=1)

    # Rate Limiting
    RATE_LIMIT_RPM: int = Field(default=30, ge=1)

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "documind_documents"
    SPARSE_MODEL: str = "Qdrant/bm25"

    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = ""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def allowed_origins_list(self) -> List[str]:
        """Split ALLOWED_ORIGINS string on comma and strip whitespace.

        Returns:
            List[str]: Cleaned list of allowed origin URLs for CORS.
        """
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


settings = Settings()
