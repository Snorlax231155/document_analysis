import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Document Intelligence & RAG Assistant"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Base Paths (Project Root: document_analasis/)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    INDEXES_DIR: Path = DATA_DIR / "indexes"
    EVALUATION_DIR: Path = DATA_DIR / "evaluation"

    # Database & FAISS Storage
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/app.db"
    FAISS_INDEX_PATH: str = str(DATA_DIR / "indexes" / "faiss.index")
    FAISS_METADATA_PATH: str = str(DATA_DIR / "indexes" / "faiss_meta.json")

    # Ingestion & Chunking Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # Retrieval & Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_WEIGHT: float = 0.6
    KEYWORD_WEIGHT: float = 0.4
    TOP_K: int = 10
    FINAL_TOP_K: int = 5

    # Reranking Settings
    ENABLE_RERANKING: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM Provider Configuration (auto, openai, ollama, mock)
    LLM_PROVIDER: str = "auto"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.0

    # Ollama Local LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Ensure all required runtime data directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        self.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
