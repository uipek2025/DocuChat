"""Application configuration settings.

Values can be overridden via environment variables.
"""
import os


class Settings:
    # JWT / Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGEME_SUPER_SECRET")
    ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # OpenAI (LLM) – used for summaries and final answers
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL")  # optional; leave unset for OpenAI cloud

    # RAG knobs
    SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "200"))
    ANSWER_MAX_TOKENS: int = int(os.getenv("ANSWER_MAX_TOKENS", "400"))

    # Vector store / embeddings
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./docuchat_backend/chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "docuchat")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Optional OCR binary paths (Linux/Windows); leave empty if on PATH
    TESSERACT_EXE: str | None = os.getenv("TESSERACT_EXE")      # e.g. "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    POPLER_BIN_DIR: str | None = os.getenv("POPLER_BIN_DIR")    # e.g. "C:\\Tools\\poppler-25.07.0\\Library\\bin"


settings = Settings()
