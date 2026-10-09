"""Application settings, read once from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = None
    llm_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    max_upload_mb: int = 20
    max_documents: int = 20
    # If set, every endpoint except /health requires the header "X-API-Key: <value>".
    api_key: str | None = None
    allowed_origins: tuple[str, ...] = ()
    # Where uploaded PDFs are written temporarily (None = system temp directory).
    tmp_dir: str | None = None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    origins = tuple(o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip())
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        chunk_size=int(os.getenv("CHUNK_SIZE", Settings.chunk_size)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", Settings.chunk_overlap)),
        top_k=int(os.getenv("TOP_K", Settings.top_k)),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", Settings.max_upload_mb)),
        max_documents=int(os.getenv("MAX_DOCUMENTS", Settings.max_documents)),
        api_key=os.getenv("API_KEY") or None,
        allowed_origins=origins,
    )


@lru_cache
def get_settings() -> Settings:
    return load_settings()
