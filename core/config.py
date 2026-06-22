"""
Core configuration for the Multi-Agent RAG system.
Loads settings from environment variables / .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    # ── LLM Provider ────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # "openai" | "ollama"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # ── Embeddings ───────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Vector Store ─────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_docs")

    # ── Retrieval ────────────────────────────────────────────────
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "8"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "6.0"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ── Hallucination Detection ──────────────────────────────────
    HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.75"))
    NLI_MODEL: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

    # ── Logging ──────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Agent Retry ──────────────────────────────────────────────
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))


config = Config()
