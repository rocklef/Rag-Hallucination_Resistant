"""
Document loader — reads PDF, TXT, Markdown files and chunks them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

from core.config import config

logger = logging.getLogger(__name__)


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def load_file(path: str | Path) -> List[Dict[str, Any]]:
    """Load a single file and return list of {text, metadata} dicts."""
    path = Path(path)
    logger.info("Loading file: %s", path)

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in {".txt", ".md", ".rst"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        logger.warning("Unsupported file type: %s — loading as plain text", path.suffix)
        loader = TextLoader(str(path), encoding="utf-8")

    raw_docs = loader.load()
    splitter = _get_splitter()
    chunks = splitter.split_documents(raw_docs)

    result = []
    for i, chunk in enumerate(chunks):
        result.append(
            {
                "text": chunk.page_content.strip(),
                "metadata": {
                    **chunk.metadata,
                    "source": str(path),
                    "chunk_index": i,
                },
            }
        )

    logger.info("Loaded %d chunks from %s", len(result), path.name)
    return result


def load_directory(directory: str | Path, glob: str = "**/*.{txt,md,pdf}") -> List[Dict[str, Any]]:
    """Load all supported files from a directory recursively."""
    directory = Path(directory)
    all_chunks: List[Dict[str, Any]] = []

    for pattern in ["**/*.txt", "**/*.md", "**/*.pdf"]:
        for file_path in directory.glob(pattern):
            try:
                chunks = load_file(file_path)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error("Failed to load %s: %s", file_path, exc)

    logger.info("Total chunks loaded from directory: %d", len(all_chunks))
    return all_chunks


def load_text(text: str, source: str = "user_input") -> List[Dict[str, Any]]:
    """Split raw text into chunks directly."""
    splitter = _get_splitter()
    pieces = splitter.split_text(text)
    return [
        {"text": p.strip(), "metadata": {"source": source, "chunk_index": i}}
        for i, p in enumerate(pieces)
        if p.strip()
    ]
