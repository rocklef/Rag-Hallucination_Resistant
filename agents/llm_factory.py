"""
LLM factory — returns the correct LLM client based on config.
Supports OpenAI GPT models and Ollama local models.
"""
from __future__ import annotations

import logging
from typing import Any

from core.config import config

logger = logging.getLogger(__name__)


def get_llm(**kwargs: Any):
    """Return a LangChain-compatible LLM instance."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI model: %s", config.OPENAI_MODEL)
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=kwargs.get("temperature", 0.0),
        )
    elif provider == "ollama":
        from langchain_community.llms import Ollama

        logger.info("Using Ollama model: %s", config.OLLAMA_MODEL)
        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=kwargs.get("temperature", 0.0),
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'openai' or 'ollama'.")
