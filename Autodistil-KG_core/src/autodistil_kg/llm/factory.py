"""
LLM Factory Module.

This module provides factory functions for creating LLM client instances.
"""
import logging
from typing import Optional

from .interface import LLMClient
from .config import LLMConfig, LLMProvider
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient
from .ollama_client import OllamaClient
from .vllm_client import VLLMClient

logger = logging.getLogger(__name__)


def create_llm_client(config: LLMConfig) -> LLMClient:
    """
    Create an LLM client instance based on configuration.
    
    Args:
        config: LLM configuration
        
    Returns:
        LLMClient instance
        
    Raises:
        ValueError: If provider is not supported or configuration is invalid
    """
    provider = config.provider.lower()
    
    if provider in (LLMProvider.OPENAI.value, LLMProvider.OPENAI_COMPATIBLE.value):
        return OpenAIClient(
            api_key=config.api_key or "no-key",
            model=config.model or "gpt-4",
            base_url=config.base_url
        )
    
    elif provider == LLMProvider.GEMINI.value:
        return GeminiClient(
            project_id=config.project_id,
            location=config.location or "us-central1",
            model=config.model or "gemini-pro",
            credentials_path=config.credentials_path
        )
    
    elif provider == LLMProvider.CLAUDE.value:
        return ClaudeClient(
            api_key=config.api_key,
            model=config.model or "claude-3-opus-20240229"
        )
    
    elif provider == LLMProvider.OLLAMA.value:
        return OllamaClient(
            base_url=config.base_url or "http://localhost:11434",
            model=config.model or "llama2"
        )
    
    elif provider == LLMProvider.VLLM.value:
        extra = config.additional_params or {}
        return VLLMClient(
            base_url=config.base_url or "http://localhost:8000",
            model=config.model,
            chat_path=extra.get("chat_path", "/v1/chat/completions"),
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
