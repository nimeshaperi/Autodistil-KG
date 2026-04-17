"""
LLM Module.

This module provides a self-contained interface for LLM operations.
Each module is independent and can be easily replaced or extended.
"""
from .interface import LLMClient, LLMMessage
from .config import LLMConfig, LLMProvider
from .factory import create_llm_client
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient
from .ollama_client import OllamaClient
from .vllm_client import VLLMClient

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMConfig",
    "LLMProvider",
    "create_llm_client",
    "OpenAIClient",
    "GeminiClient",
    "ClaudeClient",
    "OllamaClient",
    "VLLMClient",
]
