"""
LLM Configuration Module.

This module defines configuration classes for LLM providers.
"""
from pydantic import BaseModel, model_validator
from typing import Dict, Any, Optional
from enum import Enum


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"
    VLLM = "vllm"


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    provider: LLMProvider
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    # Gemini-specific
    project_id: Optional[str] = None
    location: Optional[str] = None
    credentials_path: Optional[str] = None
    # Additional parameters
    additional_params: Dict[str, Any] = {}

    @model_validator(mode="after")
    def check_provider_requirements(self) -> "LLMConfig":
        if self.provider == LLMProvider.OPENAI and not self.api_key:
            raise ValueError("OpenAI requires api_key")
        # openai_compatible does NOT require api_key (local servers, OpenRouter uses its own key pattern)
        if self.provider == LLMProvider.GEMINI and not self.project_id:
            raise ValueError("Gemini requires project_id")
        if self.provider == LLMProvider.CLAUDE and not self.api_key:
            raise ValueError("Claude requires api_key")
        return self
