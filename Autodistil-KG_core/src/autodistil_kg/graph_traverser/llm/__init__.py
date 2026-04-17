"""Backward-compatibility shim -- use ``autodistil_kg.llm`` directly."""
import warnings as _warnings

_warnings.warn(
    "autodistil_kg.graph_traverser.llm is deprecated. "
    "Import from autodistil_kg.llm instead.",
    DeprecationWarning,
    stacklevel=2,
)

from autodistil_kg.llm import (  # noqa: F401 -- re-export for compat
    LLMClient,
    LLMMessage,
    LLMConfig,
    LLMProvider,
    create_llm_client,
    OpenAIClient,
    GeminiClient,
    ClaudeClient,
    OllamaClient,
    VLLMClient,
)
