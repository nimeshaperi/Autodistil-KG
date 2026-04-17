"""
Environment Variable Configuration Helper.

This module provides utilities to load configuration from environment variables.
Requires python-dotenv: pip install python-dotenv
"""
import os
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

from .graph_db.config import GraphDatabaseConfig
from autodistil_kg.llm.config import LLMConfig, LLMProvider
from .state_storage.config import StateStorageConfig


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, searches for .env in project root.
    """
    if not DOTENV_AVAILABLE:
        raise ImportError(
            "python-dotenv is required to load .env files. "
            "Install with: pip install python-dotenv"
        )
    
    if env_path is None:
        # Try to find .env in project root
        project_root = Path(__file__).parent.parent.parent.parent
        env_path = project_root / ".env"
    
    if isinstance(env_path, str):
        env_path = Path(env_path)
    
    if env_path.exists():
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")


def get_graph_db_config_from_env() -> GraphDatabaseConfig:
    """
    Create GraphDatabaseConfig from environment variables.
    
    Returns:
        GraphDatabaseConfig instance
    """
    return GraphDatabaseConfig(
        provider="neo4j",
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE")
    )


def get_llm_config_from_env(provider: Optional[str] = None) -> LLMConfig:
    """
    Create LLMConfig from environment variables.
    
    Args:
        provider: LLM provider name. If None, tries to detect from env vars.
    
    Returns:
        LLMConfig instance
    """
    if provider is None:
        # Auto-detect provider based on available API keys
        if os.getenv("OPENAI_API_KEY"):
            provider = LLMProvider.OPENAI.value
        elif os.getenv("GEMINI_PROJECT_ID"):
            provider = LLMProvider.GEMINI.value
        elif os.getenv("CLAUDE_API_KEY"):
            provider = LLMProvider.CLAUDE.value
        elif os.getenv("OLLAMA_BASE_URL"):
            provider = LLMProvider.OLLAMA.value
        elif os.getenv("VLLM_BASE_URL"):
            provider = LLMProvider.VLLM.value
        else:
            raise ValueError("No LLM provider detected. Set appropriate environment variables.")
    
    provider_lower = provider.lower()
    
    if provider_lower == LLMProvider.OPENAI.value:
        return LLMConfig(
            provider=LLMProvider.OPENAI.value,
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            base_url=os.getenv("OPENAI_BASE_URL") or None
        )
    elif provider_lower == LLMProvider.GEMINI.value:
        return LLMConfig(
            provider=LLMProvider.GEMINI.value,
            project_id=os.getenv("GEMINI_PROJECT_ID", ""),
            location=os.getenv("GEMINI_LOCATION", "us-central1"),
            model=os.getenv("GEMINI_MODEL", "gemini-pro"),
            credentials_path=os.getenv("GEMINI_CREDENTIALS_PATH") or None
        )
    elif provider_lower == LLMProvider.CLAUDE.value:
        return LLMConfig(
            provider=LLMProvider.CLAUDE.value,
            api_key=os.getenv("CLAUDE_API_KEY", ""),
            model=os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
        )
    elif provider_lower == LLMProvider.OLLAMA.value:
        return LLMConfig(
            provider=LLMProvider.OLLAMA.value,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama2")
        )
    elif provider_lower == LLMProvider.VLLM.value:
        return LLMConfig(
            provider=LLMProvider.VLLM.value,
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
            model=os.getenv("VLLM_MODEL") or None,
            additional_params={
                "chat_path": os.getenv("VLLM_CHAT_PATH", "/v1/chat/completions"),
            },
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_state_storage_config_from_env() -> StateStorageConfig:
    """
    Create StateStorageConfig from environment variables.
    
    Returns:
        StateStorageConfig instance
    """
    _pw = (os.getenv("REDIS_PASSWORD") or "").strip()
    return StateStorageConfig(
        provider="redis",
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=_pw if _pw else None,  # None when empty/whitespace (no AUTH for passwordless Redis)
        key_prefix=os.getenv("REDIS_KEY_PREFIX", "graph_traverser:")
    )
