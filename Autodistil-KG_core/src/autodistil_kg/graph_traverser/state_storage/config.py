"""
State Storage Configuration Module.

This module defines configuration classes for state storage providers.
"""
from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional


class StateStorageConfig(BaseModel):
    """Configuration for state storage."""
    provider: Literal["redis"] = "redis"
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    key_prefix: str = "graph_traverser:"
    additional_params: Dict[str, Any] = {}
