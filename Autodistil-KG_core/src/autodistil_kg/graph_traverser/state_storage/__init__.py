"""
State Storage Module.

This module provides a self-contained interface for state storage operations.
Each module is independent and can be easily replaced or extended.
"""
from .interface import StateStorage, NodeState, NodeMetadata
from .config import StateStorageConfig
from .factory import create_state_storage
from .redis_storage import RedisStateStorage

__all__ = [
    "StateStorage",
    "NodeState",
    "NodeMetadata",
    "StateStorageConfig",
    "create_state_storage",
    "RedisStateStorage",
]
