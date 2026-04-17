"""
State Storage Factory Module.

This module provides factory functions for creating state storage instances.
"""
import logging

from .interface import StateStorage
from .config import StateStorageConfig
from .redis_storage import RedisStateStorage

logger = logging.getLogger(__name__)


def create_state_storage(config: StateStorageConfig) -> StateStorage:
    """
    Create a state storage instance based on configuration.
    
    Args:
        config: State storage configuration
        
    Returns:
        StateStorage instance
        
    Raises:
        ValueError: If provider is not supported
    """
    provider = config.provider.lower()
    
    if provider == "redis":
        return RedisStateStorage(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            key_prefix=config.key_prefix
        )
    else:
        raise ValueError(f"Unsupported state storage provider: {config.provider}")
