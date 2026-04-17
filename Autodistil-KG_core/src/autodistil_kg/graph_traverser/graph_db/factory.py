"""
Graph Database Factory Module.

This module provides factory functions for creating graph database instances.
"""
import logging
from typing import Optional

from .interface import GraphDatabase
from .config import GraphDatabaseConfig
from .neo4j_provider import Neo4jGraphDatabase

logger = logging.getLogger(__name__)


def create_graph_database(config: GraphDatabaseConfig) -> GraphDatabase:
    """
    Create a graph database instance based on configuration.
    
    Args:
        config: Graph database configuration
        
    Returns:
        GraphDatabase instance
        
    Raises:
        ValueError: If provider is not supported
    """
    provider = config.provider.lower()
    
    if provider == "neo4j":
        return Neo4jGraphDatabase(
            uri=config.uri,
            user=config.user,
            password=config.password,
            database=config.database
        )
    else:
        raise ValueError(f"Unsupported graph database provider: {config.provider}")
