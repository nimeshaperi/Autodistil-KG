"""
Graph Database Module.

This module provides a self-contained interface for graph database operations.
Each module is independent and can be easily replaced or extended.
"""
from .interface import GraphDatabase, GraphNode, GraphEdge
from .config import GraphDatabaseConfig
from .factory import create_graph_database
from .neo4j_provider import Neo4jGraphDatabase

__all__ = [
    "GraphDatabase",
    "GraphNode",
    "GraphEdge",
    "GraphDatabaseConfig",
    "create_graph_database",
    "Neo4jGraphDatabase",
]
