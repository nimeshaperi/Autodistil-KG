"""
Graph Traverser Module - Semantic-aware graph traversal and CHATML dataset generation.

This module is organized in a modular architecture where each component
(graph_db, llm, state_storage, chatml) is self-contained with its own
interface, config, factory, and implementations.
"""
# Main classes
from .graph_traverser import GraphTraverser
from .graph_traverser_agent import GraphTraverserAgent

# High-level configuration
from .config import (
    GraphTraverserAgentConfig,
    TraversalConfig,
    DatasetGenerationConfig,
    TraversalStrategy,
)

# Module configs (re-exported for convenience)
from .graph_db import GraphDatabaseConfig
from autodistil_kg.llm import LLMConfig, LLMProvider
from .state_storage import StateStorageConfig

# CHATML
from .chatml import (
    ChatMLDataset,
    ChatMLConversation,
    ChatMLMessage,
    ChatMLFormatter
)

# Interfaces (from modules)
from .graph_db import GraphDatabase, GraphNode, GraphEdge
from autodistil_kg.llm import LLMClient, LLMMessage
from .state_storage import StateStorage, NodeState, NodeMetadata

# Factory functions
from .factory import (
    create_graph_database,
    create_llm_client,
    create_state_storage
)

# Graph database implementations
from .graph_db import Neo4jGraphDatabase

# LLM implementations
from autodistil_kg.llm import (
    OpenAIClient,
    GeminiClient,
    ClaudeClient,
    OllamaClient,
    VLLMClient,
)

# State storage implementations
from .state_storage import RedisStateStorage

__all__ = [
    # Main classes
    "GraphTraverser",
    "GraphTraverserAgent",
    
    # High-level configuration
    "GraphTraverserAgentConfig",
    "TraversalConfig",
    "DatasetGenerationConfig",
    "TraversalStrategy",
    
    # Module configs
    "GraphDatabaseConfig",
    "LLMConfig",
    "LLMProvider",
    "StateStorageConfig",
    
    # CHATML
    "ChatMLDataset",
    "ChatMLConversation",
    "ChatMLMessage",
    "ChatMLFormatter",
    
    # Interfaces
    "GraphDatabase",
    "GraphNode",
    "GraphEdge",
    "LLMClient",
    "LLMMessage",
    "StateStorage",
    "NodeState",
    "NodeMetadata",
    
    # Factory functions
    "create_graph_database",
    "create_llm_client",
    "create_state_storage",
    
    # Graph database implementations
    "Neo4jGraphDatabase",
    
    # LLM implementations
    "OpenAIClient",
    "GeminiClient",
    "ClaudeClient",
    "OllamaClient",
    "VLLMClient",
    
    # State storage implementations
    "RedisStateStorage",
]
