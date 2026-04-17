"""
State Storage Interface Module.

This module defines the abstract interface for state storage providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from enum import Enum


class NodeState(str, Enum):
    """State of a node in the traversal."""
    VISITED = "visited"
    NOT_VISITED = "not_visited"
    IN_PROGRESS = "in_progress"
    SKIPPED = "skipped"


class NodeMetadata(BaseModel):
    """Metadata associated with a node during traversal."""
    node_id: str
    state: NodeState
    visit_count: int
    last_visited: Optional[float]  # timestamp
    metadata: Dict[str, Any]


class StateStorage(ABC):
    """Abstract interface for state storage providers."""

    @abstractmethod
    def get_node_state(self, node_id: str) -> Optional[NodeMetadata]:
        pass

    @abstractmethod
    def set_node_state(self, node_id: str, metadata: NodeMetadata) -> None:
        pass

    @abstractmethod
    def mark_visited(self, node_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def mark_not_visited(self, node_id: str) -> None:
        pass

    @abstractmethod
    def get_visited_nodes(self) -> List[str]:
        pass

    @abstractmethod
    def get_unvisited_nodes(self, all_node_ids: List[str]) -> List[str]:
        pass

    @abstractmethod
    def clear_state(self) -> None:
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        pass
