"""Redis state storage implementation."""
from typing import Dict, List, Optional, Any
import json
import time
import logging
import redis

from .interface import StateStorage, NodeMetadata, NodeState

logger = logging.getLogger(__name__)


class RedisStateStorage(StateStorage):
    """Redis implementation of StateStorage."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "graph_traverser:"
    ):
        """
        Initialize Redis state storage.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional Redis password
            key_prefix: Prefix for all keys stored in Redis
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.client: Optional[redis.Redis] = None
    
    def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Disconnected from Redis")
    
    def _get_key(self, node_id: str) -> str:
        """Get Redis key for a node."""
        return f"{self.key_prefix}node:{node_id}"
    
    def _get_visited_set_key(self) -> str:
        """Get Redis key for visited nodes set."""
        return f"{self.key_prefix}visited"
    
    def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if not self.client:
            self.connect()
    
    def get_node_state(self, node_id: str) -> Optional[NodeMetadata]:
        """Get the state of a node."""
        self._ensure_connected()
        
        key = self._get_key(node_id)
        data = self.client.get(key)
        
        if data:
            try:
                metadata_dict = json.loads(data)
                return NodeMetadata(
                    node_id=metadata_dict["node_id"],
                    state=NodeState(metadata_dict["state"]),
                    visit_count=metadata_dict["visit_count"],
                    last_visited=metadata_dict.get("last_visited"),
                    metadata=metadata_dict.get("metadata", {})
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing node state for {node_id}: {e}")
                return None
        return None
    
    def set_node_state(self, node_id: str, metadata: NodeMetadata) -> None:
        """Set the state of a node and persist to Redis."""
        self._ensure_connected()
        
        key = self._get_key(node_id)
        data = {
            "node_id": metadata.node_id,
            "state": metadata.state.value,
            "visit_count": metadata.visit_count,
            "last_visited": metadata.last_visited,
            "metadata": metadata.metadata
        }
        
        try:
            pipe = self.client.pipeline()
            pipe.set(key, json.dumps(data))
            visited_key = self._get_visited_set_key()
            if metadata.state == NodeState.VISITED:
                pipe.sadd(visited_key, node_id)
            else:
                pipe.srem(visited_key, node_id)
            pipe.execute()
            logger.debug(
                "Redis: saved node %s state=%s",
                node_id[:36] + "..." if len(node_id) > 36 else node_id,
                metadata.state.value,
            )
        except Exception as e:
            logger.warning("Redis set_node_state failed for %s: %s", node_id[:50], e)
            raise
    
    def mark_visited(self, node_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a node as visited."""
        existing = self.get_node_state(node_id)
        
        if existing:
            new_metadata = NodeMetadata(
                node_id=node_id,
                state=NodeState.VISITED,
                visit_count=existing.visit_count + 1,
                last_visited=time.time(),
                metadata={**(existing.metadata or {}), **(metadata or {})}
            )
        else:
            new_metadata = NodeMetadata(
                node_id=node_id,
                state=NodeState.VISITED,
                visit_count=1,
                last_visited=time.time(),
                metadata=metadata or {}
            )
        
        self.set_node_state(node_id, new_metadata)
    
    def mark_not_visited(self, node_id: str) -> None:
        """Mark a node as not visited."""
        existing = self.get_node_state(node_id)
        
        new_metadata = NodeMetadata(
            node_id=node_id,
            state=NodeState.NOT_VISITED,
            visit_count=existing.visit_count if existing else 0,
            last_visited=existing.last_visited if existing else None,
            metadata=existing.metadata if existing else {}
        )
        
        self.set_node_state(node_id, new_metadata)
    
    def get_visited_nodes(self) -> List[str]:
        """Get all visited node IDs."""
        self._ensure_connected()
        
        visited_set_key = self._get_visited_set_key()
        return list(self.client.smembers(visited_set_key))
    
    def get_unvisited_nodes(self, all_node_ids: List[str]) -> List[str]:
        """Get all unvisited node IDs from a list."""
        visited = set(self.get_visited_nodes())
        return [node_id for node_id in all_node_ids if node_id not in visited]
    
    def clear_state(self) -> None:
        """Clear all state."""
        self._ensure_connected()
        
        # Get all keys with our prefix
        pattern = f"{self.key_prefix}*"
        keys = self.client.keys(pattern)
        
        if keys:
            self.client.delete(*keys)
            logger.info(f"Cleared {len(keys)} keys from Redis")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the traversal state."""
        self._ensure_connected()
        
        visited_nodes = self.get_visited_nodes()
        
        # Count nodes by state
        states = {}
        for node_id in visited_nodes:
            state = self.get_node_state(node_id)
            if state:
                state_value = state.state.value
                states[state_value] = states.get(state_value, 0) + 1
        
        return {
            "total_visited": len(visited_nodes),
            "states": states,
            "key_prefix": self.key_prefix
        }
