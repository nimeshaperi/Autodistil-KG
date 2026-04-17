"""
GraphTraverser - High-level interface for graph traversal and dataset generation.
"""
import logging
from typing import Optional

from .graph_traverser_agent import GraphTraverserAgent
from .config import GraphTraverserAgentConfig
from .factory import create_graph_database, create_llm_client, create_state_storage
from .chatml import ChatMLDataset

logger = logging.getLogger(__name__)


class GraphTraverser:
    """
    High-level interface for graph traversal and CHATML dataset generation.
    
    This class provides a convenient wrapper around GraphTraverserAgent,
    automatically creating the necessary components from configuration.
    """
    
    def __init__(self, config: GraphTraverserAgentConfig):
        """
        Initialize GraphTraverser with configuration.
        
        Args:
            config: Complete configuration for the graph traverser
        """
        # Create components
        self.graph_db = create_graph_database(config.graph_db)
        self.llm_client = create_llm_client(config.llm)
        self.state_storage = create_state_storage(config.state_storage)
        
        # Create agent
        self.agent = GraphTraverserAgent(
            graph_db=self.graph_db,
            llm_client=self.llm_client,
            state_storage=self.state_storage,
            config=config
        )
        
        self.config = config
    
    def traverse(self) -> ChatMLDataset:
        """
        Start the graph traversal and dataset generation.
        
        Returns:
            ChatMLDataset containing all generated conversations
        """
        return self.agent.traverse()
    
    @property
    def eval_dataset_path(self) -> Optional[str]:
        """Path to the eval dataset (clean Q&A pairs) if one was generated."""
        if not self.agent._eval_samples or not self.config.dataset.output_path:
            return None
        from pathlib import Path
        p = Path(self.config.dataset.output_path)
        return str(p.parent / f"{p.stem}_eval{p.suffix}")

    def get_statistics(self) -> dict:
        """
        Get statistics about the traversal state.

        Returns:
            Dictionary with statistics
        """
        stats = self.state_storage.get_statistics()
        stats["dataset_size"] = len(self.agent.dataset)
        stats["visited_count"] = self.agent.visited_count
        return stats
