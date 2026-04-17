"""
Example usage of the Graph Traverser Agent.

This file demonstrates how to configure and use the graph traverser
to create CHATML compliant datasets from a knowledge graph.
"""
from autodistil_kg.graph_traverser import (
    GraphTraverser,
    GraphTraverserAgentConfig,
    GraphDatabaseConfig,
    LLMConfig,
    StateStorageConfig,
    TraversalConfig,
    DatasetGenerationConfig,
    TraversalStrategy,
    LLMProvider
)


def example_basic_usage():
    """Basic example of using the Graph Traverser."""
    
    # Configure graph database (Neo4j)
    graph_db_config = GraphDatabaseConfig(
        provider="neo4j",
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password",
        database="neo4j"
    )
    
    # Configure LLM (OpenAI)
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI.value,
        api_key="your_openai_api_key",
        model="gpt-4"
    )
    
    # Configure state storage (Redis)
    state_storage_config = StateStorageConfig(
        provider="redis",
        host="localhost",
        port=6379,
        db=0,
        key_prefix="graph_traverser:"
    )
    
    # Configure traversal
    traversal_config = TraversalConfig(
        strategy=TraversalStrategy.BFS,
        max_nodes=500,
        max_depth=5,
        seed_node_ids=["node1", "node2"]  # Optional: specific starting nodes
    )
    
    # Configure dataset generation
    dataset_config = DatasetGenerationConfig(
        seed_prompts=[
            "What information can you provide about this node with labels {labels}?",
            "Describe the properties and relationships of this node: {properties}",
            "Explain the context of this node in the knowledge graph."
        ],
        system_message="You are a helpful assistant that explains knowledge graph nodes.",
        output_format="jsonl",
        output_path="output/dataset.jsonl",
        include_metadata=True
    )
    
    # Create complete configuration
    config = GraphTraverserAgentConfig(
        graph_db=graph_db_config,
        llm=llm_config,
        state_storage=state_storage_config,
        traversal=traversal_config,
        dataset=dataset_config
    )
    
    # Create and run traverser
    traverser = GraphTraverser(config)
    dataset = traverser.traverse()
    
    print(f"Generated {len(dataset)} conversations")
    print(f"Statistics: {traverser.get_statistics()}")


def example_semantic_traversal():
    """Example using semantic-aware traversal."""
    
    config = GraphTraverserAgentConfig(
        graph_db=GraphDatabaseConfig(
            provider="neo4j",
            uri="bolt://localhost:7687",
            user="neo4j",
            password="your_password"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI.value,
            api_key="your_api_key",
            model="gpt-4"
        ),
        state_storage=StateStorageConfig(
            provider="redis",
            host="localhost",
            port=6379
        ),
        traversal=TraversalConfig(
            strategy=TraversalStrategy.SEMANTIC,  # Use semantic traversal
            max_nodes=200
        ),
        dataset=DatasetGenerationConfig(
            seed_prompts=["Explain this node: {properties}"],
            output_path="output/semantic_dataset.jsonl"
        )
    )
    
    traverser = GraphTraverser(config)
    dataset = traverser.traverse()
    return dataset


def example_with_gemini():
    """Example using Google Gemini."""
    
    config = GraphTraverserAgentConfig(
        graph_db=GraphDatabaseConfig(
            provider="neo4j",
            uri="bolt://localhost:7687",
            user="neo4j",
            password="your_password"
        ),
        llm=LLMConfig(
            provider=LLMProvider.GEMINI.value,
            project_id="your-gcp-project-id",
            location="us-central1",
            model="gemini-pro",
            credentials_path="/path/to/credentials.json"
        ),
        state_storage=StateStorageConfig(provider="redis"),
        traversal=TraversalConfig(
            strategy=TraversalStrategy.DFS,
            max_nodes=500
        ),
        dataset=DatasetGenerationConfig(
            seed_prompts=["Describe this node: {properties}"],
            output_path="output/gemini_dataset.jsonl"
        )
    )
    
    traverser = GraphTraverser(config)
    return traverser.traverse()


def example_with_local_llm():
    """Example using local LLM (Ollama)."""
    
    config = GraphTraverserAgentConfig(
        graph_db=GraphDatabaseConfig(
            provider="neo4j",
            uri="bolt://localhost:7687",
            user="neo4j",
            password="your_password"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OLLAMA.value,
            base_url="http://localhost:11434",
            model="llama2"
        ),
        state_storage=StateStorageConfig(provider="redis"),
        traversal=TraversalConfig(
            strategy=TraversalStrategy.RANDOM,
            max_nodes=500
        ),
        dataset=DatasetGenerationConfig(
            seed_prompts=["What is this node about? {properties}"],
            output_path="output/ollama_dataset.jsonl"
        )
    )
    
    traverser = GraphTraverser(config)
    return traverser.traverse()


if __name__ == "__main__":
    # Run basic example
    example_basic_usage()
