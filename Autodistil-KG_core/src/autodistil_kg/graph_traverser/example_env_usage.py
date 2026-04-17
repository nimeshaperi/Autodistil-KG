"""
Example: Using Environment Variables for Configuration.

This example shows how to use the env_config module to load configuration
from environment variables instead of hardcoding values.
"""
from autodistil_kg.graph_traverser import (
    GraphTraverser,
    GraphTraverserAgentConfig,
    TraversalConfig,
    DatasetGenerationConfig,
    TraversalStrategy
)
from autodistil_kg.graph_traverser.env_config import (
    load_env_file,
    get_graph_db_config_from_env,
    get_llm_config_from_env,
    get_state_storage_config_from_env
)


def example_with_env_variables():
    """Example using environment variables for configuration."""
    
    # Load environment variables from .env file
    # Make sure you have created a .env file from example.env
    try:
        load_env_file()
        print("✓ Loaded environment variables from .env file")
    except (ImportError, FileNotFoundError) as e:
        print(f"⚠ Warning: {e}")
        print("Continuing with default values...")
    
    # Create configurations from environment variables
    graph_db_config = get_graph_db_config_from_env()
    llm_config = get_llm_config_from_env()  # Auto-detects provider
    state_storage_config = get_state_storage_config_from_env()
    
    # Or specify LLM provider explicitly
    # llm_config = get_llm_config_from_env(provider="openai")
    
    # Configure traversal (can still be set in code)
    traversal_config = TraversalConfig(
        strategy=TraversalStrategy.BFS,
        max_nodes=500,
        max_depth=5
    )
    
    # Configure dataset generation
    dataset_config = DatasetGenerationConfig(
        seed_prompts=[
            "What information can you provide about this node?",
            "Describe the properties: {properties}"
        ],
        system_message="You are a helpful assistant.",
        output_format="jsonl",
        output_path="output/dataset.jsonl"
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
    return dataset


if __name__ == "__main__":
    example_with_env_variables()
