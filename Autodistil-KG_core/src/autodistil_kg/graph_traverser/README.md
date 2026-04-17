# Graph Traverser Agent

A semantic-aware agent that traverses Knowledge Graphs and creates CHATML compliant datasets for fine-tuning.

## Features

- **Multiple Graph Database Support**: Extensible interface with Neo4j implementation
- **Multiple LLM Support**: OpenAI (ChatGPT), Google Gemini (Vertex AI), Anthropic Claude, Ollama, vLLM
- **State Management**: Redis-based caching for visited/unvisited nodes
- **CHATML Compliance**: Generates datasets in CHATML format
- **Semantic Traversal**: LLM-powered node selection for intelligent traversal
- **Type-Safe Configuration**: Dataclass-based configuration with validation
- **Extensible Architecture**: Easy to add new providers and strategies

## Architecture

```
graph_traverser/
├── interfaces.py          # Abstract base classes
├── config.py              # Type-safe configuration classes
├── factory.py             # Factory functions for creating instances
├── graph_traverser_agent.py  # Main agent implementation
├── graph_traverser.py     # High-level interface
├── chatml.py              # CHATML dataset generation
├── graph_db/              # Graph database implementations
│   ├── neo4j_provider.py
│   └── __init__.py
├── llm/                   # LLM client implementations
│   ├── openai_client.py
│   ├── gemini_client.py
│   ├── claude_client.py
│   ├── ollama_client.py
│   ├── vllm_client.py
│   └── __init__.py
└── state_storage/         # State storage implementations
    ├── redis_storage.py
    └── __init__.py
```

## Quick Start

### Basic Usage

```python
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

# Configure components
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
        strategy=TraversalStrategy.BFS,
        max_nodes=500
    ),
    dataset=DatasetGenerationConfig(
        seed_prompts=[
            "What information can you provide about this node?",
            "Describe the properties: {properties}"
        ],
        output_path="output/dataset.jsonl"
    )
)

# Create and run
traverser = GraphTraverser(config)
dataset = traverser.traverse()
```

## Configuration

### Graph Database Configuration

```python
GraphDatabaseConfig(
    provider="neo4j",  # Currently only "neo4j"
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    database="neo4j"  # Optional
)
```

### LLM Configuration

#### OpenAI
```python
LLMConfig(
    provider=LLMProvider.OPENAI.value,
    api_key="sk-...",
    model="gpt-4"
)
```

#### Google Gemini
```python
LLMConfig(
    provider=LLMProvider.GEMINI.value,
    project_id="your-gcp-project",
    location="us-central1",
    model="gemini-pro",
    credentials_path="/path/to/credentials.json"
)
```

#### Anthropic Claude
```python
LLMConfig(
    provider=LLMProvider.CLAUDE.value,
    api_key="sk-ant-...",
    model="claude-3-opus-20240229"
)
```

#### Ollama (Local)
```python
LLMConfig(
    provider=LLMProvider.OLLAMA.value,
    base_url="http://localhost:11434",
    model="llama2"
)
```

#### vLLM (Local)
```python
LLMConfig(
    provider=LLMProvider.VLLM.value,
    base_url="http://localhost:8000",
    model="mistral-7b"
)
```

### State Storage Configuration

```python
StateStorageConfig(
    provider="redis",
    host="localhost",
    port=6379,
    db=0,
    password=None,  # Optional
    key_prefix="graph_traverser:"
)
```

### Traversal Configuration

```python
TraversalConfig(
    strategy=TraversalStrategy.BFS,  # BFS, DFS, RANDOM, or SEMANTIC
    max_nodes=500,                   # Maximum nodes to process (increase for larger datasets)
    max_depth=5,                     # Maximum depth
    relationship_types=["AUTHORED", "PUBLISHED_IN"],  # Filter relationships
    node_labels=["Author", "Publication"],            # Filter nodes
    seed_node_ids=["node1", "node2"]                  # Starting nodes
)
```

### Dataset Generation Configuration

```python
DatasetGenerationConfig(
    seed_prompts=[
        "Explain this node: {properties}",
        "What is the context of {labels}?"
    ],
    system_message="You are a helpful assistant.",
    prompt_template="Custom template: {properties}",  # Optional
    include_metadata=True,
    output_format="jsonl",  # "jsonl" or "json"
    output_path="output/dataset.jsonl"
)
```

## Traversal Strategies

1. **BFS (Breadth-First Search)**: Explores nodes level by level
2. **DFS (Depth-First Search)**: Explores nodes deeply before backtracking
3. **RANDOM**: Randomly selects nodes to explore
4. **SEMANTIC**: Uses LLM to intelligently select the most relevant next node

## Extending the System

### Adding a New Graph Database Provider

1. Create a new class inheriting from `GraphDatabase` in `graph_db/`
2. Implement all abstract methods
3. Update `factory.py` to include your provider

### Adding a New LLM Provider

1. Create a new class inheriting from `LLMClient` in `llm/`
2. Implement `generate()` and `stream_generate()` methods
3. Add provider to `LLMProvider` enum in `config.py`
4. Update `factory.py` to include your provider

### Adding a New State Storage Provider

1. Create a new class inheriting from `StateStorage` in `state_storage/`
2. Implement all abstract methods
3. Update `factory.py` to include your provider

## Dependencies

### Required
- Python 3.13+

### Optional (based on providers used)
- `neo4j` - For Neo4j graph database
- `redis` - For Redis state storage
- `openai` - For OpenAI/ChatGPT
- `anthropic` - For Claude
- `google-cloud-aiplatform` - For Gemini
- `requests` - For Ollama and vLLM

## Output Format

The generated dataset follows CHATML format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain this node..."},
    {"role": "assistant", "content": "This node represents..."}
  ],
  "metadata": {
    "node_id": "123",
    "labels": ["Author"],
    "depth": 2,
    "timestamp": 1234567890.0
  }
}
```

## Examples

See `example_usage.py` for complete examples of:
- Basic usage
- Semantic traversal
- Different LLM providers
- Local LLM usage
