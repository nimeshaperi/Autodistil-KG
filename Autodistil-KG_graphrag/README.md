# Autodistil-KG Graph RAG

Graph RAG implementation using LlamaIndex's `PropertyGraphIndex` with Neo4j, providing retrieval-augmented generation over an existing knowledge graph.

## Setup

```bash
cp .env.example .env
# Edit .env with your Neo4j and OpenAI credentials

poetry install
```

## Usage

Interactive mode:

```bash
poetry run autodistil-kg-graphrag
```

Single query:

```bash
poetry run autodistil-kg-graphrag --query "What are the main topics in the knowledge graph?"
```

## Architecture

This package connects to an existing Neo4j knowledge graph (populated by `Autodistil-KG_core`'s graph traverser) and provides three retrieval strategies:

- **VectorContextRetriever** -- embedding-based similarity search over graph nodes
- **LLMSynonymRetriever** -- LLM-powered keyword/synonym expansion for entity matching
- **TextToCypherRetriever** -- LLM generates Cypher queries from natural language

Retrieved context is passed to an LLM for answer generation.
