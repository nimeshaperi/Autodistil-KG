"""Configuration for the Graph RAG pipeline.

Loads settings from environment variables / .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


class RetrieverType(str, Enum):
    VECTOR = "vector"
    CYPHER = "cypher"
    SYNONYM = "synonym"


@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


@dataclass
class LLMConfig:
    api_key: str = ""
    model: str = "gpt-4"
    base_url: Optional[str] = None


@dataclass
class EmbeddingConfig:
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    base_url: Optional[str] = None


@dataclass
class RetrieverConfig:
    enabled: List[RetrieverType] = field(
        default_factory=lambda: [
            RetrieverType.VECTOR,
            RetrieverType.CYPHER,
            RetrieverType.SYNONYM,
        ]
    )
    similarity_top_k: int = 5
    path_depth: int = 2  # 2-hop neighbourhood expansion by default (set GRAPHRAG_PATH_DEPTH=3 to enable 3-hop)
    synonym_max_keywords: int = 10
    include_text: bool = True


@dataclass
class GraphRAGConfig:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)


def _parse_retrievers(raw: str) -> List[RetrieverType]:
    """Parse a comma-separated retriever list like 'vector,cypher,synonym'."""
    result: list[RetrieverType] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if token:
            result.append(RetrieverType(token))
    return result


def load_config(env_path: Optional[str | Path] = None) -> GraphRAGConfig:
    """Build a ``GraphRAGConfig`` from environment variables.

    If *env_path* is provided, that ``.env`` file is loaded first.
    """
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY", "")

    neo4j = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", Neo4jConfig.uri),
        user=os.getenv("NEO4J_USER", Neo4jConfig.user),
        password=os.getenv("NEO4J_PASSWORD", Neo4jConfig.password),
        database=os.getenv("NEO4J_DATABASE", Neo4jConfig.database),
    )

    llm = LLMConfig(
        api_key=openai_key,
        model=os.getenv("GRAPHRAG_LLM_MODEL", LLMConfig.model),
        base_url=os.getenv("GRAPHRAG_LLM_BASE_URL") or None,
    )

    embedding = EmbeddingConfig(
        api_key=openai_key,
        model=os.getenv("GRAPHRAG_EMBEDDING_MODEL", EmbeddingConfig.model),
        dimensions=int(
            os.getenv("GRAPHRAG_EMBEDDING_DIMENSIONS", str(EmbeddingConfig.dimensions))
        ),
        base_url=os.getenv("GRAPHRAG_EMBEDDING_BASE_URL") or None,
    )

    raw_retrievers = os.getenv("GRAPHRAG_RETRIEVERS", "vector,cypher,synonym")
    retriever = RetrieverConfig(
        enabled=_parse_retrievers(raw_retrievers),
        similarity_top_k=int(
            os.getenv("GRAPHRAG_SIMILARITY_TOP_K", str(RetrieverConfig.similarity_top_k))
        ),
        path_depth=int(
            os.getenv("GRAPHRAG_PATH_DEPTH", str(RetrieverConfig.path_depth))
        ),
    )

    return GraphRAGConfig(
        neo4j=neo4j,
        llm=llm,
        embedding=embedding,
        retriever=retriever,
    )
