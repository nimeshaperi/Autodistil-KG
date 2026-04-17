"""
Graph Traverser Configuration Module.

This module defines high-level configuration classes that compose
the individual module configurations.
"""
from pydantic import BaseModel, model_validator
from typing import Dict, Any, List, Literal, Optional
from enum import Enum

# Import module configs
from .graph_db.config import GraphDatabaseConfig
from autodistil_kg.llm.config import LLMConfig
from .state_storage.config import StateStorageConfig


class TraversalStrategy(str, Enum):
    """Enumeration of traversal strategies."""
    BFS = "bfs"  # Breadth-First Search
    DFS = "dfs"  # Depth-First Search
    RANDOM = "random"
    SEMANTIC = "semantic"  # Semantic-aware traversal using LLM
    REASONING = "reasoning"  # Deep multi-hop reasoning with subgraph exploration


class TraversalConfig(BaseModel):
    """Configuration for graph traversal."""
    strategy: TraversalStrategy
    max_nodes: Optional[int] = None
    max_depth: Optional[int] = None
    relationship_types: Optional[List[str]] = None
    node_labels: Optional[List[str]] = None
    seed_node_ids: Optional[List[str]] = None
    reasoning_depth: int = 2
    max_paths_per_node: int = 15
    path_batch_size: int = 5
    num_workers: int = 1
    additional_params: Dict[str, Any] = {}


class AlignmentConfig(BaseModel):
    """Configuration for aligning generated Q&A data with target requirements.

    Four dimensions of alignment:
    - Domain: steer traversal/generation toward a specific topic or field
    - Style: constrain answer tone, format, length, and target audience
    - Quality: post-generation LLM scoring to filter weak Q&A pairs
    - Reference: ground generation against supplied reference texts
    """
    # --- Domain alignment ---
    domain_focus: Optional[str] = None
    """Free-text description of the target domain (e.g. 'clinical pharmacology
    and drug–drug interactions').  Injected into synthesis and QA prompts."""
    domain_keywords: List[str] = []
    """Optional keywords that bias node selection and path prioritisation."""

    # --- Style / format alignment ---
    style_guide: Optional[str] = None
    """Prose instructions for answer tone and structure (e.g. 'Use concise
    bullet points suitable for a clinical reference card')."""
    target_audience: Optional[str] = None
    """Who the training data is for (e.g. 'medical students', 'senior engineers').
    Included in the system message when set."""
    max_answer_length: Optional[int] = None
    """Soft upper bound on answer length in words.  Communicated to the LLM as a
    guideline — not a hard truncation."""
    min_answer_length: Optional[int] = None
    """Soft lower bound on answer length in words."""

    # --- Quality alignment (post-generation gate) ---
    quality_filter: bool = False
    """When True, each generated Q&A pair is scored by the LLM and pairs below
    *quality_threshold* are discarded."""
    quality_threshold: float = 0.7
    """Minimum normalised score (0–1) to keep a Q&A pair.  Only used when
    *quality_filter* is True."""

    # --- Reference alignment ---
    reference_texts_path: Optional[str] = None
    """Path to a plain-text or JSONL file containing reference passages.  When
    set, relevant excerpts are injected into synthesis and QA prompts so the LLM
    can ground its output against authoritative material."""


class DatasetGenerationConfig(BaseModel):
    """Configuration for dataset generation."""
    seed_prompts: List[str] = []
    system_message: Optional[str] = None
    prompt_template: Optional[str] = None
    include_metadata: bool = True
    output_format: Literal["jsonl", "json"] = "jsonl"
    output_path: Optional[str] = None
    alignment: AlignmentConfig = AlignmentConfig()
    eval_rephrase: bool = True
    """When True, evaluation questions are rephrased by the LLM so they
    test comprehension of the same knowledge using different wording,
    structure, and emphasis.  This prevents the model from scoring high
    simply by memorising the exact sentence patterns it was trained on."""
    eval_sample_ratio: float = 0.2
    """Fraction of traversed nodes (0.0–1.0) to include in the eval set.
    Nodes are sampled randomly; their QA pairs are still included in the
    training dataset but the eval dataset gets a rephrased version of the
    question paired with the original reference answer."""
    additional_params: Dict[str, Any] = {}


class GraphTraverserAgentConfig(BaseModel):
    """Complete configuration for GraphTraverserAgent."""
    graph_db: GraphDatabaseConfig
    llm: LLMConfig
    state_storage: StateStorageConfig
    traversal: TraversalConfig
    dataset: DatasetGenerationConfig

    @model_validator(mode="after")
    def check_strategy(self) -> "GraphTraverserAgentConfig":
        if not self.traversal.strategy:
            raise ValueError("Traversal strategy must be specified")
        return self
