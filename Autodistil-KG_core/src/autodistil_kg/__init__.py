"""
autodistil_kg: Graph Traversing, ChatML Dataset Creation, FineTuning, Evaluation.

A 4-stage pipeline for generating fine-tuning datasets from Knowledge Graphs:

    Graph Traverser -> ChatML Converter -> FineTuner -> Evaluator

Each stage is modular and can be run separately or as part of the pipeline.

Quick start::

    from autodistil_kg.pipeline import Pipeline, PipelineConfig
    config = PipelineConfig(...)
    pipeline = Pipeline(config)
    context, results = pipeline.run()

Subpackages: ``chatml``, ``pipeline``, ``finetuner``, ``exceptions``.
The ``graph_traverser`` subpackage is optional (requires neo4j + redis);
import it explicitly to avoid pulling optional dependencies.
"""
from . import chatml
from . import pipeline
from . import finetuner
from .exceptions import (  # noqa: F401
    AutodistilError,
    PipelineCancelled,
    ConfigurationError,
    StageError,
    ProviderError,
    GraphDBError,
    LLMError,
    StateStorageError,
)
# graph_traverser optional: requires neo4j; use "from autodistil_kg.graph_traverser import ..."

__all__ = [
    "chatml",
    "pipeline",
    "finetuner",
    # Exceptions
    "AutodistilError",
    "PipelineCancelled",
    "ConfigurationError",
    "StageError",
    "ProviderError",
    "GraphDBError",
    "LLMError",
    "StateStorageError",
]
