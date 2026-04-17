"""Export JSON schemas for all Pydantic config and event models.

Two helpers are provided:

- :func:`get_all_schemas` -- config model schemas (pipeline, stages, providers).
- :func:`get_event_schemas` -- re-exported from :mod:`pipeline.events` for convenience.

Together they cover every typed Pydantic model in the package, making it easy
to generate documentation or derive TypeScript types for the frontend.
"""
from typing import Dict, Any

from .pipeline.config import (
    PipelineConfig,
    GraphTraverserStageConfig,
    ChatMLConverterStageConfig,
    FineTunerStageConfig,
    EvaluatorStageConfig,
)
from .pipeline.interfaces import PipelineContext, StageResult
from .pipeline.events import get_event_schemas  # noqa: F401 -- re-export
from .graph_traverser.config import (
    GraphTraverserAgentConfig,
    TraversalConfig,
    DatasetGenerationConfig,
)
from .graph_traverser.graph_db.config import GraphDatabaseConfig
from .graph_traverser.state_storage.config import StateStorageConfig
from .llm.config import LLMConfig
from .finetuner.config import UnslothFineTunerConfig
from .eval.evalg_adapter import EvalSystemConfig
from .chatml.dataset import ChatMLMessage, ChatMLConversation


def get_all_schemas() -> Dict[str, Any]:
    """Return JSON schemas for all config models, keyed by class name.

    Does **not** include event schemas -- use :func:`get_event_schemas` for
    those, or call :func:`get_combined_schemas` to get everything in one dict.
    """
    models = [
        PipelineConfig,
        GraphTraverserStageConfig,
        ChatMLConverterStageConfig,
        FineTunerStageConfig,
        EvaluatorStageConfig,
        PipelineContext,
        StageResult,
        GraphTraverserAgentConfig,
        TraversalConfig,
        DatasetGenerationConfig,
        GraphDatabaseConfig,
        StateStorageConfig,
        LLMConfig,
        UnslothFineTunerConfig,
        EvalSystemConfig,
        ChatMLMessage,
        ChatMLConversation,
    ]
    return {m.__name__: m.model_json_schema() for m in models}


def get_combined_schemas() -> Dict[str, Any]:
    """Return JSON schemas for all config models **and** all event models."""
    schemas = get_all_schemas()
    schemas.update(get_event_schemas())
    return schemas
