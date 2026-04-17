"""
Pipeline Module.

Orchestrates configurable stages: Graph Traverser -> ChatML Converter -> FineTuner -> Evaluator.
Each stage can be run independently or as part of the full pipeline.

Key classes:

- :class:`Pipeline` -- runs stages in sequence, manages logging and cancellation.
- :class:`Stage` -- abstract base for a single pipeline stage.
- :class:`PipelineContext` -- shared data carrier passed between stages.
- :class:`StageResult` -- success/failure result returned by every stage.

Events:

Stages emit typed events (see :mod:`autodistil_kg.pipeline.events`) that flow
through the API to the frontend for real-time visualisation.  Import
:func:`get_event_schemas` to inspect all event types programmatically.
"""
from .interfaces import Stage, StageResult, PipelineContext
from .config import (
    StageId,
    STAGE_ORDER,
    PipelineConfig,
    GraphTraverserStageConfig,
    ChatMLConverterStageConfig,
    FineTunerStageConfig,
    EvaluatorStageConfig,
)
from .events import (
    PipelineEvent,
    # Traversal events
    TraversalStart,
    TraversalComplete,
    TraversalNodeStart,
    SubgraphLoaded,
    PathReasoning,
    Synthesis,
    QAGeneration,
    QualityScored,
    TraversalNodeDone,
    # Finetuner events
    FinetunerTrainingStart,
    FinetunerTrainingStep,
    FinetunerTrainingEnd,
    # Evaluator events
    EvalStart,
    EvalComplete,
    EvalPredictionDone,
    # Lifecycle events
    RunStart,
    PipelineStart,
    StageStart,
    StageEnd,
    PipelineDone,
    PipelineError,
    LogEvent,
    # Schema helper
    get_event_schemas,
)
from .pipeline import Pipeline
from .stages.graph_traverser_stage import GraphTraverserStage
from .stages.chatml_converter_stage import ChatMLConverterStage
from .stages.finetuner_stage import FineTunerStage
from .stages.evaluator_stage import EvaluatorStage

__all__ = [
    # Interfaces
    "Stage",
    "StageResult",
    "PipelineContext",
    # Config
    "StageId",
    "STAGE_ORDER",
    "PipelineConfig",
    "GraphTraverserStageConfig",
    "ChatMLConverterStageConfig",
    "FineTunerStageConfig",
    "EvaluatorStageConfig",
    # Events (commonly used subset -- full set in pipeline.events)
    "PipelineEvent",
    "TraversalStart",
    "TraversalComplete",
    "TraversalNodeStart",
    "SubgraphLoaded",
    "PathReasoning",
    "Synthesis",
    "QAGeneration",
    "QualityScored",
    "TraversalNodeDone",
    "FinetunerTrainingStart",
    "FinetunerTrainingStep",
    "FinetunerTrainingEnd",
    "EvalStart",
    "EvalComplete",
    "EvalPredictionDone",
    "RunStart",
    "PipelineStart",
    "StageStart",
    "StageEnd",
    "PipelineDone",
    "PipelineError",
    "LogEvent",
    "get_event_schemas",
    # Runner
    "Pipeline",
    # Stages
    "GraphTraverserStage",
    "ChatMLConverterStage",
    "FineTunerStage",
    "EvaluatorStage",
]
