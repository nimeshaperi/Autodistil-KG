"""
Typed event models for the Autodistil-KG pipeline.

Every event emitted by pipeline stages (traversal, fine-tuning, evaluation)
and by the API orchestration layer (lifecycle events) is defined here as a
Pydantic model.  This module is the **single source of truth** for the event
contract that flows:

    Core (emit) -> API (intercept & forward) -> Client (render)

Usage::

    from autodistil_kg.pipeline.events import TraversalNodeStart, FinetunerTrainingStep

    # Inspect all event schemas
    from autodistil_kg.pipeline.events import get_event_schemas
    schemas = get_event_schemas()
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class PipelineEvent(BaseModel):
    """Base class for all pipeline events."""
    event: str
    """Top-level event channel (e.g. 'traversal_progress', 'finetuner_progress')."""
    type: str
    """Sub-type within the channel (e.g. 'node_start', 'training_step')."""


# ===========================================================================
# Traversal events  (channel: "traversal_progress")
# ===========================================================================

class TraversalEvent(PipelineEvent):
    """Base for all graph-traversal events."""
    event: Literal["traversal_progress"] = "traversal_progress"


class TraversalStart(TraversalEvent):
    """Emitted once at the start of a traversal run."""
    type: Literal["traversal_start"] = "traversal_start"
    strategy: str
    seed_nodes: int
    max_nodes: Optional[int] = None
    max_depth: Optional[int] = None
    num_workers: int = 1


class TraversalComplete(TraversalEvent):
    """Emitted once when traversal finishes successfully."""
    type: Literal["traversal_complete"] = "traversal_complete"
    visited: int
    dataset_size: int
    strategy: str
    alignment: Optional[Dict[str, Any]] = None


class TraversalNodeStart(TraversalEvent):
    """Emitted when the agent begins processing a graph node."""
    type: Literal["node_start"] = "node_start"
    node_id: str
    depth: int
    visited: int
    total: int
    step: str
    reasoning_depth: int = 0


class SubgraphLoaded(TraversalEvent):
    """Emitted after loading a node's local subgraph (center + neighbors + paths)."""
    type: Literal["subgraph_loaded"] = "subgraph_loaded"
    node_id: str
    labels: List[str] = []
    node_count: int = 0
    edge_count: int = 0
    path_count: int = 0
    visited: int = 0
    total: int = 0
    center: Optional[Dict[str, Any]] = None
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []


class PathReasoning(TraversalEvent):
    """Emitted when the agent starts reasoning over relationship paths."""
    type: Literal["path_reasoning"] = "path_reasoning"
    node_id: str
    path_index: int
    total_paths: int
    batch_size: int = 1
    step: str = ""


class Synthesis(TraversalEvent):
    """Emitted when the agent synthesises path analyses into a coherent answer."""
    type: Literal["synthesis"] = "synthesis"
    node_id: str
    path_analyses_count: int
    step: str = ""


class QAGeneration(TraversalEvent):
    """Emitted when the agent generates Q&A pairs for a node."""
    type: Literal["qa_generation"] = "qa_generation"
    node_id: str
    step: str = ""


class QualityScored(TraversalEvent):
    """Emitted after quality scoring a generated Q&A pair."""
    type: Literal["quality_scored"] = "quality_scored"
    node_id: str
    scores: Dict[str, float] = {}
    threshold: float = 0.0
    passed: bool = True


class TraversalNodeDone(TraversalEvent):
    """Emitted when processing of a single node is complete."""
    type: Literal["node_done"] = "node_done"
    node_id: str
    visited: int = 0
    total: int = 0
    dataset_size: int = 0
    labels: List[str] = []
    status: Optional[str] = None
    error: Optional[str] = None
    quality_scores: Optional[Dict[str, float]] = None
    quality_filtered_total: Optional[int] = None
    paths_analyzed: Optional[int] = None


# All traversal event sub-types
TraversalEventTypes = Union[
    TraversalStart,
    TraversalComplete,
    TraversalNodeStart,
    SubgraphLoaded,
    PathReasoning,
    Synthesis,
    QAGeneration,
    QualityScored,
    TraversalNodeDone,
]


# ===========================================================================
# Finetuner events  (channel: "finetuner_progress")
# ===========================================================================

class FinetunerEvent(PipelineEvent):
    """Base for all fine-tuning events."""
    event: Literal["finetuner_progress"] = "finetuner_progress"


class FinetunerLoadingModel(FinetunerEvent):
    """Emitted when the finetuner starts loading a base model."""
    type: Literal["loading_model"] = "loading_model"
    model_name: str


class FinetunerModelLoaded(FinetunerEvent):
    """Emitted when the base model has been loaded successfully."""
    type: Literal["model_loaded"] = "model_loaded"
    model_name: str


class FinetunerLoadingData(FinetunerEvent):
    """Emitted when the finetuner starts loading training data."""
    type: Literal["loading_data"] = "loading_data"


class FinetunerPreparingData(FinetunerEvent):
    """Emitted when the finetuner is tokenising/preparing data splits."""
    type: Literal["preparing_data"] = "preparing_data"
    train_samples: int = 0
    eval_samples: int = 0


class FinetunerDataReady(FinetunerEvent):
    """Emitted when training data is prepared and ready."""
    type: Literal["data_ready"] = "data_ready"
    train_samples: int = 0
    eval_samples: int = 0


class FinetunerTrainingStart(FinetunerEvent):
    """Emitted when training begins."""
    type: Literal["training_start"] = "training_start"
    total_epochs: int
    total_steps: int
    batch_size: int
    learning_rate: float


class FinetunerTrainingStep(FinetunerEvent):
    """Emitted after each training step with loss and gradient info."""
    type: Literal["training_step"] = "training_step"
    step: int
    total_steps: int
    epoch: Optional[float] = None
    total_epochs: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None


class FinetunerEpochDone(FinetunerEvent):
    """Emitted at the end of each training epoch."""
    type: Literal["epoch_done"] = "epoch_done"
    epoch: int
    total_epochs: int
    step: int
    total_steps: int


class FinetunerTrainingEnd(FinetunerEvent):
    """Emitted when training finishes."""
    type: Literal["training_end"] = "training_end"
    total_steps: int
    train_loss: Optional[float] = None


class FinetunerSavingModel(FinetunerEvent):
    """Emitted when the finetuner starts saving the trained adapter."""
    type: Literal["saving_model"] = "saving_model"
    output_dir: str


class FinetunerModelSaved(FinetunerEvent):
    """Emitted when the trained adapter has been saved to disk."""
    type: Literal["model_saved"] = "model_saved"
    output_dir: str


# All finetuner event sub-types
FinetunerEventTypes = Union[
    FinetunerLoadingModel,
    FinetunerModelLoaded,
    FinetunerLoadingData,
    FinetunerPreparingData,
    FinetunerDataReady,
    FinetunerTrainingStart,
    FinetunerTrainingStep,
    FinetunerEpochDone,
    FinetunerTrainingEnd,
    FinetunerSavingModel,
    FinetunerModelSaved,
]


# ===========================================================================
# Evaluator events  (channel: "eval_progress")
# ===========================================================================

class EvalEvent(PipelineEvent):
    """Base for all evaluation events."""
    event: Literal["eval_progress"] = "eval_progress"


class EvalStart(EvalEvent):
    """Emitted when evaluation begins."""
    type: Literal["eval_start"] = "eval_start"
    systems: List[Dict[str, str]] = []
    metrics: List[str] = []
    eval_dataset_path: str = ""


class EvalDatasetLoaded(EvalEvent):
    """Emitted after the evaluation dataset is loaded."""
    type: Literal["dataset_loaded"] = "dataset_loaded"
    num_samples: int = 0


class EvalSystemStart(EvalEvent):
    """Emitted when evaluation begins for a particular system (distilled, base, graphrag)."""
    type: Literal["system_start"] = "system_start"
    system_id: str
    system_label: Optional[str] = None
    kind: str = ""
    system_index: int = 0
    total_systems: int = 0


class EvalPredictionsStart(EvalEvent):
    """Emitted when prediction generation begins for a system."""
    type: Literal["predictions_start"] = "predictions_start"
    system_id: str
    total_samples: int = 0
    resumed_count: int = 0


class EvalPredictionDone(EvalEvent):
    """Emitted after each individual prediction completes."""
    type: Literal["prediction_done"] = "prediction_done"
    system_id: str
    sample_index: int = 0
    total_samples: int = 0
    latency_sec: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    error: Optional[str] = None


class EvalSystemDone(EvalEvent):
    """Emitted when all predictions for a system are complete."""
    type: Literal["system_done"] = "system_done"
    system_id: str
    total_predictions: int = 0
    resumed: Optional[bool] = None


class EvalSystemError(EvalEvent):
    """Emitted when a system encounters a fatal error during evaluation."""
    type: Literal["system_error"] = "system_error"
    system_id: str
    error: str = ""


class EvalVllmStarting(EvalEvent):
    """Emitted just before the vLLM server begins loading the model."""
    type: Literal["vllm_starting"] = "vllm_starting"
    model: str = ""


class EvalVllmReady(EvalEvent):
    """Emitted when the vLLM server is healthy and ready to serve."""
    type: Literal["vllm_ready"] = "vllm_ready"
    model: str = ""
    url: str = ""


class EvalScoringStart(EvalEvent):
    """Emitted when metric scoring begins."""
    type: Literal["scoring_start"] = "scoring_start"
    total_samples: int = 0
    scorers: List[str] = []


class EvalScoringDone(EvalEvent):
    """Emitted after each sample is scored."""
    type: Literal["scoring_done"] = "scoring_done"
    sample_index: int = 0
    total_samples: int = 0
    scores: Dict[str, Dict[str, Optional[float]]] = {}


class EvalComplete(EvalEvent):
    """Emitted when evaluation finishes with aggregate results."""
    type: Literal["eval_complete"] = "eval_complete"
    num_samples: int = 0
    active_systems: List[str] = []
    aggregate_metrics: Dict[str, Dict[str, float]] = {}
    aggregate_performance: Dict[str, Dict[str, Any]] = {}
    report_path: str = ""


# All evaluator event sub-types
EvalEventTypes = Union[
    EvalStart,
    EvalDatasetLoaded,
    EvalVllmStarting,
    EvalVllmReady,
    EvalSystemStart,
    EvalPredictionsStart,
    EvalPredictionDone,
    EvalSystemDone,
    EvalSystemError,
    EvalScoringStart,
    EvalScoringDone,
    EvalComplete,
]


# ===========================================================================
# Pipeline lifecycle events  (channel: varies per event)
# ===========================================================================

class RunStart(PipelineEvent):
    """Emitted by the API when a pipeline run begins."""
    event: Literal["run_start"] = "run_start"
    type: Literal["run_start"] = "run_start"
    run_id: str
    resumed_from: Optional[str] = None


class PipelineStart(PipelineEvent):
    """Emitted by the API when stage execution begins."""
    event: Literal["pipeline_start"] = "pipeline_start"
    type: Literal["pipeline_start"] = "pipeline_start"
    stages: List[str]


class StageStart(PipelineEvent):
    """Emitted at the start of each pipeline stage."""
    event: Literal["stage_start"] = "stage_start"
    type: Literal["stage_start"] = "stage_start"
    stage: str


class StageEnd(PipelineEvent):
    """Emitted at the end of each pipeline stage."""
    event: Literal["stage_end"] = "stage_end"
    type: Literal["stage_end"] = "stage_end"
    stage: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class PipelineDone(PipelineEvent):
    """Emitted when the entire pipeline run finishes."""
    event: Literal["done"] = "done"
    type: Literal["done"] = "done"
    success: bool
    cancelled: bool = False
    context: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []


class PipelineError(PipelineEvent):
    """Emitted when the pipeline encounters a fatal error."""
    event: Literal["error"] = "error"
    type: Literal["error"] = "error"
    message: str


class LogEvent(PipelineEvent):
    """A regular log message forwarded to the client."""
    event: Literal["log"] = "log"
    type: Literal["log"] = "log"
    level: str
    logger: str
    message: str


class StopAcknowledged(PipelineEvent):
    """Emitted when a stop request has been received and processing."""
    event: Literal["stop_acknowledged"] = "stop_acknowledged"
    type: Literal["stop_acknowledged"] = "stop_acknowledged"
    run_id: str


# All lifecycle event sub-types
LifecycleEventTypes = Union[
    RunStart,
    PipelineStart,
    StageStart,
    StageEnd,
    PipelineDone,
    PipelineError,
    LogEvent,
    StopAcknowledged,
]


# ===========================================================================
# Aggregate union of all events
# ===========================================================================

AllEvents = Union[
    TraversalEventTypes,
    FinetunerEventTypes,
    EvalEventTypes,
    LifecycleEventTypes,
]


# ===========================================================================
# Schema export helper
# ===========================================================================

_ALL_EVENT_MODELS = [
    # Traversal
    TraversalStart,
    TraversalComplete,
    TraversalNodeStart,
    SubgraphLoaded,
    PathReasoning,
    Synthesis,
    QAGeneration,
    QualityScored,
    TraversalNodeDone,
    # Finetuner
    FinetunerLoadingModel,
    FinetunerModelLoaded,
    FinetunerLoadingData,
    FinetunerPreparingData,
    FinetunerDataReady,
    FinetunerTrainingStart,
    FinetunerTrainingStep,
    FinetunerEpochDone,
    FinetunerTrainingEnd,
    FinetunerSavingModel,
    FinetunerModelSaved,
    # Evaluator
    EvalStart,
    EvalDatasetLoaded,
    EvalSystemStart,
    EvalPredictionsStart,
    EvalPredictionDone,
    EvalSystemDone,
    EvalSystemError,
    EvalScoringStart,
    EvalScoringDone,
    EvalComplete,
    # Lifecycle
    RunStart,
    PipelineStart,
    StageStart,
    StageEnd,
    PipelineDone,
    PipelineError,
    LogEvent,
    StopAcknowledged,
]


def get_event_schemas() -> Dict[str, Any]:
    """Return JSON Schema for every event model, keyed by class name.

    Useful for generating TypeScript types or serving via the API's
    ``/schemas/events`` endpoint.
    """
    return {m.__name__: m.model_json_schema() for m in _ALL_EVENT_MODELS}
