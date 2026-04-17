"""Schema introspection endpoint — serves JSON schemas for all API types."""
from fastapi import APIRouter
from typing import Any, Dict

from ..models.requests import (
    PipelineRunRequest,
    GraphTraverserConfigRequest,
    ChatMLConverterConfigRequest,
    FineTunerConfigRequest,
    EvaluatorConfigRequest,
    InferenceLLMRequest,
    InferenceGraphRAGRequest,
    Neo4jConfigRequest,
    RedisConfigRequest,
    LLMConfigRequest,
    TraversalConfigRequest,
    DatasetConfigRequest,
    AlignmentConfigRequest,
    GraphRAGConfigRequest,
)
from ..models.responses import (
    HealthResponse,
    PipelineRunResultResponse,
    RunListResponse,
    RunEventsResponse,
    InferenceLLMResponse,
    InferenceGraphRAGResponse,
    ModelsListResponse,
    StageResultResponse,
)

router = APIRouter()

_MODELS = [
    # Requests
    PipelineRunRequest,
    GraphTraverserConfigRequest,
    ChatMLConverterConfigRequest,
    FineTunerConfigRequest,
    EvaluatorConfigRequest,
    InferenceLLMRequest,
    InferenceGraphRAGRequest,
    Neo4jConfigRequest,
    RedisConfigRequest,
    LLMConfigRequest,
    TraversalConfigRequest,
    DatasetConfigRequest,
    AlignmentConfigRequest,
    GraphRAGConfigRequest,
    # Responses
    HealthResponse,
    PipelineRunResultResponse,
    RunListResponse,
    RunEventsResponse,
    InferenceLLMResponse,
    InferenceGraphRAGResponse,
    ModelsListResponse,
    StageResultResponse,
]


@router.get("/schemas")
def get_schemas() -> Dict[str, Any]:
    """Return JSON schemas for all API request and response types.

    Useful for generating client-side types or validating payloads.
    """
    return {m.__name__: m.model_json_schema() for m in _MODELS}


@router.get("/schemas/events")
def get_event_schemas() -> Dict[str, Any]:
    """Return JSON schemas for all pipeline event types.

    Events are the typed messages streamed over the WebSocket during a
    pipeline run.  Three channels exist:

    - ``traversal_progress`` -- graph traversal events (node visits, paths, Q&A)
    - ``finetuner_progress`` -- fine-tuning events (epochs, loss, saving)
    - ``eval_progress`` -- evaluation events (predictions, scoring, metrics)

    Plus lifecycle events: ``run_start``, ``pipeline_start``, ``stage_start``,
    ``stage_end``, ``done``, ``error``, ``log``, ``stop_acknowledged``.
    """
    from autodistil_kg.pipeline.events import get_event_schemas as _get
    return _get()
