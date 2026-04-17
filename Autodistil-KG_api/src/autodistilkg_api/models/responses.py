"""Strictly typed Pydantic response models for the KG Pipeline API."""
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional


class HealthResponse(BaseModel):
    status: str
    service: str


class StageResultResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    message: Optional[str] = None


RunStatus = Literal["running", "completed", "failed", "cancelled", "unknown"]


class PipelineRunResultResponse(BaseModel):
    run_id: str
    status: RunStatus
    success: bool
    context: Optional[Dict[str, Any]] = None
    results: Optional[List[StageResultResponse]] = None
    error: Optional[str] = None
    stages: Optional[List[str]] = None
    current_stage: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    error: Optional[str] = None
    stages: Optional[List[str]] = None
    resumable: bool = False
    checkpoint_conversations: Optional[int] = None


class RunListResponse(BaseModel):
    runs: List[RunSummary]


class RunEventsResponse(BaseModel):
    run_id: str
    events: List[Dict[str, Any]]
    total: int


class InferenceLLMResponse(BaseModel):
    response: str
    provider: str
    model: Optional[str] = None


class InferenceGraphRAGResponse(BaseModel):
    answer: str
    source_nodes: List[Any] = []
    metadata: Dict[str, Any] = {}


class AvailableModel(BaseModel):
    run_id: str
    model_path: str
    label: str


class RegisteredModel(BaseModel):
    model_id: str
    model_path: str
    base_model: Optional[str] = None
    description: Optional[str] = None


class ModelsListResponse(BaseModel):
    models: List[AvailableModel]


class RegisteredModelsResponse(BaseModel):
    models: List[RegisteredModel]


class InferenceFinetunedResponse(BaseModel):
    response: str
    model_id: str


class FileUploadResponse(BaseModel):
    path: str
