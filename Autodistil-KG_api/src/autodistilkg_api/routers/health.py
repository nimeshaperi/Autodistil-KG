"""Health check endpoint."""
from fastapi import APIRouter

from ..models.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="kg-pipeline-api")
