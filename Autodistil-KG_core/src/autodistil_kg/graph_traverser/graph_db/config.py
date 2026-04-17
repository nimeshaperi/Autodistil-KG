"""
Graph Database Configuration Module.

This module defines configuration classes for graph database providers.
"""
from pydantic import BaseModel, field_validator
from typing import Dict, Any, Literal, Optional


class GraphDatabaseConfig(BaseModel):
    """Configuration for graph database connection."""
    provider: Literal["neo4j"] = "neo4j"
    uri: str
    user: str
    password: str
    database: Optional[str] = None
    additional_params: Dict[str, Any] = {}

    @field_validator("uri")
    @classmethod
    def uri_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Graph database URI must not be empty")
        return v
