"""
Custom exception hierarchy for Autodistil-KG.

All exceptions inherit from :class:`AutodistilError` so callers can catch the
full family with a single ``except AutodistilError``.  More specific sub-types
carry structured context (stage name, provider name, etc.) so the API layer
can map them to appropriate HTTP status codes and the frontend can display
actionable messages.

Mapping to HTTP status codes (recommended for the API layer)::

    ConfigurationError  -> 422 Unprocessable Entity
    StageError          -> 500 Internal Server Error
    ProviderError       -> 502 Bad Gateway  (external service failure)
    PipelineCancelled   -> 499 Client Closed Request  (or 200 with cancelled flag)
"""


class AutodistilError(Exception):
    """Base class for all Autodistil-KG errors."""


class PipelineCancelled(AutodistilError):
    """Raised when a pipeline run is cancelled by the user."""


class ConfigurationError(AutodistilError):
    """Raised when configuration is invalid or incomplete.

    Examples: missing required credentials, invalid enum value, path does not
    exist when it should.
    """


class StageError(AutodistilError):
    """Raised when a pipeline stage fails during execution.

    Attributes:
        stage: Name of the stage that failed (e.g. ``"graph_traverser"``).
    """

    def __init__(self, message: str, stage: str = "unknown") -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class ProviderError(AutodistilError):
    """Raised when an external service provider fails.

    Attributes:
        provider: Name of the provider (e.g. ``"neo4j"``, ``"openai"``).
    """

    def __init__(self, message: str, provider: str = "unknown") -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class GraphDBError(ProviderError):
    """Raised when the graph database (Neo4j) encounters an error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, provider="neo4j")


class LLMError(ProviderError):
    """Raised when an LLM provider call fails."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message, provider=provider)


class StateStorageError(ProviderError):
    """Raised when the state storage backend (Redis) encounters an error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, provider="redis")
