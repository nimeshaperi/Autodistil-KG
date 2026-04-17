"""
Pipeline interfaces: stage contract and context passed between stages.

Core types:

- :class:`CancellationToken` -- thread-safe flag for cooperative cancellation.
- :class:`StageResult` -- success/failure envelope returned by every stage.
- :class:`PipelineContext` -- shared data carrier that flows through the pipeline.
- :class:`Stage` -- abstract base that every stage implementation extends.

The context carries both **file paths** (for disk-based hand-offs) and **typed
in-memory objects** (for zero-copy chaining).  Stages should prefer in-memory
objects when available and fall back to loading from paths.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import threading
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional


class CancellationToken:
    """Thread-safe cancellation token for stopping pipeline runs.

    Shared across all stages in a pipeline run.  Call :meth:`cancel` to
    request cooperative shutdown.  Stages should periodically check
    :attr:`is_cancelled` or call :meth:`raise_if_cancelled` inside loops.

    Example::

        token = CancellationToken()
        # In another thread:
        token.cancel()
        # In the stage:
        token.raise_if_cancelled()  # raises PipelineCancelled
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Check whether cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise ``PipelineCancelled`` if cancellation was requested."""
        if self._event.is_set():
            raise PipelineCancelled("Pipeline run was cancelled by user")


# Re-export from central exceptions module for backward compatibility.
from autodistil_kg.exceptions import PipelineCancelled  # noqa: F401


class StageResult(BaseModel):
    """Result of running a single pipeline stage.

    Every :meth:`Stage.run` call returns a ``StageResult``.  The pipeline
    stops on the first result where ``success`` is ``False``.

    Attributes:
        success: Whether the stage completed without error.
        output: Stage-specific output data (statistics, counts, etc.).
        error: Human-readable error description when ``success`` is False.
        metadata: Arbitrary metadata for logging or downstream consumers.
    """
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class PipelineContext(BaseModel):
    """Shared data carrier passed between pipeline stages.

    Holds **file paths** (set by stages or pre-populated for standalone runs)
    and **in-memory objects** (for zero-copy chaining when stages run in sequence).

    Stages read their inputs from the context and write their outputs back.
    This allows stages to be run in any order when paths are pre-configured,
    or chained efficiently via in-memory objects.

    Attributes:
        chatml_dataset_path: Path to the raw CHATML dataset JSONL file.
        prepared_dataset_path: Path to the fine-tuning-ready prepared JSONL.
        eval_dataset_path: Path to the clean Q&A pairs reserved for evaluation.
        model_output_path: Path to the fine-tuned model/adapter directory.
        eval_report_path: Path to the evaluation report JSON.
        chatml_dataset: In-memory ChatMLDataset (set by GraphTraverserStage).
        finetuner_model_name: Base model name used by the finetuner (set by FineTunerStage).
        prepared_messages: Prepared message dicts (set by ChatMLConverterStage).
        cancel_token: Shared cancellation token for cooperative shutdown.
        extra: Escape hatch for custom stage data.  Prefer named fields above.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- File paths (set by stages or user for standalone runs) --
    chatml_dataset_path: Optional[str] = None
    prepared_dataset_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    model_output_path: Optional[str] = None
    eval_report_path: Optional[str] = None

    # -- In-memory objects (optional; used when chaining without writing to disk) --
    # Type is ChatMLDataset at runtime; declared as Any here to avoid circular
    # imports.  Stages that produce or consume this field import the concrete
    # type themselves.
    chatml_dataset: Any = Field(default=None, exclude=True)
    prepared_dataset: Optional[List[Dict[str, Any]]] = Field(default=None, exclude=True)

    # -- Typed inter-stage fields (promoted from extra) --
    finetuner_model_name: Optional[str] = None
    prepared_messages: Optional[List[Dict[str, Any]]] = Field(default=None, exclude=True)

    # -- Cancellation --
    cancel_token: Optional[CancellationToken] = Field(default=None, exclude=True)

    # -- Escape hatch --
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extras for custom stage data.  Prefer named fields.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the context to a JSON-safe dict (paths and extras only)."""
        return {
            "chatml_dataset_path": self.chatml_dataset_path,
            "prepared_dataset_path": self.prepared_dataset_path,
            "eval_dataset_path": self.eval_dataset_path,
            "model_output_path": self.model_output_path,
            "eval_report_path": self.eval_report_path,
            "finetuner_model_name": self.finetuner_model_name,
            "extra": self.extra,
        }


class Stage(ABC):
    """Abstract base for a pipeline stage.

    Every stage is **configurable** (receives its config at ``__init__``) and
    **independently runnable** (reads from and writes to :class:`PipelineContext`).

    To implement a custom stage::

        class MyStage(Stage):
            name = "my_stage"

            def __init__(self, config: MyStageConfig):
                self.config = config

            def run(self, context: PipelineContext) -> StageResult:
                # Read inputs from context
                data = load(context.chatml_dataset_path)
                # Do work ...
                # Write outputs back to context
                context.model_output_path = "/path/to/output"
                return StageResult(success=True, output={"items": 42})
    """

    name: str = "base"

    @abstractmethod
    def run(self, context: PipelineContext) -> StageResult:
        """Execute the stage.

        Read inputs from *context*, write outputs back to *context*.
        Return a :class:`StageResult` indicating success or failure.
        """
        pass

    def validate_config(self) -> None:
        """Override to validate stage-specific config.  Raise ``ValueError`` if invalid."""
        pass
