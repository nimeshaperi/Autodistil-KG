"""
Pipeline runner: run all stages or individual stages with linked I/O.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .interfaces import CancellationToken, PipelineCancelled, PipelineContext, StageResult
from .config import PipelineConfig
from .stages.graph_traverser_stage import GraphTraverserStage
from .stages.chatml_converter_stage import ChatMLConverterStage
from .stages.finetuner_stage import FineTunerStage
from .stages.evaluator_stage import EvaluatorStage

logger = logging.getLogger(__name__)

STAGE_NAMES = ("graph_traverser", "chatml_converter", "finetuner", "evaluator")


class Pipeline:
    """
    Orchestrates stages: Graph Traverser -> ChatML Converter -> FineTuner -> Evaluator.
    Each stage can be run separately (standalone) or as part of the full pipeline.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._stages: Dict[str, object] = {}
        self._build_stages()
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Set up log level and optional file logging for the pipeline run."""
        level = getattr(logging, self.config.log_level, logging.INFO)
        logging.getLogger("autodistil_kg").setLevel(level)

        self._file_handler: Optional[logging.FileHandler] = None
        # Remove stale file handlers left by previous pipeline runs
        pkg_logger = logging.getLogger("autodistil_kg")
        for h in pkg_logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                pkg_logger.removeHandler(h)
        log_dir = self.config.log_dir
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"pipeline_{ts}.log"
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setLevel(logging.DEBUG)  # File always captures everything
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logging.getLogger("autodistil_kg").addHandler(handler)
            self._file_handler = handler
            logger.info("File logging enabled: %s", log_file)

        logger.debug("Log level set to %s", self.config.log_level)

    def _build_stages(self) -> None:
        cfg = self.config
        if cfg.graph_traverser is not None:
            self._stages["graph_traverser"] = GraphTraverserStage(cfg.graph_traverser)
        if cfg.chatml_converter is not None:
            self._stages["chatml_converter"] = ChatMLConverterStage(cfg.chatml_converter)
        if cfg.finetuner is not None:
            self._stages["finetuner"] = FineTunerStage(cfg.finetuner)
        if cfg.evaluator is not None:
            self._stages["evaluator"] = EvaluatorStage(cfg.evaluator)

    def _resolve_output_dir(self, stage_name: str) -> Optional[str]:
        base = self.config.output_dir
        if not base:
            return None
        return str(Path(base) / stage_name)

    def run(
        self,
        stages: Optional[Sequence[str]] = None,
        context: Optional[PipelineContext] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> Tuple[PipelineContext, List[StageResult]]:
        """
        Run the pipeline or a subset of stages.
        Stages receive and update the same context (paths and in-memory data).
        If *cancel_token* is provided it is stored on the context so stages
        can check it, and the loop checks it between stages.
        """
        context = context or PipelineContext()
        if cancel_token is not None:
            context.cancel_token = cancel_token
        run_order = stages or self.config.run_stages or list(self._stages.keys())
        # Preserve logical order
        ordered = [s for s in STAGE_NAMES if s in run_order and s in self._stages]
        results: List[StageResult] = []
        logger.info(
            "Pipeline starting: stages=%s, log_level=%s",
            " → ".join(ordered), self.config.log_level,
        )
        for idx, name in enumerate(ordered, 1):
            # Check cancellation before each stage
            if context.cancel_token and context.cancel_token.is_cancelled:
                logger.info("Pipeline cancelled before stage: %s", name)
                results.append(StageResult(success=False, error="Cancelled by user"))
                break
            stage = self._stages[name]
            logger.info("Stage %d/%d starting: %s", idx, len(ordered), name)
            try:
                result = stage.run(context)
            except PipelineCancelled:
                logger.warning("Pipeline cancelled during stage: %s", name)
                results.append(StageResult(success=False, error="Cancelled by user"))
                break
            results.append(result)
            if result.success:
                logger.info("Stage %d/%d completed: %s", idx, len(ordered), name)
            else:
                logger.error("Stage %d/%d failed: %s — %s", idx, len(ordered), name, result.error)
                break
        completed = sum(1 for r in results if r.success)
        logger.info("Pipeline finished: %d/%d stages completed", completed, len(ordered))
        # Clean up file handler
        if self._file_handler:
            logging.getLogger("autodistil_kg").removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        return context, results

    def run_stage(
        self,
        stage_name: str,
        context: Optional[PipelineContext] = None,
    ) -> StageResult:
        """Run a single stage by name. Use for standalone execution (e.g. only graph traverser)."""
        if stage_name not in self._stages:
            raise ValueError(f"Unknown or unconfigured stage: {stage_name}. Available: {list(self._stages.keys())}")
        context = context or PipelineContext()
        return self._stages[stage_name].run(context)

    @property
    def available_stages(self) -> List[str]:
        """Return list of configured stage names."""
        return list(self._stages.keys())
