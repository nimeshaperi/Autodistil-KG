"""
FineTuner stage: runs Unsloth (or other) finetuner on prepared CHATML dataset.
Runnable standalone (provide train_data_path) or after chatml_converter (use context).
"""
import logging
from pathlib import Path
from typing import Optional

from ..interfaces import Stage, StageResult, PipelineContext
from ..config import FineTunerStageConfig
from ...finetuner import UnslothFineTuner, UnslothFineTunerConfig
from ...finetuner.config import ModelType

logger = logging.getLogger(__name__)


class FineTunerStage(Stage):
    """Stage that runs the finetuner on a prepared dataset."""

    name = "finetuner"  # matches StageId.FINETUNER

    def __init__(self, config: FineTunerStageConfig):
        self.config = config

    def _build_finetuner_config(self, context: PipelineContext) -> UnslothFineTunerConfig:
        cfg = self.config
        train_path = (
            cfg.train_data_path
            or context.prepared_dataset_path
            or context.chatml_dataset_path
        )
        if not train_path:
            raise ValueError("FineTunerStage: no train_data_path and none in context")
        if not Path(train_path).exists():
            raise FileNotFoundError(f"Train data path does not exist: {train_path}")

        if cfg.finetuner_config is not None:
            # Use provided config but override paths if needed
            finetuner_cfg = cfg.finetuner_config
            if not getattr(finetuner_cfg, "train_data_path", None):
                finetuner_cfg.train_data_path = train_path
            if cfg.output_dir and not getattr(finetuner_cfg, "output_dir", None):
                finetuner_cfg.output_dir = cfg.output_dir
            return finetuner_cfg

        model_type = None
        if cfg.model_type is not None:
            model_type = ModelType(cfg.model_type) if isinstance(cfg.model_type, str) else cfg.model_type
        return UnslothFineTunerConfig(
            model_name=cfg.model_name or "unsloth/gemma-2-2b-it",
            model_type=model_type,
            max_seq_length=cfg.max_seq_length,
            train_data_path=train_path,
            eval_data_path=cfg.eval_data_path,
            output_dir=cfg.output_dir or "./finetuner_output",
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            learning_rate=cfg.learning_rate,
            **cfg.additional_params,
        )

    def run(self, context: PipelineContext) -> StageResult:
        try:
            finetuner_cfg = self._build_finetuner_config(context)
            finetuner = UnslothFineTuner(finetuner_cfg)
            output_dir = finetuner.train(
                train_path=finetuner_cfg.train_data_path,
                eval_path=finetuner_cfg.eval_data_path,
                output_dir=finetuner_cfg.output_dir,
                cancel_token=context.cancel_token,
            )
            context.model_output_path = output_dir
            # Store base model name for downstream stages (e.g. vLLM serving)
            context.finetuner_model_name = finetuner_cfg.model_name
            return StageResult(
                success=True,
                output=output_dir,
                metadata={"output_dir": output_dir},
            )
        except Exception as e:
            logger.exception("Finetuner stage failed")
            return StageResult(success=False, error=str(e))
