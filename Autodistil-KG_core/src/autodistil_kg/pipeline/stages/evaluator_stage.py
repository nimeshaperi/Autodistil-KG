"""
Evaluator stage: integrates EvalG-based evaluation over one or more systems.

This stage is intended to:
- Take the fine-tuned model (and optionally other systems: base models, external
  APIs, graph-RAG pipelines) and an evaluation dataset.
- Delegate metric computation to EvalG via the evalg_adapter.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..interfaces import Stage, StageResult, PipelineContext
from ..config import EvaluatorStageConfig
from ...eval.evalg_adapter import EvalSystemConfig, run_evalg

logger = logging.getLogger(__name__)


class EvaluatorStage(Stage):
    """Stage that evaluates one or more systems using EvalG."""

    name = "evaluator"  # matches StageId.EVALUATOR

    def __init__(self, config: EvaluatorStageConfig):
        self.config = config

    def _get_base_model_name(self, context: PipelineContext) -> Optional[str]:
        """Resolve the base model name from config or finetuner stage context."""
        if self.config.base_model_name:
            return self.config.base_model_name
        # Fall back to finetuner's model name from pipeline context
        return context.finetuner_model_name

    @staticmethod
    def _merge_lora_adapter(base_model: str, lora_path: str) -> str:
        """Merge a LoRA adapter into the base model and save as a full model.

        Uses Unsloth to load the base model + LoRA adapter, merges the
        weights, and saves the result to ``<lora_path>_merged/``.  The
        merged model can be loaded directly by vLLM without LoRA support.

        Returns the path to the merged model directory.
        """
        import gc
        import torch
        from unsloth import FastLanguageModel

        merged_dir = str(Path(lora_path).parent / f"{Path(lora_path).name}_merged")
        if Path(merged_dir).exists() and any(Path(merged_dir).iterdir()):
            logger.info("Merged model already exists at %s — reusing", merged_dir)
            return merged_dir

        logger.info("Merging LoRA adapter into base model: base=%s, lora=%s", base_model, lora_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=4096,
        )
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        logger.info("Merged model saved to %s", merged_dir)

        # Free GPU memory before vLLM starts
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return merged_dir

    def _build_systems(self, model_path: Optional[str]) -> List[EvalSystemConfig]:
        """
        Build the list of systems to pass into EvalG.

        Includes the distilled (finetuned) model when model_path is available.
        Auto-creates base model and Graph RAG systems from first-class config
        fields when present. Additional systems can still be supplied via
        ``config.additional_params["systems"]``.
        """
        systems: List[EvalSystemConfig] = []

        # 1. Include the distilled (finetuned) model if available and not explicitly disabled.
        if model_path and self.config.eval_distilled is not False:
            systems.append(
                EvalSystemConfig(
                    id="distilled",
                    label="Finetuned model",
                    kind="distilled",
                    model_path=model_path,
                    extra={
                        "max_new_tokens": self.config.max_new_tokens,
                        "max_seq_length": self.config.max_seq_length,
                    },
                )
            )

        # 2. Auto-create base model system from config fields.
        if self.config.base_model_provider and self.config.eval_base is not False:
            if self.config.base_model_provider == "local":
                # Load base model locally via Unsloth (no LoRA).
                # Runs sequentially after the finetuned model to share VRAM.
                systems.append(
                    EvalSystemConfig(
                        id="base",
                        label=f"Base model ({self.config.base_model_name or 'unknown'})",
                        kind="local_base",
                        model_path=self.config.base_model_name,
                        extra={
                            "max_new_tokens": self.config.max_new_tokens,
                            "max_seq_length": self.config.max_seq_length,
                        },
                    )
                )
            else:
                systems.append(
                    EvalSystemConfig(
                        id="base",
                        label=f"Base model ({self.config.base_model_name or 'default'} via {self.config.base_model_provider})",
                        kind="base",
                        provider=self.config.base_model_provider,
                        model=self.config.base_model_name,
                        extra={
                            "api_key": self.config.base_model_api_key,
                            "base_url": self.config.base_model_base_url,
                        },
                    )
                )

        # 3. Auto-create Graph RAG system from config fields.
        if self.config.graph_rag_config and self.config.eval_graph_rag is not False:
            systems.append(
                EvalSystemConfig(
                    id="graph_rag",
                    label="Graph RAG",
                    kind="graph_rag",
                    rag_config=self.config.graph_rag_config,
                )
            )

        # 4. Additional systems from additional_params (backward-compatible).
        extra = self.config.additional_params or {}
        raw_systems = extra.get("systems", [])
        if isinstance(raw_systems, list):
            for idx, s in enumerate(raw_systems):
                if not isinstance(s, dict):
                    logger.warning("Ignoring non-dict system config at index %s", idx)
                    continue
                try:
                    systems.append(
                        EvalSystemConfig(
                            id=str(s.get("id") or f"system_{idx}"),
                            label=s.get("label"),
                            kind=s.get("kind", "external"),
                            model_path=s.get("model_path"),
                            provider=s.get("provider"),
                            model=s.get("model"),
                            rag_config=s.get("rag_config"),
                            predictions_path=s.get("predictions_path"),
                            extra=s.get("extra") or {},
                        )
                    )
                except Exception as e:
                    logger.warning("Failed to parse system config at index %s: %s", idx, e)
        elif raw_systems:
            logger.warning("EvaluatorStage.additional_params.systems is not a list; ignoring.")

        return systems

    def run(self, context: PipelineContext) -> StageResult:
        model_path = self.config.model_path or context.model_output_path

        # Resolve eval dataset: try each candidate in order, pick the first
        # that actually exists on disk.
        eval_candidates = [
            self.config.eval_dataset_path,
            context.eval_dataset_path,
            context.prepared_dataset_path,
            context.chatml_dataset_path,
        ]
        eval_path = None
        for candidate in eval_candidates:
            if candidate and Path(candidate).exists():
                eval_path = candidate
                break

        if not eval_path:
            tried = [c for c in eval_candidates if c]
            return StageResult(
                success=False,
                error=f"Evaluator requires an eval dataset. Tried: {tried}",
            )
        has_any_system = (
            model_path
            or self.config.graph_rag_config
            or self.config.base_model_provider
        )
        if not has_any_system:
            return StageResult(
                success=False,
                error="Evaluator requires at least one of: model_path, base_model_provider, or graph_rag_config.",
            )
        if model_path and not Path(model_path).exists():
            return StageResult(success=False, error=f"Model path does not exist: {model_path}")

        output_report = self.config.output_report_path or context.eval_report_path
        if not output_report:
            if model_path:
                base_dir = Path(model_path)
                if base_dir.is_file():
                    base_dir = base_dir.parent
                output_report = str(base_dir / "eval_report.json")
            else:
                # No model path — default to eval dataset directory
                output_report = str(Path(eval_path).parent / "eval_report.json")

        # ── vLLM server lifecycle ──────────────────────────────────────
        vllm_server = None
        vllm_sequential = False  # True when merged model requires sequential serving
        pre_system_callback = None
        base_model: Optional[str] = None
        merged_path: Optional[str] = None

        if self.config.use_vllm:
            from ...eval.vllm_server import VLLMServerManager

            # Determine the base model name
            base_model = self._get_base_model_name(context)
            if not base_model:
                # Try to read from the LoRA adapter_config.json
                adapter_cfg_path = Path(model_path) / "adapter_config.json" if model_path else None
                if adapter_cfg_path and adapter_cfg_path.exists():
                    import json as _json
                    with open(adapter_cfg_path) as f:
                        adapter_data = _json.load(f)
                    base_model = adapter_data.get("base_model_name_or_path")
                if not base_model:
                    return StageResult(
                        success=False,
                        error="vLLM requires a base model name. Set base_model_name in the evaluator config or finetuner model_name.",
                    )

            # Only pass LoRA path if it's a valid adapter directory
            lora_path = None
            if model_path and Path(model_path).exists():
                adapter_cfg = Path(model_path) / "adapter_config.json"
                if adapter_cfg.exists():
                    lora_path = model_path
                    logger.info("LoRA adapter found at %s", model_path)
                else:
                    logger.info("No adapter_config.json in %s — starting vLLM without LoRA", model_path)

            vllm_server = VLLMServerManager(
                model_name=base_model,
                lora_path=lora_path,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
            )
            logger.info(
                "vllm starting: %s",
                base_model,
                extra={"eval_event": {"type": "vllm_starting", "model": base_model}},
            )
            try:
                vllm_url = vllm_server.start()
                logger.info(
                    "vllm ready: %s at %s",
                    base_model,
                    vllm_url,
                    extra={"eval_event": {"type": "vllm_ready", "model": base_model, "url": vllm_url}},
                )
            except Exception as e:
                if lora_path:
                    logger.warning(
                        "vLLM failed with LoRA adapter — merging LoRA into base model "
                        "and using sequential vLLM serving: %s", e
                    )
                    vllm_server.stop()
                    vllm_server = None
                    try:
                        merged_path = self._merge_lora_adapter(base_model, lora_path)
                    except Exception as merge_err:
                        logger.exception("Failed to merge LoRA adapter")
                        return StageResult(
                            success=False,
                            error=f"vLLM LoRA loading failed and merge fallback also failed: {merge_err}",
                        )
                    vllm_sequential = True
                else:
                    logger.exception("Failed to start vLLM server")
                    return StageResult(success=False, error=f"vLLM server failed to start: {e}")

        systems = self._build_systems(model_path)

        if self.config.use_vllm and not vllm_sequential:
            # ── Normal mode: single vLLM server with LoRA or base model ──
            if vllm_server:
                for sys_cfg in systems:
                    if sys_cfg.kind in ("distilled", "local_base"):
                        sys_cfg.extra["vllm_base_url"] = vllm_server.base_url
                        if sys_cfg.kind == "distilled" and vllm_server.lora_path:
                            sys_cfg.extra["vllm_model_name"] = vllm_server.lora_model_name
                        else:
                            sys_cfg.extra["vllm_model_name"] = vllm_server.model_name
                    elif sys_cfg.kind == "graph_rag" and sys_cfg.rag_config:
                        sys_cfg.rag_config["llm_base_url"] = vllm_server.base_url
                        sys_cfg.rag_config["llm_model"] = vllm_server.model_name
                        sys_cfg.rag_config["llm_api_key"] = "not-needed"
                        sys_cfg.rag_config["api_key"] = "not-needed"
                        sys_cfg.rag_config["base_url"] = vllm_server.base_url
                        sys_cfg.rag_config["model"] = vllm_server.model_name
                        logger.info(
                            "Graph RAG LLM overridden to vLLM: %s model=%s",
                            vllm_server.base_url, vllm_server.model_name,
                        )

        elif self.config.use_vllm and vllm_sequential:
            # ── Sequential mode: restart vLLM with the right model per system ──
            # Map each system kind to the model it should use
            _model_for_kind: Dict[str, str] = {}
            if merged_path:
                _model_for_kind["distilled"] = merged_path
            _model_for_kind["local_base"] = base_model
            _model_for_kind["graph_rag"] = base_model

            # Pre-populate system configs with vLLM URL (port stays the same)
            _port = self.config.vllm_port if hasattr(self.config, "vllm_port") else 8432
            _vllm_base_url = f"http://localhost:{_port}/v1"
            for sys_cfg in systems:
                if sys_cfg.kind in ("distilled", "local_base"):
                    sys_cfg.extra["vllm_base_url"] = _vllm_base_url
                    sys_cfg.extra["vllm_model_name"] = _model_for_kind.get(sys_cfg.kind, base_model)
                elif sys_cfg.kind == "graph_rag" and sys_cfg.rag_config:
                    sys_cfg.rag_config["llm_base_url"] = _vllm_base_url
                    sys_cfg.rag_config["llm_model"] = _model_for_kind.get("graph_rag", base_model)
                    sys_cfg.rag_config["llm_api_key"] = "not-needed"
                    sys_cfg.rag_config["api_key"] = "not-needed"
                    sys_cfg.rag_config["base_url"] = _vllm_base_url
                    sys_cfg.rag_config["model"] = _model_for_kind.get("graph_rag", base_model)

            # State for the sequential vLLM manager
            _current_model: List[Optional[str]] = [None]  # mutable for closure
            _server: List[Optional[Any]] = [None]

            def _sequential_vllm_callback(sys_cfg) -> None:
                """Restart vLLM if the system needs a different model."""
                needed_model = _model_for_kind.get(sys_cfg.kind)
                if not needed_model:
                    return  # System doesn't use vLLM

                if _current_model[0] == needed_model and _server[0] is not None:
                    # Server already running with the right model — emit vllm_ready
                    # to clear any leftover loading banner in the UI (e.g. from the
                    # initial vllm_starting that was emitted before the LoRA fallback).
                    logger.info(
                        "vLLM already serving %s for system %r — reusing",
                        needed_model, sys_cfg.id,
                        extra={"eval_event": {"type": "vllm_ready", "model": needed_model}},
                    )
                    return

                # Stop the current server if running
                if _server[0] is not None:
                    logger.info("Stopping vLLM server (was serving %s)", _current_model[0])
                    _server[0].stop()
                    _server[0] = None
                    _current_model[0] = None

                logger.info(
                    "Starting vLLM with model %s for system %r",
                    needed_model, sys_cfg.id,
                    extra={"eval_event": {"type": "vllm_starting", "model": needed_model}},
                )
                srv = VLLMServerManager(
                    model_name=needed_model,
                    lora_path=None,
                    gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                    max_model_len=self.config.vllm_max_model_len,
                )
                srv.start()
                _server[0] = srv
                _current_model[0] = needed_model
                logger.info(
                    "vLLM ready: %s",
                    needed_model,
                    extra={"eval_event": {"type": "vllm_ready", "model": needed_model}},
                )

            pre_system_callback = _sequential_vllm_callback
            # Expose the server list so the finally block can stop it
            self._sequential_server = _server

        extra: Dict[str, Any] = self.config.additional_params or {}

        evalg_mode = self.config.evalg_mode or extra.get("evalg_mode", "internal")

        evalg_command = extra.get("evalg_command")
        if isinstance(evalg_command, list):
            evalg_cmd_list = [str(x) for x in evalg_command]
        elif isinstance(evalg_command, str):
            evalg_cmd_list = [evalg_command]
        else:
            evalg_cmd_list = None

        # Build evalg_extra_args from first-class config fields.
        evalg_extra_args: Dict[str, Any] = extra.get("evalg_extra_args") or {}
        if self.config.metrics:
            evalg_extra_args.setdefault("metrics", self.config.metrics)
        if self.config.max_eval_samples:
            evalg_extra_args.setdefault("max_samples", self.config.max_eval_samples)
        if self.config.judge_provider:
            evalg_extra_args.setdefault("judge_config", {
                "provider": self.config.judge_provider,
                "model": self.config.judge_model,
                "api_key": self.config.judge_api_key,
                "base_url": self.config.judge_base_url,
                "max_tokens": self.config.judge_max_tokens,
            })

        logger.info(
            "Running evaluator with EvalG (mode=%s) on dataset=%s, model=%s, systems=%s%s",
            evalg_mode,
            eval_path,
            model_path,
            [s.id for s in systems],
            " [sequential vLLM]" if vllm_sequential else "",
        )

        try:
            report = run_evalg(
                eval_dataset_path=str(eval_path),
                systems=systems,
                output_report_path=str(output_report),
                evalg_mode=str(evalg_mode),
                evalg_command=evalg_cmd_list,
                evalg_extra_args=evalg_extra_args,
                cancel_token=context.cancel_token,
                pre_system_callback=pre_system_callback,
            )
        except Exception as e:
            logger.exception("EvalG evaluation failed")
            return StageResult(success=False, error=str(e))
        finally:
            if vllm_server:
                vllm_server.stop()
            # Clean up sequential vLLM server if active
            seq_srv = getattr(self, "_sequential_server", None)
            if seq_srv and seq_srv[0] is not None:
                seq_srv[0].stop()
                seq_srv[0] = None

        context.eval_report_path = str(output_report)

        metadata: Dict[str, Any] = {
            "model_path": str(model_path),
            "eval_dataset_path": str(eval_path),
            "eval_report_path": str(output_report),
        }
        if isinstance(report, dict):
            metadata["metrics"] = report.get("metrics", {})
            metadata["raw_report"] = report

        return StageResult(success=True, output=report, metadata=metadata)

