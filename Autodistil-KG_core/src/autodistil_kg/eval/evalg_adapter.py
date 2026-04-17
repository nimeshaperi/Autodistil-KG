"""
EvalG adapter and interfaces.

This module intentionally keeps the integration with EvalG light-weight and
config-driven so the concrete EvalG library or CLI can be wired in without
changing the rest of the pipeline.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


class EvalSystemConfig(BaseModel):
    """
    Description of a system to be evaluated by EvalG.

    Examples:
    - kind = "distilled", model_path points to the fine-tuned model
    - kind = "base", provider/model identify a base LLM
    - kind = "external", provider/model point to OpenAI / Anthropic, etc.
    - kind = "graph_rag", rag_config points to a graph-RAG pipeline config
    """

    id: str
    label: Optional[str] = None
    kind: Literal["distilled", "base", "local_base", "external", "graph_rag"] = "distilled"

    model_path: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    rag_config: Optional[Dict[str, Any]] = None

    predictions_path: Optional[str] = None

    extra: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class EvalReport(BaseModel):
    """Typed evaluation report structure."""
    evalg_mode: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    timestamp: Optional[str] = None
    num_samples: Optional[int] = None
    metrics_used: Optional[List[str]] = None
    systems: Optional[Dict[str, Any]] = None
    per_question: Optional[List[Dict[str, Any]]] = None


def run_evalg(
    eval_dataset_path: str,
    systems: List[EvalSystemConfig],
    output_report_path: str,
    evalg_mode: str = "cli",
    evalg_command: Optional[List[str]] = None,
    evalg_extra_args: Optional[Dict[str, Any]] = None,
    cancel_token: Any = None,
    pre_system_callback: Any = None,
) -> Dict[str, Any]:
    """
    Run EvalG given an evaluation dataset and a list of systems.

    Parameters
    ----------
    eval_dataset_path:
        Path to the evaluation dataset JSONL.
    systems:
        Systems to compare (distilled/base/external/graph_rag).
    output_report_path:
        Where the EvalG JSON report should be written.
    evalg_mode:
        How to invoke EvalG: "internal", "cli", or "noop".
    evalg_command:
        When evalg_mode="cli", the base command to execute.
    evalg_extra_args:
        Optional extra configuration forwarded to EvalG.

    Returns
    -------
    Parsed JSON report (dict) that was written to output_report_path.
    """

    eval_dataset_path = str(Path(eval_dataset_path).resolve())
    output_report_path = str(Path(output_report_path).resolve())
    Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)

    systems_payload = [s.to_dict() for s in systems]
    evalg_extra_args = evalg_extra_args or {}

    if evalg_mode == "internal":
        from .internal_evaluator import InternalEvaluator

        evaluator = InternalEvaluator(
            eval_dataset_path=eval_dataset_path,
            systems=systems,
            output_report_path=output_report_path,
            metrics=evalg_extra_args.get("metrics", ["rouge"]),
            judge_config=evalg_extra_args.get("judge_config"),
            max_samples=evalg_extra_args.get("max_samples"),
            cancel_token=cancel_token,
            pre_system_callback=pre_system_callback,
        )
        return evaluator.run()

    if evalg_mode == "noop":
        logger.warning("EvalG mode is 'noop' – emitting stub metrics only.")
        report: Dict[str, Any] = {
            "evalg_mode": "noop",
            "eval_dataset_path": eval_dataset_path,
            "systems": systems_payload,
            "metrics": {},
        }
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    if evalg_mode == "cli":
        if not evalg_command:
            logger.warning(
                "EvalG mode 'cli' requested but no evalg_command provided; "
                "falling back to stub report."
            )
            return run_evalg(
                eval_dataset_path=eval_dataset_path,
                systems=systems,
                output_report_path=output_report_path,
                evalg_mode="noop",
                evalg_extra_args=evalg_extra_args,
            )

        payload = {
            "eval_dataset_path": eval_dataset_path,
            "systems": systems_payload,
            "output_report_path": output_report_path,
            "extra": evalg_extra_args,
        }

        try:
            proc = subprocess.run(
                evalg_command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception as e:
            logger.exception("Failed to invoke EvalG CLI: %s", e)
            return run_evalg(
                eval_dataset_path=eval_dataset_path,
                systems=systems,
                output_report_path=output_report_path,
                evalg_mode="noop",
                evalg_extra_args=evalg_extra_args,
            )

        if proc.returncode != 0:
            logger.error("EvalG CLI exited with code %s: %s", proc.returncode, proc.stderr)
            return run_evalg(
                eval_dataset_path=eval_dataset_path,
                systems=systems,
                output_report_path=output_report_path,
                evalg_mode="noop",
                evalg_extra_args=evalg_extra_args,
            )

        try:
            with open(output_report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("EvalG CLI completed but report could not be read: %s", e)
            return run_evalg(
                eval_dataset_path=eval_dataset_path,
                systems=systems,
                output_report_path=output_report_path,
                evalg_mode="noop",
                evalg_extra_args=evalg_extra_args,
            )

    logger.warning("Unknown EvalG mode '%s'; using stub report.", evalg_mode)
    return run_evalg(
        eval_dataset_path=eval_dataset_path,
        systems=systems,
        output_report_path=output_report_path,
        evalg_mode="noop",
        evalg_extra_args=evalg_extra_args,
    )
