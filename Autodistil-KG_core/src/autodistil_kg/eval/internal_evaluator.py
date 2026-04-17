"""
Internal evaluator: orchestrates prediction generation and metric scoring
for comparing base model, finetuned model, and Graph RAG systems.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evalg_adapter import EvalSystemConfig
from .predictors import Predictor, PredictionResult, build_predictor
from .scorers import Scorer, build_scorers

logger = logging.getLogger(__name__)


def _emit_eval_event(event_type: str, **data) -> None:
    """Emit a structured evaluation event via logging extras.

    The API's QueueLogHandler picks up the ``eval_event`` extra and
    forwards it to the WebSocket so the UI can display real-time
    evaluation progress.
    """
    logger.info(
        "eval_event: %s %s",
        event_type,
        json.dumps({k: v for k, v in data.items() if v is not None}, default=str),
        extra={"eval_event": {"type": event_type, **data}},
    )


def _strip_reasoning_traces(text: str) -> str:
    """Remove LLM reasoning/thinking traces from text for clean evaluation.

    Reasoning models (Gemini 2.5, Qwen3, DeepSeek-R1, etc.) emit
    chain-of-thought blocks that must be stripped so the judge LLM
    evaluates actual content, not meta-reasoning noise.
    """
    # 1. Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Strip orphaned </think> — everything before it is reasoning
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # 3. Strip "Thinking Process:" blocks up to known content markers
    text = re.sub(
        r"^[\s]*Thinking(?:\s+Process)?:\s*\n.*?"
        r"(?=\*\*Question:\*\*|\*\*Answer:\*\*|\n##\s)",
        "",
        text,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Aggressive fallback if no content markers found
    if re.match(r"^\s*Thinking(?:\s+Process)?:", text, re.IGNORECASE):
        text = re.sub(
            r"^[\s]*Thinking(?:\s+Process)?:\s*\n(?:.*\n)*?(?=\S)",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    # 4. Strip numbered meta-reasoning blocks before **Question:**/**Answer:**
    text = re.sub(
        r"^\s*\d+\.\s+\*\*.*?(?=\*\*Question:\*\*|\*\*Answer:\*\*)",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # 5. Strip chat-template markers
    text = re.sub(r"<\|im_start\|>.*?(?:\n|$)", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<start_of_turn>.*?(?:\n|$)", "", text)
    text = re.sub(r"<end_of_turn>", "", text)
    return text.strip()


def _extract_eval_sample(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract system_prompt, question, and reference from a CHATML record.

    Expected format:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    The system message is optional.  Reasoning traces are stripped from
    the question and reference so the judge LLM evaluates clean content.
    """
    messages = record.get("messages")
    if not messages or not isinstance(messages, list):
        return None

    system_prompt = ""
    question = ""
    reference = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            question = content
        elif role == "assistant":
            reference = content

    # Strip reasoning traces so the judge evaluates clean Q&A
    question = _strip_reasoning_traces(question)
    reference = _strip_reasoning_traces(reference)

    if not question:
        return None

    return {
        "system_prompt": system_prompt,
        "question": question,
        "reference": reference,
    }


class InternalEvaluator:
    """Runs in-process evaluation across multiple systems.

    1. Loads the eval dataset (JSONL)
    2. Builds a predictor per system
    3. Generates predictions for every (system, sample) pair
    4. Scores predictions with configured metrics
    5. Writes a structured JSON comparison report

    Supports checkpoint/resume: predictions and scores are written
    incrementally to a ``.checkpoint.jsonl`` file beside the output
    report.  On restart, completed work is loaded from the checkpoint
    so evaluation resumes where it left off.
    """

    def __init__(
        self,
        eval_dataset_path: str,
        systems: List[EvalSystemConfig],
        output_report_path: str,
        metrics: Optional[List[str]] = None,
        judge_config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        cancel_token: Optional[Any] = None,
        pre_system_callback: Optional[Any] = None,
    ) -> None:
        self.eval_dataset_path = eval_dataset_path
        self.systems = systems
        self.output_report_path = output_report_path
        self.metrics = metrics or ["answer_relevancy"]
        self.judge_config = judge_config
        self.max_samples = max_samples
        self.cancel_token = cancel_token
        self._pre_system_callback = pre_system_callback

        # Checkpoint file lives beside the output report
        report_path = Path(output_report_path)
        self._checkpoint_path = report_path.parent / f"{report_path.stem}.checkpoint.jsonl"

    # ── Checkpoint helpers ────────────────────────────────────────────

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load existing checkpoint data.

        Returns a dict with:
          predictions: {(system_id, sample_index): prediction_text}
          perf: {(system_id, sample_index): perf_dict}
          scores: {(system_id, sample_index): scores_dict}
        """
        checkpoint: Dict[str, Any] = {
            "predictions": {},
            "perf": {},
            "scores": {},
        }
        if not self._checkpoint_path.exists():
            return checkpoint

        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = entry.get("kind")
                    sys_id = entry.get("system_id")
                    idx = entry.get("sample_index")
                    if kind == "prediction" and sys_id is not None and idx is not None:
                        checkpoint["predictions"][(sys_id, idx)] = entry.get("prediction", "")
                        checkpoint["perf"][(sys_id, idx)] = entry.get("perf", {})
                    elif kind == "score" and sys_id is not None and idx is not None:
                        checkpoint["scores"][(sys_id, idx)] = entry.get("scores", {})
        except Exception as e:
            logger.warning("Failed to load checkpoint %s: %s — starting fresh", self._checkpoint_path, e)
            checkpoint = {"predictions": {}, "perf": {}, "scores": {}}

        num_preds = len(checkpoint["predictions"])
        num_scores = len(checkpoint["scores"])
        if num_preds > 0 or num_scores > 0:
            logger.info(
                "Loaded checkpoint: %d predictions, %d scores from %s",
                num_preds, num_scores, self._checkpoint_path,
            )
        return checkpoint

    def _append_checkpoint(self, entry: Dict[str, Any]) -> None:
        """Append a single checkpoint entry to the JSONL file."""
        try:
            with open(self._checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write checkpoint entry: %s", e)

    def _remove_checkpoint(self) -> None:
        """Remove the checkpoint file after a successful run."""
        try:
            if self._checkpoint_path.exists():
                self._checkpoint_path.unlink()
                logger.info("Removed checkpoint file %s", self._checkpoint_path)
        except Exception as e:
            logger.warning("Failed to remove checkpoint file: %s", e)

    def _load_dataset(self) -> List[Dict[str, str]]:
        """Load and parse the eval JSONL file."""
        samples: List[Dict[str, str]] = []
        path = Path(self.eval_dataset_path)
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", line_no)
                    continue

                sample = _extract_eval_sample(record)
                if sample is None:
                    logger.warning("Skipping record at line %d: missing question", line_no)
                    continue
                samples.append(sample)

                if self.max_samples and len(samples) >= self.max_samples:
                    break

        logger.info("Loaded %d evaluation samples from %s", len(samples), path)
        return samples

    def _generate_predictions(
        self,
        samples: List[Dict[str, str]],
        checkpoint: Dict[str, Any],
    ) -> tuple[List[str], List[Dict[str, str]], List[Dict[str, Dict[str, Any]]]]:
        """Generate predictions for every (system, sample) pair.

        Systems are run sequentially: all samples for one system complete
        before the next system starts.  This allows GPU-intensive systems
        (finetuned model, local base model) to release VRAM before the
        next system loads.

        Already-completed predictions from the checkpoint are restored
        without re-running the predictor.

        Returns:
            active_system_ids: list of system ids that successfully produced predictions.
            all_predictions: list (one per sample) of dicts mapping system_id -> prediction text.
            all_perf: list (one per sample) of dicts mapping system_id -> perf metrics dict.
        """
        n = len(samples)
        all_predictions: List[Dict[str, str]] = [{} for _ in range(n)]
        all_perf: List[Dict[str, Dict[str, Any]]] = [{} for _ in range(n)]
        active_system_ids: List[str] = []

        cached_preds = checkpoint.get("predictions", {})
        cached_perf = checkpoint.get("perf", {})

        total_systems = len(self.systems)
        for sys_idx, sys_cfg in enumerate(self.systems):
            if self.cancel_token and self.cancel_token.is_cancelled:
                logger.info("Cancellation requested — skipping remaining systems")
                break
            sys_id = sys_cfg.id

            # Check how many samples are already cached for this system
            cached_for_system = sum(1 for i in range(n) if (sys_id, i) in cached_preds)
            all_cached = cached_for_system == n

            if all_cached:
                # Restore all predictions from checkpoint — no need to build predictor
                logger.info(
                    "System %r: all %d predictions restored from checkpoint", sys_id, n
                )
                for idx in range(n):
                    all_predictions[idx][sys_id] = cached_preds[(sys_id, idx)]
                    all_perf[idx][sys_id] = cached_perf.get((sys_id, idx), {})
                active_system_ids.append(sys_id)
                _emit_eval_event(
                    "system_done",
                    system_id=sys_id,
                    total_predictions=n,
                    resumed=True,
                )
                continue

            _emit_eval_event(
                "system_start",
                system_id=sys_id,
                system_label=sys_cfg.label,
                kind=sys_cfg.kind,
                system_index=sys_idx,
                total_systems=total_systems,
            )
            # Allow the caller to prepare resources (e.g. restart vLLM
            # with a different model) before building the predictor.
            if self._pre_system_callback and not all_cached:
                try:
                    self._pre_system_callback(sys_cfg)
                except Exception as cb_err:
                    logger.error(
                        "pre_system_callback failed for system %r: %s — skipping",
                        sys_id, cb_err,
                    )
                    _emit_eval_event("system_error", system_id=sys_id, error=str(cb_err))
                    continue

            try:
                predictor = build_predictor(sys_cfg)
                logger.info("Built predictor for system %r (kind=%s)", sys_id, sys_cfg.kind)
            except Exception as e:
                logger.error(
                    "Failed to build predictor for system %r: %s — skipping", sys_id, e
                )
                _emit_eval_event(
                    "system_error",
                    system_id=sys_id,
                    error=str(e),
                )
                continue

            active_system_ids.append(sys_id)

            resumed_count = cached_for_system
            if resumed_count > 0:
                logger.info(
                    "System %r: resuming — %d/%d predictions already cached",
                    sys_id, resumed_count, n,
                )

            _emit_eval_event(
                "predictions_start",
                system_id=sys_id,
                total_samples=n,
                resumed_count=resumed_count,
            )

            for idx, sample in enumerate(samples):
                # Check for cancellation between predictions
                if self.cancel_token and self.cancel_token.is_cancelled:
                    logger.info("Cancellation requested — stopping predictions for system %r", sys_id)
                    break

                # Skip if already in checkpoint
                if (sys_id, idx) in cached_preds:
                    all_predictions[idx][sys_id] = cached_preds[(sys_id, idx)]
                    all_perf[idx][sys_id] = cached_perf.get((sys_id, idx), {})
                    continue

                perf_data: Dict[str, Any] = {}
                sample_error: Optional[str] = None
                try:
                    result = predictor.predict(sample["system_prompt"], sample["question"])
                    all_predictions[idx][sys_id] = result.text
                    perf_data = {
                        "latency_sec": round(result.latency_sec, 4),
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "total_tokens": result.total_tokens,
                        "tokens_per_sec": round(result.output_tokens / result.latency_sec, 1)
                        if result.latency_sec > 0 else 0.0,
                    }
                    all_perf[idx][sys_id] = perf_data
                except Exception as e:
                    sample_error = str(e)
                    logger.error(
                        "Prediction failed for system %r on sample %d: %s", sys_id, idx, e
                    )
                    all_predictions[idx][sys_id] = f"[ERROR: {e}]"
                    perf_data = {"latency_sec": 0, "input_tokens": 0, "output_tokens": 0,
                                 "total_tokens": 0, "tokens_per_sec": 0}
                    all_perf[idx][sys_id] = perf_data

                # Write to checkpoint file immediately
                self._append_checkpoint({
                    "kind": "prediction",
                    "system_id": sys_id,
                    "sample_index": idx,
                    "prediction": all_predictions[idx][sys_id],
                    "perf": perf_data,
                })

                _emit_eval_event(
                    "prediction_done",
                    system_id=sys_id,
                    sample_index=idx,
                    total_samples=n,
                    latency_sec=perf_data.get("latency_sec"),
                    tokens_per_sec=perf_data.get("tokens_per_sec"),
                    error=sample_error,
                )

            # Release resources (frees GPU VRAM for local models)
            predictor.close()
            _emit_eval_event(
                "system_done",
                system_id=sys_id,
                total_predictions=n,
            )

        return active_system_ids, all_predictions, all_perf

    def _score_predictions(
        self,
        samples: List[Dict[str, str]],
        all_predictions: List[Dict[str, str]],
        scorers: List[Scorer],
        checkpoint: Dict[str, Any],
    ) -> List[Dict[str, Dict[str, Any]]]:
        """Score every (system, sample) prediction.

        Already-scored entries from the checkpoint are restored without
        re-running the scorers.

        Returns a list (one per sample) of dicts mapping system_id -> merged scores.
        Scores may contain ``None`` values for metrics that failed evaluation.
        """
        cached_scores = checkpoint.get("scores", {})
        all_scores: List[Dict[str, Dict[str, Any]]] = []
        n = len(samples)
        _emit_eval_event(
            "scoring_start",
            total_samples=n,
            scorers=[s.name for s in scorers],
        )

        # Use a shared thread pool for all scoring.  Cap at 4 workers so we
        # don't blast rate-limited judge APIs (the semaphore in DeepEvalScorer
        # serialises to 2 concurrent LLM calls regardless).
        max_workers = min(len(scorers) * len(all_predictions[0]) if all_predictions else 4, 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx, (sample, preds) in enumerate(zip(samples, all_predictions)):
                # Check for cancellation between scoring samples
                if self.cancel_token and self.cancel_token.is_cancelled:
                    logger.info("Cancellation requested — stopping scoring at sample %d", idx)
                    break
                sample_scores: Dict[str, Dict[str, Any]] = {}

                # Collect all (system, scorer) pairs that need scoring
                futures_map: Dict[Any, tuple[str, Scorer]] = {}
                for sys_id, prediction in preds.items():
                    if (sys_id, idx) in cached_scores:
                        sample_scores[sys_id] = cached_scores[(sys_id, idx)]
                        continue
                    sample_scores[sys_id] = {}
                    for scorer in scorers:
                        fut = pool.submit(
                            scorer.score,
                            prediction=prediction,
                            reference=sample["reference"],
                            question=sample["question"],
                        )
                        futures_map[fut] = (sys_id, scorer)

                # Gather results
                for future in as_completed(futures_map):
                    sys_id, scorer = futures_map[future]
                    try:
                        result = future.result()
                        sample_scores[sys_id].update(result)
                    except Exception as e:
                        logger.error(
                            "Scorer %r failed for system %r on sample %d: %s",
                            scorer.name, sys_id, idx, e,
                        )

                # Write checkpoints for all systems scored in this sample
                for sys_id in preds:
                    if (sys_id, idx) not in cached_scores and sys_id in sample_scores:
                        self._append_checkpoint({
                            "kind": "score",
                            "system_id": sys_id,
                            "sample_index": idx,
                            "scores": sample_scores[sys_id],
                        })

                all_scores.append(sample_scores)

            _emit_eval_event(
                "scoring_done",
                sample_index=idx,
                total_samples=n,
                scores={
                    sid: {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in sc.items()}
                    for sid, sc in sample_scores.items()
                },
            )

        return all_scores

    @staticmethod
    def _aggregate_metrics(
        all_scores: List[Dict[str, Dict[str, Any]]],
        system_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-system average metrics.

        Scores of ``None`` (failed evaluations) are excluded from
        averages.  ``*_failed`` marker keys are counted separately and
        reported as ``<metric>_fail_count`` in the aggregate.
        """
        sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        fail_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for sample_scores in all_scores:
            for sys_id in system_ids:
                scores = sample_scores.get(sys_id, {})
                for metric_name, value in scores.items():
                    # Skip internal failure marker keys
                    if metric_name.endswith("_failed"):
                        if value:
                            base_metric = metric_name.removesuffix("_failed")
                            fail_counts[sys_id][base_metric] += 1
                        continue
                    # Skip None (failed) scores — don't count toward average
                    if value is None:
                        continue
                    sums[sys_id][metric_name] += value
                    counts[sys_id][metric_name] += 1

        aggregated: Dict[str, Dict[str, Any]] = {}
        for sys_id in system_ids:
            aggregated[sys_id] = {}
            for metric_name in sums[sys_id]:
                n = counts[sys_id][metric_name]
                aggregated[sys_id][metric_name] = round(sums[sys_id][metric_name] / n, 4) if n else None

            # Include failure counts so consumers can see how many evals failed
            for metric_name, fc in fail_counts[sys_id].items():
                if fc > 0:
                    aggregated[sys_id][f"{metric_name}_fail_count"] = fc

            # Compute judge_avg from correctness sub-scores if present
            judge_keys = [k for k in aggregated[sys_id] if k.startswith("judge_") and k != "judge_avg" and not k.endswith("_fail_count")]
            if judge_keys:
                vals = [aggregated[sys_id][k] for k in judge_keys if aggregated[sys_id][k] is not None and aggregated[sys_id][k] > 0]
                aggregated[sys_id]["judge_avg"] = round(sum(vals) / len(vals), 4) if vals else None

            # Compute grounding_avg from grounding sub-scores if present
            grounding_keys = [
                k for k in aggregated[sys_id]
                if k.endswith("_grounding") and k != "grounding_avg"
            ]
            if grounding_keys:
                vals = [aggregated[sys_id][k] for k in grounding_keys if aggregated[sys_id][k] is not None]
                aggregated[sys_id]["grounding_avg"] = round(sum(vals) / len(vals), 4) if vals else None

            # Compute rouge_avg from rouge sub-scores if present
            rouge_keys = [k for k in aggregated[sys_id] if k in ("rouge1", "rouge2", "rougeL")]
            if rouge_keys:
                vals = [aggregated[sys_id][k] for k in rouge_keys if aggregated[sys_id][k] is not None]
                aggregated[sys_id]["rouge_avg"] = round(sum(vals) / len(vals), 4) if vals else None

        return aggregated

    @staticmethod
    def _aggregate_performance(
        all_perf: List[Dict[str, Dict[str, Any]]],
        system_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-system aggregate performance stats."""
        agg: Dict[str, Dict[str, Any]] = {}
        for sys_id in system_ids:
            latencies = []
            total_input_tokens = 0
            total_output_tokens = 0
            total_time = 0.0
            response_lengths = []

            for sample_perf in all_perf:
                p = sample_perf.get(sys_id)
                if not p or p["latency_sec"] == 0:
                    continue
                latencies.append(p["latency_sec"])
                total_input_tokens += p["input_tokens"]
                total_output_tokens += p["output_tokens"]
                total_time += p["latency_sec"]
                response_lengths.append(p["output_tokens"])

            if not latencies:
                agg[sys_id] = {}
                continue

            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)

            def percentile(data: list, pct: float) -> float:
                idx = int(pct / 100.0 * (len(data) - 1))
                return round(data[idx], 4)

            agg[sys_id] = {
                "num_predictions": n,
                "total_time_sec": round(total_time, 2),
                "avg_latency_sec": round(total_time / n, 4),
                "p50_latency_sec": percentile(latencies_sorted, 50),
                "p95_latency_sec": percentile(latencies_sorted, 95),
                "p99_latency_sec": percentile(latencies_sorted, 99),
                "min_latency_sec": round(latencies_sorted[0], 4),
                "max_latency_sec": round(latencies_sorted[-1], 4),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "avg_output_tokens": round(total_output_tokens / n, 1),
                "throughput_tokens_per_sec": round(total_output_tokens / total_time, 1)
                if total_time > 0 else 0.0,
            }
        return agg

    def _build_report(
        self,
        samples: List[Dict[str, str]],
        all_predictions: List[Dict[str, str]],
        all_scores: List[Dict[str, Dict[str, float]]],
        aggregate: Dict[str, Dict[str, float]],
        all_perf: List[Dict[str, Dict[str, Any]]],
        aggregate_perf: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble the final evaluation report."""
        systems_section: Dict[str, Any] = {}
        for sys_cfg in self.systems:
            if sys_cfg.id in aggregate:
                systems_section[sys_cfg.id] = {
                    "label": sys_cfg.label or sys_cfg.id,
                    "kind": sys_cfg.kind,
                    "aggregate_metrics": aggregate[sys_cfg.id],
                    "performance": aggregate_perf.get(sys_cfg.id, {}),
                }

        per_question: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            per_question.append({
                "index": idx,
                "question": sample["question"],
                "reference": sample["reference"],
                "predictions": all_predictions[idx],
                "scores": all_scores[idx],
                "performance": all_perf[idx],
            })

        return {
            "evalg_mode": "internal",
            "eval_dataset_path": str(Path(self.eval_dataset_path).resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(samples),
            "metrics_used": self.metrics,
            "systems": systems_section,
            "per_question": per_question,
        }

    def run(self) -> Dict[str, Any]:
        """Execute the full evaluation pipeline.

        Loads any existing checkpoint so work resumes where it left off.
        On successful completion the checkpoint file is removed.
        """
        logger.info("Starting internal evaluation")
        _emit_eval_event(
            "eval_start",
            systems=[{"id": s.id, "label": s.label, "kind": s.kind} for s in self.systems],
            metrics=self.metrics,
            eval_dataset_path=self.eval_dataset_path,
        )

        # 1. Load dataset
        samples = self._load_dataset()
        if not samples:
            logger.warning("No evaluation samples found")
            report = {
                "evalg_mode": "internal",
                "eval_dataset_path": str(Path(self.eval_dataset_path).resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "num_samples": 0,
                "metrics_used": self.metrics,
                "systems": {},
                "per_question": [],
            }
            Path(self.output_report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return report

        _emit_eval_event("dataset_loaded", num_samples=len(samples))

        # 1b. Load checkpoint for resume support
        checkpoint = self._load_checkpoint()

        # 2. Build scorers
        scorers = build_scorers(self.metrics, judge_config=self.judge_config)
        if not scorers:
            logger.warning("No scorers configured; predictions will be generated but not scored")

        # 3. Generate predictions sequentially per system (frees GPU VRAM between systems)
        logger.info("Generating predictions for %d systems across %d samples", len(self.systems), len(samples))
        active_system_ids, all_predictions, all_perf = self._generate_predictions(samples, checkpoint)

        if not active_system_ids:
            raise RuntimeError("No predictors could be initialised; aborting evaluation")

        # 4. Score predictions
        logger.info("Scoring predictions with %d scorer(s)", len(scorers))
        all_scores = self._score_predictions(samples, all_predictions, scorers, checkpoint)

        # 5. Aggregate
        aggregate = self._aggregate_metrics(all_scores, active_system_ids)
        aggregate_perf = self._aggregate_performance(all_perf, active_system_ids)

        for sys_id, perf in aggregate_perf.items():
            if perf:
                logger.info(
                    "Performance [%s]: avg_latency=%.3fs  throughput=%.1f tok/s  total=%.1fs",
                    sys_id, perf.get("avg_latency_sec", 0),
                    perf.get("throughput_tokens_per_sec", 0),
                    perf.get("total_time_sec", 0),
                )

        # 6. Build and write report
        report = self._build_report(
            samples, all_predictions, all_scores, aggregate, all_perf, aggregate_perf,
        )

        Path(self.output_report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report written to %s", self.output_report_path)

        # Clean up checkpoint after successful completion
        self._remove_checkpoint()

        _emit_eval_event(
            "eval_complete",
            num_samples=len(samples),
            active_systems=active_system_ids,
            aggregate_metrics=aggregate,
            aggregate_performance=aggregate_perf,
            report_path=self.output_report_path,
        )

        return report
