"""WebSocket endpoint for live pipeline streaming."""
import asyncio
import json
import logging
import queue as import_queue
from typing import Any, Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from autodistil_kg.pipeline import Pipeline
from autodistil_kg.pipeline.interfaces import PipelineContext, StageResult

from autodistil_kg.pipeline.interfaces import CancellationToken, PipelineCancelled

from ..config_loader import config_from_dict, context_from_config
from ..redis_client import REDIS_QUEUE_KEY, get_redis_url, pipeline_run_channel
from ..state import STAGE_ORDER, WORKSPACE, cancel_tokens, get_executor, make_json_safe, persist_run, prepare_run_dir, run_store

import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

# Will be set by main.py lifespan
redis_available = False


class QueueLogHandler(logging.Handler):
    """Log handler that pushes structured events to a thread-safe queue."""

    def __init__(self, queue: "import_queue.Queue[Dict[str, Any]]", level: int = logging.INFO):
        super().__init__(level)
        self.queue = queue
        self.allowed_loggers = {
            "autodistil_kg", "autodistil_kg_graphrag", "unsloth", "trl",
            "transformers", "datasets", "unsloth.trainer",
        }
        # Frontend receives INFO+ logs by default; DEBUG only when log_level is DEBUG
        self.frontend_level = logging.INFO

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not any(record.name.startswith(p) for p in self.allowed_loggers):
                return
            # Structured events always flow through regardless of level
            traversal_event = getattr(record, "traversal_event", None)
            if traversal_event and isinstance(traversal_event, dict):
                self.queue.put({"event": "traversal_progress", **traversal_event})
                return
            eval_event = getattr(record, "eval_event", None)
            if eval_event and isinstance(eval_event, dict):
                self.queue.put({"event": "eval_progress", **eval_event})
                return
            finetuner_event = getattr(record, "finetuner_event", None)
            if finetuner_event and isinstance(finetuner_event, dict):
                self.queue.put({"event": "finetuner_progress", **finetuner_event})
                return
            # Regular logs: filter by frontend_level
            if record.levelno < self.frontend_level:
                return
            msg = self.format(record)
            if len(msg) > 500:
                msg = msg[:497] + "..."
            self.queue.put({"event": "log", "level": record.levelname, "logger": record.name, "message": msg})
        except Exception:
            pass


async def _ws_run_via_redis(websocket: WebSocket, config_dict: Dict[str, Any]) -> None:
    """Enqueue job to Redis, subscribe to run channel, forward events."""
    import redis.asyncio as aioredis

    run_id = str(uuid.uuid4())
    try:
        _tmp_cfg = config_from_dict(config_dict, WORKSPACE)
        _tmp_pipe = Pipeline(_tmp_cfg)
        _tmp_order = _tmp_cfg.run_stages or list(_tmp_pipe.available_stages)
        pre_stages = [s for s in STAGE_ORDER if s in _tmp_order and s in _tmp_pipe.available_stages]
    except Exception:
        pre_stages = config_dict.get("run_stages", [])

    run_store[run_id] = {
        "status": "running", "context": None, "results": None, "error": None,
        "stages": pre_stages, "current_stage": pre_stages[0] if pre_stages else None, "events": [],
    }

    job = {"run_id": run_id, "config": config_dict}
    red = aioredis.from_url(get_redis_url(), decode_responses=True)
    try:
        await red.lpush(REDIS_QUEUE_KEY, json.dumps(job))
        await websocket.send_json({"event": "run_start", "run_id": run_id})
        channel = pipeline_run_channel(run_id)
        pubsub = red.pubsub()
        await pubsub.subscribe(channel)
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                if msg is None:
                    continue
                if msg["type"] != "message":
                    continue
                payload = json.loads(msg["data"])
                if run_id in run_store:
                    run_store[run_id].setdefault("events", []).append(payload)
                await websocket.send_json(payload)
                if payload.get("event") in ("done", "error"):
                    break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    finally:
        await red.aclose()


async def _ws_run_in_process(websocket: WebSocket, config_dict: Dict[str, Any], resume_from: str | None = None) -> None:
    """Run pipeline in-process and send events directly."""
    run_id = str(uuid.uuid4())
    token = CancellationToken()
    cancel_tokens[run_id] = token
    run_store[run_id] = {
        "status": "running", "context": None, "results": None, "error": None,
        "stages": [], "current_stage": None, "events": [],
        **({"resumed_from": resume_from} if resume_from else {}),
    }
    await websocket.send_json({"event": "run_start", "run_id": run_id, **({"resumed_from": resume_from} if resume_from else {})})
    run_dir = prepare_run_dir(run_id, config_dict, resume_from=resume_from)
    config = config_from_dict(config_dict, run_dir)
    pipeline = Pipeline(config)
    context = context_from_config(config)
    context.cancel_token = token
    run_order = config.run_stages or list(pipeline.available_stages)
    ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
    run_store[run_id]["stages"] = ordered
    results: List[StageResult] = []
    await websocket.send_json({"event": "pipeline_start", "stages": ordered})
    run_store[run_id]["events"].append({"event": "pipeline_start", "stages": ordered})

    log_queue: import_queue.Queue[Dict[str, Any]] = import_queue.Queue()
    log_level = getattr(logging, config_dict.get("log_level", "INFO"), logging.INFO)
    log_handler = QueueLogHandler(log_queue, level=log_level)
    log_handler.frontend_level = log_level
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    async def _drain() -> None:
        while True:
            try:
                payload = log_queue.get_nowait()
            except import_queue.Empty:
                break
            try:
                await websocket.send_json(make_json_safe(payload))
                run_store[run_id].setdefault("events", []).append(payload)
            except Exception:
                break  # WebSocket likely closed

    async def _drain_periodically(done_event: asyncio.Event) -> None:
        """Drain the log queue every 0.5s while a stage is running."""
        while not done_event.is_set():
            await _drain()
            try:
                await asyncio.wait_for(done_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

    loop = asyncio.get_event_loop()
    try:
        for name in ordered:
            if token.is_cancelled:
                results.append(StageResult(success=False, error="Cancelled by user"))
                break
            run_store[run_id]["current_stage"] = name
            await websocket.send_json({"event": "stage_start", "stage": name})
            run_store[run_id]["events"].append({"event": "stage_start", "stage": name})
            try:
                stage_done = asyncio.Event()
                drain_task = asyncio.ensure_future(_drain_periodically(stage_done))
                result = await loop.run_in_executor(
                    get_executor(), lambda n=name: pipeline.run_stage(n, context),
                )
                stage_done.set()
                await drain_task
                await _drain()
                results.append(result)
                evt = make_json_safe({"event": "stage_end", "stage": name, "success": result.success, "error": result.error, "metadata": result.metadata or {}})
                await websocket.send_json(evt)
                run_store[run_id]["events"].append(evt)
                # Persist after each stage so state survives disconnects
                run_store[run_id]["results"] = [
                    {"success": r.success, "error": r.error, "metadata": r.metadata or {}} for r in results
                ]
                persist_run(run_id)
                if not result.success:
                    break
            except PipelineCancelled:
                stage_done.set()
                await drain_task
                await _drain()
                evt = {"event": "stage_end", "stage": name, "success": False, "error": "Cancelled by user", "metadata": {}}
                await websocket.send_json(evt)
                run_store[run_id]["events"].append(evt)
                results.append(StageResult(success=False, error="Cancelled by user"))
                break
            except Exception as e:
                logger.exception("Stage %s failed", name)
                stage_done.set()
                await drain_task
                await _drain()
                evt = {"event": "stage_end", "stage": name, "success": False, "error": str(e), "metadata": {}}
                await websocket.send_json(evt)
                run_store[run_id]["events"].append(evt)
                results.append(StageResult(success=False, error=str(e)))
                break
    finally:
        root_logger.removeHandler(log_handler)
        try:
            await _drain()
        except Exception:
            pass
        cancel_tokens.pop(run_id, None)
        # Synthesize stage_end for any stage that started but never got a
        # stage_end event (e.g. process killed mid-stage). This ensures the
        # frontend can derive correct state from the event stream.
        stored_events = run_store[run_id].get("events", [])
        started_stages = {e["stage"] for e in stored_events if e.get("event") == "stage_start"}
        ended_stages = {e["stage"] for e in stored_events if e.get("event") == "stage_end"}
        for stage in started_stages - ended_stages:
            error_msg = next((r.error for r in results if not r.success), "Interrupted")
            synthetic_evt = {"event": "stage_end", "stage": stage, "success": False, "error": error_msg, "metadata": {}}
            stored_events.append(synthetic_evt)
        # Safety persist — ensures state is saved even if WS disconnected
        # during the stage loop (the post-loop persist below may not run).
        run_store[run_id]["results"] = [
            {"success": r.success, "error": r.error, "metadata": r.metadata or {}} for r in results
        ]
        if not results or not all(r.success for r in results):
            run_store[run_id]["status"] = "failed"
            run_store[run_id]["error"] = next((r.error for r in results if not r.success), "Interrupted")
        persist_run(run_id)

    cancelled = token.is_cancelled
    success = not cancelled and all(r.success for r in results)
    run_store[run_id]["status"] = "cancelled" if cancelled else ("completed" if success else "failed")
    run_store[run_id]["context"] = context.to_dict()
    run_store[run_id]["results"] = [
        {"success": r.success, "error": r.error, "metadata": r.metadata or {}} for r in results
    ]
    run_store[run_id]["error"] = next((r.error for r in results if not r.success), None)
    persist_run(run_id)
    try:
        await websocket.send_json(make_json_safe({
            "event": "done", "success": success, "cancelled": cancelled,
            "context": context.to_dict(),
            "results": [{"success": r.success, "error": r.error, "metadata": r.metadata or {}} for r in results],
        }))
    except Exception:
        pass  # WebSocket may have disconnected — state already persisted


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket: send {"action": "run", "config": {...}} to run a pipeline."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            if action == "stop":
                target_run_id = data.get("run_id")
                token = cancel_tokens.get(target_run_id) if target_run_id else None
                if token:
                    token.cancel()
                    await websocket.send_json({"event": "stop_acknowledged", "run_id": target_run_id})
                else:
                    await websocket.send_json({"event": "error", "message": "No active run to stop"})
                continue
            if action != "run":
                await websocket.send_json({"event": "error", "message": f"Unknown action: {action}"})
                continue
            config_dict = data.get("config") or {}
            resume_from = data.get("resume_from")  # optional: run_id to resume
            logger.info("Pipeline run requested via WebSocket, stages=%s, resume_from=%s", config_dict.get("run_stages", []), resume_from)
            try:
                config_from_dict(config_dict, WORKSPACE)
            except Exception as e:
                logger.exception("Pipeline config validation failed")
                await websocket.send_json({"event": "error", "message": str(e)})
                continue
            if redis_available:
                await _ws_run_via_redis(websocket, config_dict)
            else:
                await _ws_run_in_process(websocket, config_dict, resume_from=resume_from)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
