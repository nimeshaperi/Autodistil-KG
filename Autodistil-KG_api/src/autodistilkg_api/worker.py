"""
Background pipeline worker that processes jobs from a Redis queue.

When Redis is available, the API enqueues pipeline run requests into
``REDIS_QUEUE_KEY``.  This worker runs in a daemon thread, pops jobs,
executes the pipeline stages, and publishes progress events back to the
Redis pub/sub channel for the WebSocket layer to forward to clients.
"""
import json
import logging
import threading

import redis

from autodistil_kg.pipeline import Pipeline
from autodistil_kg.pipeline.interfaces import CancellationToken, PipelineCancelled

from .config_loader import config_from_dict, context_from_config
from .log_handlers import RedisLogHandler
from .redis_client import REDIS_QUEUE_KEY, get_redis_url, pipeline_run_channel
from .state import STAGE_ORDER, cancel_tokens, persist_run, prepare_run_dir, run_store

logger = logging.getLogger(__name__)


def pipeline_worker_loop(stop_event: threading.Event) -> None:
    """Process pipeline jobs from the Redis queue until *stop_event* is set.

    Each job is a JSON dict with ``run_id`` and ``config`` keys.  The worker:

    1. Prepares the run directory and config.
    2. Installs a :class:`RedisLogHandler` to stream events to the WebSocket.
    3. Runs stages in order, publishing lifecycle events after each.
    4. Persists run state after every stage so it survives crashes.
    """
    url = get_redis_url()
    try:
        r = redis.from_url(url, decode_responses=True)
        r.ping()
    except Exception as e:
        logger.warning("Redis not available for pipeline queue: %s", e)
        return
    logger.info("Pipeline queue worker started")

    while not stop_event.is_set():
        try:
            result = r.brpop(REDIS_QUEUE_KEY, timeout=2)
            if result is None:
                continue
            _, raw = result
            job = json.loads(raw)
            run_id = job["run_id"]
            config_dict = job["config"]
            run_dir = prepare_run_dir(run_id, config_dict)
            token = CancellationToken()
            cancel_tokens[run_id] = token
            run_store[run_id] = {
                "status": "running", "context": None, "results": None,
                "error": None, "stages": [], "current_stage": None, "events": [],
            }
            channel = pipeline_run_channel(run_id)

            log_level = getattr(logging, config_dict.get("log_level", "INFO"), logging.INFO)
            log_handler = RedisLogHandler(r, channel, level=log_level)
            log_handler.frontend_level = log_level
            log_handler.setFormatter(logging.Formatter("%(message)s"))
            root_logger = logging.getLogger()
            prev_root_level = root_logger.level
            if log_level < root_logger.level:
                root_logger.setLevel(log_level)
            root_logger.addHandler(log_handler)

            def _store_event(evt: dict) -> None:  # type: ignore[type-arg]
                run_store[run_id].setdefault("events", []).append(evt)
                r.publish(channel, json.dumps(evt))

            try:
                config = config_from_dict(config_dict, run_dir)
                pipeline = Pipeline(config)
                context = context_from_config(config)
                context.cancel_token = token
                run_order = config.run_stages or list(pipeline.available_stages)
                ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
                run_store[run_id]["stages"] = ordered
                _store_event({"event": "pipeline_start", "stages": ordered})
                results = []
                for name in ordered:
                    if token.is_cancelled:
                        results.append({"success": False, "error": "Cancelled by user", "metadata": {}})
                        break
                    run_store[run_id]["current_stage"] = name
                    _store_event({"event": "stage_start", "stage": name})
                    try:
                        stage_result = pipeline.run_stage(name, context)
                        results.append({"success": stage_result.success, "error": stage_result.error, "metadata": stage_result.metadata or {}})
                        _store_event({"event": "stage_end", "stage": name, "success": stage_result.success, "error": stage_result.error, "metadata": stage_result.metadata or {}})
                        run_store[run_id]["results"] = results[:]
                        persist_run(run_id)
                        if not stage_result.success:
                            break
                    except PipelineCancelled:
                        results.append({"success": False, "error": "Cancelled by user", "metadata": {}})
                        _store_event({"event": "stage_end", "stage": name, "success": False, "error": "Cancelled by user", "metadata": {}})
                        break
                    except Exception as e:
                        logger.exception("Stage %s failed", name)
                        results.append({"success": False, "error": str(e), "metadata": {}})
                        _store_event({"event": "stage_end", "stage": name, "success": False, "error": str(e), "metadata": {}})
                        break
                cancelled = token.is_cancelled
                success = not cancelled and all(x["success"] for x in results)
                _store_event({"event": "done", "success": success, "cancelled": cancelled, "context": context.to_dict(), "results": results})
                status = "cancelled" if cancelled else ("completed" if success else "failed")
                run_store[run_id].update({"status": status, "context": context.to_dict(), "results": results})
                run_store[run_id]["error"] = next((x.get("error") for x in results if not x.get("success")), None)
                persist_run(run_id)
            except Exception as e:
                logger.exception("Pipeline run failed for run_id=%s", run_id)
                run_store[run_id].update({"status": "failed", "error": str(e)})
                r.publish(channel, json.dumps({"event": "error", "message": str(e)}))
                persist_run(run_id)
            finally:
                root_logger.removeHandler(log_handler)
                root_logger.setLevel(prev_root_level)
                cancel_tokens.pop(run_id, None)
        except redis.ConnectionError:
            if not stop_event.is_set():
                logger.warning("Pipeline worker Redis connection lost")
        except Exception as e:
            if not stop_event.is_set():
                logger.exception("Pipeline worker error: %s", e)
    logger.info("Pipeline queue worker stopped")
