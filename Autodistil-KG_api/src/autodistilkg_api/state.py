"""Shared application state: run store, workspace, thread pool, constants."""
import concurrent.futures
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from autodistil_kg.pipeline.interfaces import CancellationToken

logger = logging.getLogger(__name__)

# Base directory for resolving config paths
WORKSPACE = Path(__file__).resolve().parents[2] / "workspace"
if os.environ.get("KG_PIPELINE_WORKSPACE"):
    WORKSPACE = Path(os.environ["KG_PIPELINE_WORKSPACE"]).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

# In-memory store for async run status
run_store: Dict[str, Dict[str, Any]] = {}

# Cancellation tokens for active runs
cancel_tokens: Dict[str, CancellationToken] = {}

# Ordered stage names -- re-exported from core for convenience
from autodistil_kg.pipeline.config import STAGE_ORDER  # noqa: F401,E402

# Artifact key mapping
ARTIFACT_KEYS = {
    "chatml": "chatml_dataset_path",
    "prepared": "prepared_dataset_path",
    "eval_report": "eval_report_path",
}

# GraphRAG engine cache
graphrag_engines: Dict[str, Any] = {}

# Thread pool
_executor: Optional[concurrent.futures.Executor] = None

# Lock for file writes
_persist_lock = threading.Lock()

RUN_STATE_FILENAME = "run_state.json"


def get_executor() -> concurrent.futures.Executor:
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    return _executor


def prepare_run_dir(run_id: str, config_dict: Dict[str, Any], resume_from: Optional[str] = None) -> Path:
    """Create a per-run workspace directory and inject run_id into config for isolation.

    If *resume_from* is set, the new run inherits the Redis key prefix from the
    old run (so visited-node state carries over) and copies the old checkpoint
    file into the new run directory so the traverser can resume.
    """
    run_dir = WORKSPACE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    gt = config_dict.get("graph_traverser")
    if gt and isinstance(gt, dict):
        redis_cfg = gt.setdefault("redis", {})
        if isinstance(redis_cfg, dict) and not redis_cfg.get("key_prefix"):
            redis_cfg["key_prefix"] = f"run:{run_id}:gt:"

    # Copy checkpoint files and inherit Redis prefix from the old run
    if resume_from:
        _inherit_resume_state(resume_from, run_dir, config_dict)
        config_dict["_resumed_from"] = resume_from

    # Persist the pipeline config for reproducibility / debugging
    _save_run_config(run_dir, config_dict)
    return run_dir


def _inherit_resume_state(old_run_id: str, new_run_dir: Path, config_dict: Dict[str, Any]) -> None:
    """Inherit checkpoint files and Redis prefix from a previous run.

    1. Copies dataset checkpoint files so the traverser can resume.
    2. Reads the old run's config to get its actual Redis key_prefix
       and applies it to the new config, ensuring visited-node state
       carries over regardless of whether the prefix was auto-generated
       or user-supplied.
    """
    import shutil

    old_run_dir = WORKSPACE / "runs" / old_run_id

    # --- Inherit Redis prefix from old config ---
    old_config_file = old_run_dir / "config.json"
    if old_config_file.exists():
        try:
            old_cfg = json.loads(old_config_file.read_text(encoding="utf-8"))
            old_prefix = (old_cfg.get("graph_traverser") or {}).get("redis", {}).get("key_prefix")
            if old_prefix:
                gt = config_dict.get("graph_traverser")
                if gt and isinstance(gt, dict):
                    gt.setdefault("redis", {})["key_prefix"] = old_prefix
                    logger.info("Resume: inherited Redis key_prefix=%s from run %s", old_prefix, old_run_id)
        except Exception as e:
            logger.warning("Resume: could not read old config for Redis prefix: %s", e)

    # --- Copy checkpoint files ---
    old_output = old_run_dir / "output"
    if not old_output.exists():
        # The old run may have used the global workspace/output/ path
        old_output = WORKSPACE / "output"
    new_output = new_run_dir / "output"
    new_output.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.jsonl", "dataset_eval.jsonl", "dataset_manifest.jsonl"):
        src = old_output / name
        if src.exists() and src.stat().st_size > 0:
            dst = new_output / name
            shutil.copy2(str(src), str(dst))
            logger.info("Resume: copied %s → %s (%d bytes)", src, dst, src.stat().st_size)


def _save_run_config(run_dir: Path, config_dict: Dict[str, Any]) -> None:
    """Write the pipeline config to the run directory for future reference."""
    try:
        config_file = run_dir / "config.json"
        safe = make_json_safe(config_dict)
        # Redact secrets before writing
        _redact_secrets(safe)
        config_file.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save run config: %s", e)


def _redact_secrets(obj: Any, _secret_keys: frozenset = frozenset({
    "password", "api_key", "base_model_api_key", "judge_api_key",
    "llm_api_key", "embedding_api_key",
})) -> None:
    """Redact known secret fields in-place."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _secret_keys and isinstance(v, str) and v:
                obj[k] = v[:4] + "***" if len(v) > 4 else "***"
            else:
                _redact_secrets(v)
    elif isinstance(obj, list):
        for item in obj:
            _redact_secrets(item)


def make_json_safe(obj: Any) -> Any:
    """Ensure dict is JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# JSON file persistence
# ---------------------------------------------------------------------------

def persist_run(run_id: str) -> None:
    """Write the current run state to a JSON file in the run directory."""
    rec = run_store.get(run_id)
    if rec is None:
        return
    run_dir = WORKSPACE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / RUN_STATE_FILENAME
    data = make_json_safe(rec)
    with _persist_lock:
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(state_file)


def load_persisted_runs() -> None:
    """Scan workspace/runs/ and reload run state from JSON files.

    Discovers runs in two ways:
    1. Runs with a ``run_state.json`` — full state is restored.
    2. Runs with only a ``config.json`` (e.g. process killed before state
       could be persisted) — a minimal "failed" record is synthesised so
       the run appears in history and can be resumed.
    """
    runs_dir = WORKSPACE / "runs"
    if not runs_dir.exists():
        return
    loaded = 0
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        state_file = run_dir / RUN_STATE_FILENAME
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                # Mark any previously "running" runs as failed (server restarted)
                if data.get("status") == "running":
                    data["status"] = "failed"
                    data["error"] = data.get("error") or "Server restarted while run was in progress"
                    # Synthesize missing stage_end / done events so frontend can
                    # derive correct state from the event stream on reconnect.
                    events = data.get("events", [])
                    started = {e["stage"] for e in events if e.get("event") == "stage_start"}
                    ended = {e["stage"] for e in events if e.get("event") == "stage_end"}
                    for stage in started - ended:
                        events.append({"event": "stage_end", "stage": stage, "success": False, "error": data["error"], "metadata": {}})
                    if not any(e.get("event") == "done" for e in events):
                        events.append({"event": "done", "success": False, "cancelled": False, "context": data.get("context"), "results": data.get("results")})
                run_store[run_id] = data
                loaded += 1
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load run state from %s: %s", state_file, e)
        else:
            # No run_state.json — try to reconstruct from config.json
            config_file = run_dir / "config.json"
            if not config_file.exists():
                continue
            try:
                cfg = json.loads(config_file.read_text(encoding="utf-8"))
                stages = cfg.get("run_stages", [])
                if not stages:
                    # Infer stages from which config sections exist
                    for s in STAGE_ORDER:
                        if cfg.get(s):
                            stages.append(s)
                run_store[run_id] = {
                    "status": "failed",
                    "error": "Process terminated before state could be saved",
                    "context": None,
                    "results": None,
                    "stages": stages,
                    "current_stage": None,
                    "events": [
                        {"event": "done", "success": False, "cancelled": False, "context": None, "results": None},
                    ],
                }
                loaded += 1
                logger.info("Reconstructed run %s from config.json (no run_state.json)", run_id)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to reconstruct run from %s: %s", config_file, e)
    if loaded:
        logger.info("Restored %d pipeline run(s) from disk", loaded)
