"""Pipeline execution and management endpoints."""
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from autodistil_kg.pipeline import Pipeline
from autodistil_kg.pipeline.interfaces import PipelineContext, StageResult

from ..config_loader import config_from_dict, context_from_config
from ..models.requests import PipelineRunRequest
from ..models.responses import (
    FileUploadResponse,
    PipelineRunResultResponse,
    RunEventsResponse,
    RunListResponse,
    RunSummary,
    StageResultResponse,
)
from autodistil_kg.pipeline.interfaces import CancellationToken, PipelineCancelled
from ..state import (
    ARTIFACT_KEYS,
    STAGE_ORDER,
    WORKSPACE,
    cancel_tokens,
    get_executor,
    make_json_safe,
    persist_run,
    prepare_run_dir,
    run_store,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_pipeline_sync(
    config_dict: Dict[str, Any],
    base_dir: Path,
    run_id: str | None = None,
    cancel_token: CancellationToken | None = None,
) -> tuple[PipelineContext, list[StageResult]]:
    """Build config, pipeline, and run to completion."""
    config = config_from_dict(config_dict, base_dir)
    pipeline = Pipeline(config)
    context = context_from_config(config)
    if cancel_token is not None:
        context.cancel_token = cancel_token
    run_order = config.run_stages or list(pipeline.available_stages)
    ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
    results: List[StageResult] = []
    for name in ordered:
        # Check cancellation before each stage
        if cancel_token and cancel_token.is_cancelled:
            results.append(StageResult(success=False, error="Cancelled by user"))
            break
        if run_id and run_id in run_store:
            run_store[run_id]["current_stage"] = name
            run_store[run_id]["events"].append({"event": "stage_start", "stage": name})
        try:
            result = pipeline.run_stage(name, context)
        except PipelineCancelled:
            result = StageResult(success=False, error="Cancelled by user")
        results.append(result)
        if run_id and run_id in run_store:
            run_store[run_id]["events"].append({
                "event": "stage_end",
                "stage": name,
                "success": result.success,
                "error": result.error,
                "metadata": result.metadata or {},
            })
        if not result.success:
            break
    return context, results


@router.post("/run", response_model=PipelineRunResultResponse)
def run_pipeline(
    body: PipelineRunRequest,
    async_run: bool = Query(False, alias="async"),
) -> PipelineRunResultResponse:
    """Run a pipeline. Use ?async=true to run in background."""
    config_dict = body.model_dump(exclude_none=True)

    if async_run:
        run_id = str(uuid.uuid4())
        run_dir = prepare_run_dir(run_id, config_dict)
        try:
            config = config_from_dict(config_dict, run_dir)
            pipeline = Pipeline(config)
            run_order = config.run_stages or list(pipeline.available_stages)
            ordered = [s for s in STAGE_ORDER if s in run_order and s in pipeline.available_stages]
        except Exception:
            ordered = []
        token = CancellationToken()
        cancel_tokens[run_id] = token
        run_store[run_id] = {
            "status": "running", "context": None, "results": None, "error": None,
            "stages": ordered, "current_stage": ordered[0] if ordered else None, "events": [],
        }

        def task() -> None:
            try:
                run_store[run_id]["events"].append({"event": "pipeline_start", "stages": ordered})
                context, results = _run_pipeline_sync(config_dict, run_dir, run_id=run_id, cancel_token=token)
                cancelled = token.is_cancelled
                run_store[run_id]["status"] = "cancelled" if cancelled else "completed"
                run_store[run_id]["current_stage"] = None
                run_store[run_id]["context"] = context.to_dict()
                run_store[run_id]["results"] = [
                    {"success": r.success, "error": r.error, "metadata": r.metadata} for r in results
                ]
                if cancelled:
                    run_store[run_id]["events"].append({"event": "done", "success": False, "cancelled": True})
                else:
                    run_store[run_id]["events"].append({"event": "done", "success": all(r.success for r in results)})
                persist_run(run_id)
            except Exception as e:
                logger.exception("Async pipeline run failed")
                run_store[run_id]["status"] = "failed"
                run_store[run_id]["error"] = str(e)
                run_store[run_id]["events"].append({"event": "error", "message": str(e)})
                persist_run(run_id)
            finally:
                cancel_tokens.pop(run_id, None)

        threading.Thread(target=task, daemon=True).start()
        return PipelineRunResultResponse(run_id=run_id, status="running", success=True)

    try:
        sync_run_id = str(uuid.uuid4())
        sync_run_dir = prepare_run_dir(sync_run_id, config_dict)
        context, results = _run_pipeline_sync(config_dict, sync_run_dir)
        success = all(r.success for r in results)
        return PipelineRunResultResponse(
            run_id=sync_run_id,
            status="completed",
            success=success,
            context=context.to_dict(),
            results=[StageResultResponse(success=r.success, error=r.error, metadata=r.metadata or {}) for r in results],
            error=next((r.error for r in results if not r.success), None),
        )
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs", response_model=RunListResponse)
def list_runs() -> RunListResponse:
    """List all known pipeline runs (newest first)."""
    runs = []
    for rid, rec in run_store.items():
        resumable = False
        checkpoint_convs = None
        if rec.get("status") in ("failed", "cancelled"):
            # Check if a checkpoint file exists for this run
            cp = _find_checkpoint(rid)
            if cp is not None:
                resumable = True
                checkpoint_convs = cp
        runs.append(RunSummary(
            run_id=rid,
            status=rec.get("status", "unknown"),
            error=rec.get("error"),
            stages=rec.get("stages"),
            resumable=resumable,
            checkpoint_conversations=checkpoint_convs,
        ))
    runs.reverse()
    return RunListResponse(runs=runs)


def _find_checkpoint(run_id: str) -> int | None:
    """Return the number of conversations in a run's checkpoint file, or None."""
    for candidate in (
        WORKSPACE / "runs" / run_id / "output" / "dataset.jsonl",
        WORKSPACE / "output" / "dataset.jsonl",
    ):
        if candidate.exists() and candidate.stat().st_size > 0:
            return sum(1 for _ in open(candidate, encoding="utf-8"))
    return None


@router.get("/runs/{run_id}", response_model=PipelineRunResultResponse)
def get_run_status(run_id: str) -> PipelineRunResultResponse:
    """Get status and result of a pipeline run."""
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    rec = run_store[run_id]
    context = rec.get("context")
    if context is not None:
        context = make_json_safe(context)
    results = rec.get("results")
    stages = rec.get("stages")
    current = rec.get("current_stage")
    if not current and stages and rec["status"] == "running":
        current = stages[0] if stages else None
    return PipelineRunResultResponse(
        run_id=run_id,
        status=rec["status"],
        success=rec["status"] == "completed" and not any(
            r.get("success") is False for r in (results or [])
        ),
        context=context,
        results=[StageResultResponse(**r) for r in results] if results else None,
        error=rec.get("error"),
        stages=stages,
        current_stage=current,
    )


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> Dict[str, Any]:
    """Stop a running pipeline."""
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    rec = run_store[run_id]
    if rec["status"] != "running":
        raise HTTPException(status_code=409, detail=f"Run is not running (status: {rec['status']})")
    token = cancel_tokens.get(run_id)
    if token is None:
        raise HTTPException(status_code=409, detail="No cancellation token for this run")
    token.cancel()
    rec["events"].append({"event": "stop_requested"})
    return {"run_id": run_id, "message": "Stop signal sent"}


@router.delete("/runs/{run_id}")
def delete_run(run_id: str) -> Dict[str, Any]:
    """Delete a pipeline run and its workspace files.

    Removes the run from the in-memory store and deletes the
    ``workspace/runs/{run_id}/`` directory recursively.
    Running pipelines cannot be deleted — stop them first.
    """
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    rec = run_store[run_id]
    if rec.get("status") == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running pipeline. Stop it first.")

    # Remove from in-memory store
    run_store.pop(run_id, None)
    cancel_tokens.pop(run_id, None)

    # Delete workspace directory
    import shutil
    run_dir = WORKSPACE / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    logger.info("Deleted run %s", run_id)
    return {"status": "ok", "run_id": run_id}


@router.get("/runs/{run_id}/events", response_model=RunEventsResponse)
def get_run_events(run_id: str, since: int = Query(0, ge=0)) -> RunEventsResponse:
    """Return stored events for a run (for replaying progress)."""
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    events = run_store[run_id].get("events", [])
    return RunEventsResponse(run_id=run_id, events=events[since:], total=len(events))


@router.get("/runs/{run_id}/config")
def get_run_config(run_id: str) -> Dict[str, Any]:
    """Return the original pipeline config for a run."""
    config_file = WORKSPACE / "runs" / run_id / "config.json"
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Config not found for this run")
    import json
    return json.loads(config_file.read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/artifacts/{artifact_key}")
def get_run_artifact(run_id: str, artifact_key: str) -> FileResponse:
    """Download an output file from a run."""
    if artifact_key not in ARTIFACT_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown artifact: {artifact_key}. Use one of: {list(ARTIFACT_KEYS)}")
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    rec = run_store[run_id]
    context = rec.get("context") or {}
    path_key = ARTIFACT_KEYS[artifact_key]
    raw_path = context.get(path_key)
    if not raw_path or not isinstance(raw_path, str):
        raise HTTPException(status_code=404, detail=f"No {path_key} for this run")
    file_path = Path(raw_path).resolve()
    try:
        file_path.relative_to(WORKSPACE.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Artifact path is outside workspace")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=file_path.name, media_type="application/json")


_ALLOWED_UPLOAD_EXTS = {".jsonl", ".json", ".csv"}


@router.post("/uploads", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)) -> FileUploadResponse:
    """Upload a data file to the workspace (e.g. a training dataset)."""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_UPLOAD_EXTS)}",
        )
    safe_stem = re.sub(r"[^\w\-]", "_", Path(filename).stem)
    safe_name = f"{safe_stem}_{uuid.uuid4().hex[:8]}{suffix}"

    uploads_dir = WORKSPACE / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = (uploads_dir / safe_name).resolve()

    try:
        dest.relative_to(WORKSPACE.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid file path")

    content = await file.read()
    dest.write_bytes(content)
    return FileUploadResponse(path=f"uploads/{safe_name}")
