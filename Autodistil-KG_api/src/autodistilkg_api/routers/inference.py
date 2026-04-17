"""Inference / prompting endpoints."""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..models.requests import (
    InferenceLLMRequest,
    InferenceGraphRAGRequest,
    InferenceFinetunedRequest,
    RegisterModelRequest,
    UploadToHubRequest,
)
from ..models.responses import (
    InferenceLLMResponse,
    InferenceGraphRAGResponse,
    InferenceFinetunedResponse,
    ModelsListResponse,
    AvailableModel,
    RegisteredModel,
    RegisteredModelsResponse,
)
from ..state import WORKSPACE, get_executor, graphrag_engines, run_store

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------
MODELS_INDEX = WORKSPACE / "models.json"

# Cache of loaded finetuned model predictors (heavy — reuse across requests)
_finetuned_cache: Dict[str, Any] = {}


def _load_model_index() -> Dict[str, Dict[str, str]]:
    """Load the models.json index file."""
    if not MODELS_INDEX.exists():
        return {}
    try:
        return json.loads(MODELS_INDEX.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_model_index(index: Dict[str, Dict[str, str]]) -> None:
    MODELS_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")


@router.post("/llm", response_model=InferenceLLMResponse)
async def inference_llm(body: InferenceLLMRequest) -> InferenceLLMResponse:
    """Prompt a base model via any supported LLM provider."""
    try:
        from autodistil_kg.llm.config import LLMConfig as CoreLLMConfig
        from autodistil_kg.llm.factory import create_llm_client
        from autodistil_kg.llm.interface import LLMMessage

        config = CoreLLMConfig(
            provider=body.provider,
            api_key=body.api_key,
            model=body.model,
            base_url=body.base_url,
            project_id=body.project_id,
            location=body.location,
            credentials_path=body.credentials_path,
        )
        client = create_llm_client(config)
        messages = [LLMMessage(role=m["role"], content=m["content"]) for m in body.messages]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            get_executor(),
            lambda: client.generate(messages, temperature=body.temperature, max_tokens=body.max_tokens),
        )
        return InferenceLLMResponse(response=response, provider=body.provider, model=body.model)
    except Exception as e:
        logger.exception("LLM inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graphrag", response_model=InferenceGraphRAGResponse)
async def inference_graphrag(body: InferenceGraphRAGRequest) -> InferenceGraphRAGResponse:
    """Query through Graph RAG engine."""
    try:
        try:
            from autodistil_kg_graphrag.config import (
                GraphRAGConfig, Neo4jConfig as GRAGNeo4jConfig, LLMConfig as GRAGLLMConfig,
                EmbeddingConfig, RetrieverConfig, RetrieverType,
            )
            from autodistil_kg_graphrag.query_engine import GraphRAGEngine
        except ImportError as ie:
            raise HTTPException(
                status_code=503,
                detail=f"GraphRAG module not installed. Install with: pip install -e ../Autodistil-KG_graphrag — {ie}",
            )

        neo4j_cfg = GRAGNeo4jConfig(uri=body.neo4j_uri, user=body.neo4j_user, password=body.neo4j_password, database=body.neo4j_database)
        llm_cfg = GRAGLLMConfig(api_key=body.llm_api_key or "", model=body.llm_model, base_url=body.llm_base_url or None)
        embed_cfg = EmbeddingConfig(api_key=body.embedding_api_key or body.llm_api_key or "", model=body.embedding_model, base_url=body.embedding_base_url or None)
        retriever_types = []
        for r in body.retrievers:
            try:
                retriever_types.append(RetrieverType(r))
            except ValueError:
                pass
        retriever_cfg = RetrieverConfig(
            enabled=retriever_types or [RetrieverType.VECTOR, RetrieverType.CYPHER, RetrieverType.SYNONYM],
            similarity_top_k=body.similarity_top_k,
        )
        config = GraphRAGConfig(neo4j=neo4j_cfg, llm=llm_cfg, embedding=embed_cfg, retriever=retriever_cfg)

        cache_key = f"{body.neo4j_uri}:{body.neo4j_database}:{body.llm_model}:{body.embedding_model}"
        engine = graphrag_engines.get(cache_key)
        if engine is None:
            engine = GraphRAGEngine(config)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(get_executor(), engine.initialise)
            graphrag_engines[cache_key] = engine

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(get_executor(), lambda: engine.query(body.question))
        return InferenceGraphRAGResponse(answer=result.answer, source_nodes=result.source_nodes, metadata=result.metadata)
    except Exception as e:
        logger.exception("GraphRAG inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelsListResponse)
def list_available_models() -> ModelsListResponse:
    """List finetuned models from completed pipeline runs."""
    models = []
    for rid, rec in run_store.items():
        if rec.get("status") != "completed":
            continue
        ctx = rec.get("context") or {}
        model_path = ctx.get("model_output_path")
        if model_path and Path(model_path).exists():
            models.append(AvailableModel(run_id=rid, model_path=model_path, label=f"Run {rid[:8]} finetuned model"))
    runs_dir = WORKSPACE / "runs"
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for candidate in ["output/finetuned", "finetuner_output"]:
                model_dir = run_dir / candidate
                if model_dir.exists() and (model_dir / "config.json").exists():
                    rid = run_dir.name
                    if not any(m.run_id == rid for m in models):
                        models.append(AvailableModel(run_id=rid, model_path=str(model_dir), label=f"Run {rid[:8]} finetuned model"))
    return ModelsListResponse(models=models)


# ---------------------------------------------------------------------------
# Registered model CRUD
# ---------------------------------------------------------------------------

@router.get("/models/registered", response_model=RegisteredModelsResponse)
def list_registered_models() -> RegisteredModelsResponse:
    """List all named models from the models.json registry."""
    index = _load_model_index()
    return RegisteredModelsResponse(
        models=[
            RegisteredModel(model_id=mid, **entry)
            for mid, entry in index.items()
        ]
    )


@router.post("/models/register", response_model=RegisteredModel)
def register_model(body: RegisterModelRequest) -> RegisteredModel:
    """Register (or update) a named model in the index."""
    model_dir = Path(body.model_path)
    if not model_dir.exists():
        raise HTTPException(status_code=400, detail=f"Model path does not exist: {body.model_path}")

    index = _load_model_index()
    entry: Dict[str, str] = {"model_path": str(model_dir.resolve())}
    if body.base_model:
        entry["base_model"] = body.base_model
    if body.description:
        entry["description"] = body.description

    # Evict from cache if path changed
    if body.model_id in _finetuned_cache:
        old = index.get(body.model_id, {})
        if old.get("model_path") != entry["model_path"]:
            pred = _finetuned_cache.pop(body.model_id, None)
            if pred and hasattr(pred, "close"):
                pred.close()

    index[body.model_id] = entry
    _save_model_index(index)
    logger.info("Registered model %r → %s", body.model_id, entry["model_path"])
    return RegisteredModel(model_id=body.model_id, **entry)


@router.delete("/models/registered/{model_id}")
def unregister_model(model_id: str) -> Dict[str, str]:
    """Remove a model from the registry."""
    index = _load_model_index()
    if model_id not in index:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry")
    del index[model_id]
    _save_model_index(index)
    # Evict from cache
    pred = _finetuned_cache.pop(model_id, None)
    if pred and hasattr(pred, "close"):
        pred.close()
    return {"status": "ok", "model_id": model_id}


# ---------------------------------------------------------------------------
# Finetuned model inference
# ---------------------------------------------------------------------------

@router.post("/finetuned", response_model=InferenceFinetunedResponse)
async def inference_finetuned(body: InferenceFinetunedRequest) -> InferenceFinetunedResponse:
    """Run inference on a registered fine-tuned model."""
    index = _load_model_index()
    entry = index.get(body.model_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_id}' not found in registry. Register it first via POST /inference/models/register")

    model_path = entry["model_path"]
    if not Path(model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model path no longer exists: {model_path}")

    try:
        # Get or create predictor (cached for performance — model loading is slow)
        predictor = _finetuned_cache.get(body.model_id)
        if predictor is None:
            from autodistil_kg.eval.predictors import FinetunedModelPredictor

            logger.info("Loading finetuned model %r from %s (first request — this may take a moment)", body.model_id, model_path)
            predictor = FinetunedModelPredictor(
                model_path=model_path,
                max_new_tokens=body.max_new_tokens,
            )
            _finetuned_cache[body.model_id] = predictor

        # Extract system prompt and user message from the messages list
        system_prompt = ""
        user_question = ""
        for msg in body.messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_question = msg["content"]

        if not user_question:
            raise HTTPException(status_code=400, detail="Messages must include at least one 'user' message")

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            get_executor(),
            lambda: predictor.predict(system_prompt, user_question),
        )
        return InferenceFinetunedResponse(response=response, model_id=body.model_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Finetuned inference failed for model %r", body.model_id)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# HuggingFace Hub upload
# ---------------------------------------------------------------------------

@router.post("/models/upload-to-hub")
async def upload_model_to_hub(body: UploadToHubRequest) -> Dict[str, Any]:
    """Upload a fine-tuned LoRA adapter (or merged model) to HuggingFace Hub.

    The model directory must contain at least ``adapter_config.json`` or
    ``config.json``.  If ``merge_before_upload`` is True, the LoRA adapter
    is merged into the base model first (requires GPU and more memory).
    """
    model_dir = Path(body.model_path)
    if not model_dir.exists():
        raise HTTPException(status_code=400, detail=f"Model path does not exist: {body.model_path}")

    # Detect base model from adapter_config.json if not provided
    base_model = body.base_model
    if not base_model:
        adapter_cfg = model_dir / "adapter_config.json"
        if adapter_cfg.exists():
            try:
                cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                base_model = cfg.get("base_model_name_or_path")
            except Exception:
                pass

    try:
        loop = asyncio.get_event_loop()

        if body.merge_before_upload:
            # Merge LoRA into base model and push the full model
            if not base_model:
                raise HTTPException(
                    status_code=400,
                    detail="base_model is required when merge_before_upload is True (could not auto-detect).",
                )

            def _merge_and_push() -> str:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading base model %s for merge...", base_model)
                tokenizer = AutoTokenizer.from_pretrained(base_model, token=body.hf_token)
                model = AutoModelForCausalLM.from_pretrained(base_model, token=body.hf_token)
                logger.info("Loading LoRA adapter from %s...", model_dir)
                model = PeftModel.from_pretrained(model, str(model_dir))
                logger.info("Merging LoRA weights...")
                model = model.merge_and_unload()
                commit = body.commit_message or f"Upload merged model from {model_dir.name}"
                logger.info("Pushing merged model to %s...", body.repo_id)
                model.push_to_hub(body.repo_id, token=body.hf_token, private=body.private, commit_message=commit)
                tokenizer.push_to_hub(body.repo_id, token=body.hf_token, private=body.private)
                return f"https://huggingface.co/{body.repo_id}"

            repo_url = await loop.run_in_executor(get_executor(), _merge_and_push)
        else:
            # Push the LoRA adapter directory as-is
            def _push_adapter() -> str:
                from huggingface_hub import HfApi

                api = HfApi(token=body.hf_token)
                api.create_repo(repo_id=body.repo_id, private=body.private, exist_ok=True)
                commit = body.commit_message or f"Upload LoRA adapter from {model_dir.name}"
                logger.info("Uploading adapter directory %s to %s...", model_dir, body.repo_id)
                api.upload_folder(
                    folder_path=str(model_dir),
                    repo_id=body.repo_id,
                    commit_message=commit,
                )
                return f"https://huggingface.co/{body.repo_id}"

            repo_url = await loop.run_in_executor(get_executor(), _push_adapter)

        logger.info("Model uploaded to %s", repo_url)
        return {
            "status": "ok",
            "repo_url": repo_url,
            "repo_id": body.repo_id,
            "merged": body.merge_before_upload,
            "base_model": base_model,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("HuggingFace upload failed")
        raise HTTPException(status_code=500, detail=str(e))
