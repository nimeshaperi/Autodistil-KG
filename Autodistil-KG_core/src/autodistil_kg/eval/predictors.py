"""
Predictors: generate answers from different systems for evaluation.

Each predictor wraps a specific inference backend (LLM API, finetuned model,
or Graph RAG) behind a common interface.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a single prediction with performance metrics."""

    text: str
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _clean_model_output(text: str) -> str:
    """Clean raw model output by stripping thinking blocks and chat tokens.

    Qwen3/3.5 models produce ``<think>...</think>`` reasoning blocks before
    the actual answer.  Some models also leak chat-template markers like
    ``<|im_start|>`` / ``<|im_end|>`` into decoded text.
    """
    # Strip <think>...</think> blocks (greedy — remove all of them)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip any remaining chat-template markers
    text = re.sub(r"<\|im_start\|>.*?(?:\n|$)", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    # Strip <start_of_turn>/<end_of_turn> (Gemma)
    text = re.sub(r"<start_of_turn>.*?(?:\n|$)", "", text)
    text = re.sub(r"<end_of_turn>", "", text)
    return text.strip()


class Predictor(ABC):
    """Abstract base for a system that generates answers."""

    @abstractmethod
    def predict(self, system_prompt: str, user_question: str) -> PredictionResult:
        """Return a prediction result with text and performance metrics."""

    def close(self) -> None:
        """Release resources (optional override)."""


class BaseLLMPredictor(Predictor):
    """Generates predictions via the LLM factory (OpenAI, Ollama, vLLM, etc.)."""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        from ..llm.config import LLMConfig
        from ..llm.factory import create_llm_client

        extra = extra or {}
        llm_config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            project_id=extra.get("project_id"),
            location=extra.get("location"),
            credentials_path=extra.get("credentials_path"),
            additional_params=extra,
        )
        self._client = create_llm_client(llm_config)

    def predict(self, system_prompt: str, user_question: str) -> PredictionResult:
        from ..llm.interface import LLMMessage

        messages: List[LLMMessage] = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=user_question))

        input_chars = sum(len(m.content) for m in messages)
        t0 = time.perf_counter()
        text = self._client.generate(messages, temperature=0.1)
        latency = time.perf_counter() - t0

        # Approximate token counts (≈4 chars/token for English)
        input_tokens = input_chars // 4
        output_tokens = len(text) // 4
        return PredictionResult(
            text=text,
            latency_sec=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )


class FinetunedModelPredictor(Predictor):
    """Loads a LoRA-adapted model (or plain HF model) via Unsloth and runs local inference.

    Always uses the two-step approach: render chat template to text, then
    tokenize.  This avoids compatibility issues with Unsloth-modified
    tokenizers that break when ``apply_chat_template`` is called with
    ``tokenize=True, return_tensors="pt"`` combined.
    """

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 4096,
        max_new_tokens: int = 2048,
    ) -> None:
        from unsloth import FastLanguageModel

        logger.info("Loading model from %s (max_seq_length=%d, max_new_tokens=%d)",
                     model_path, max_seq_length, max_new_tokens)
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
        )
        # Disable sliding window attention for Qwen3 models — the hybrid
        # full+sliding-window mechanism causes KV cache shape mismatches
        # during generate() ("output shape doesn't match broadcast shape").
        if hasattr(self._model.config, "sliding_window"):
            self._model.config.sliding_window = None
        FastLanguageModel.for_inference(self._model)
        self._max_new_tokens = max_new_tokens

    def predict(self, system_prompt: str, user_question: str) -> PredictionResult:
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_question})

        # Two-step: render template → tokenize.  Avoids Unsloth tokenizer
        # incompatibilities ("string indices must be integers" etc.).
        # Disable Qwen3 thinking mode (``enable_thinking=False``) to avoid
        # the model wasting the token budget on <think> reasoning blocks.
        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Only pass enable_thinking when the tokenizer supports it (Qwen3+)
        try:
            text = self._tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages, **template_kwargs,
            )
        # Use the underlying text tokenizer to avoid VL processor routing
        # the input through an image pipeline (e.g. Qwen3-VL).
        raw_tokenizer = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        input_ids = raw_tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self._model.device)

        input_token_count = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        latency = time.perf_counter() - t0

        # Decode only the newly generated tokens.
        new_tokens = outputs[0][input_ids.shape[1]:]
        output_token_count = len(new_tokens)
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return PredictionResult(
            text=_clean_model_output(raw),
            latency_sec=latency,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            total_tokens=input_token_count + output_token_count,
        )

    def close(self) -> None:
        """Free model from GPU memory so the next system can use the VRAM."""
        import gc
        import torch

        logger.info("Unloading model from GPU")
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class VLLMPredictor(Predictor):
    """Generates predictions via a local vLLM OpenAI-compatible server.

    Used for both base model and LoRA-adapted (finetuned) inference
    when vLLM serving is enabled.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 2048,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        logger.info("VLLMPredictor: model=%s at %s", model, base_url)

    def predict(self, system_prompt: str, user_question: str) -> PredictionResult:
        import requests as _requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_question})

        t0 = time.perf_counter()
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self._max_tokens,
            # Disable thinking/reasoning for models that support it (Qwen3+)
            # so evaluation measures answer quality, not reasoning traces.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _requests.post(
            f"{self._base_url}/chat/completions",
            json=body,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - t0

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return PredictionResult(
            text=_clean_model_output(text),
            latency_sec=latency,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )


class GraphRAGPredictor(Predictor):
    """Wraps the GraphRAGEngine from the graphrag submodule."""

    def __init__(self, rag_config: Dict[str, Any]) -> None:
        from autodistil_kg_graphrag.config import (
            GraphRAGConfig,
            Neo4jConfig,
            LLMConfig as GraphRAGLLMConfig,
            EmbeddingConfig,
            RetrieverConfig,
            RetrieverType,
        )
        from autodistil_kg_graphrag.query_engine import GraphRAGEngine

        neo4j_cfg = Neo4jConfig(
            uri=rag_config.get("neo4j_uri", "bolt://localhost:7687"),
            user=rag_config.get("neo4j_user", "neo4j"),
            password=rag_config.get("neo4j_password", "password"),
            database=rag_config.get("neo4j_database", "neo4j"),
        )
        llm_cfg = GraphRAGLLMConfig(
            api_key=rag_config.get("llm_api_key", "") or rag_config.get("api_key", ""),
            model=rag_config.get("llm_model", "") or rag_config.get("model", "gpt-4"),
            base_url=rag_config.get("llm_base_url", "") or rag_config.get("base_url") or None,
        )
        embed_cfg = EmbeddingConfig(
            api_key=rag_config.get("embedding_api_key", "") or llm_cfg.api_key,
            model=rag_config.get("embedding_model", "text-embedding-3-small"),
            dimensions=int(rag_config.get("embedding_dimensions", 1536)),
            base_url=rag_config.get("embedding_base_url") or None,
        )

        enabled_retrievers = rag_config.get("retrievers", ["vector", "cypher", "synonym"])
        retriever_cfg = RetrieverConfig(
            enabled=[RetrieverType(r) for r in enabled_retrievers],
        )

        config = GraphRAGConfig(
            neo4j=neo4j_cfg,
            llm=llm_cfg,
            embedding=embed_cfg,
            retriever=retriever_cfg,
        )

        self._engine = GraphRAGEngine(config)
        logger.info("Initialising GraphRAGEngine for evaluation")
        self._engine.initialise()

    def predict(self, system_prompt: str, user_question: str) -> PredictionResult:
        t0 = time.perf_counter()
        response = self._engine.query(user_question)
        latency = time.perf_counter() - t0

        text = response.answer
        # Approximate tokens (no tokenizer available for RAG pipeline)
        input_tokens = len(user_question) // 4
        output_tokens = len(text) // 4
        return PredictionResult(
            text=text,
            latency_sec=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )


def build_predictor(system_config) -> Predictor:
    """Build the appropriate Predictor from an EvalSystemConfig."""
    kind = system_config.kind

    if kind == "distilled":
        if not system_config.model_path:
            raise ValueError("Distilled system requires model_path")
        extra = system_config.extra or {}
        # Use vLLM if a server URL is provided
        vllm_url = extra.get("vllm_base_url")
        if vllm_url:
            vllm_model = extra.get("vllm_model_name", "finetuned")
            return VLLMPredictor(
                base_url=vllm_url,
                model=vllm_model,
                max_tokens=int(extra.get("max_new_tokens", 2048)),
            )
        return FinetunedModelPredictor(
            model_path=system_config.model_path,
            max_seq_length=int(extra.get("max_seq_length", 4096)),
            max_new_tokens=int(extra.get("max_new_tokens", 2048)),
        )

    if kind == "local_base":
        if not system_config.model_path:
            raise ValueError("Local base system requires model_path (HuggingFace model name)")
        extra = system_config.extra or {}
        # Use vLLM if a server URL is provided
        vllm_url = extra.get("vllm_base_url")
        if vllm_url:
            vllm_model = extra.get("vllm_model_name", system_config.model_path)
            return VLLMPredictor(
                base_url=vllm_url,
                model=vllm_model,
                max_tokens=int(extra.get("max_new_tokens", 2048)),
            )
        return FinetunedModelPredictor(
            model_path=system_config.model_path,
            max_seq_length=int(extra.get("max_seq_length", 4096)),
            max_new_tokens=int(extra.get("max_new_tokens", 2048)),
        )

    if kind == "base":
        if not system_config.provider:
            raise ValueError("Base system requires provider")
        extra = system_config.extra or {}
        return BaseLLMPredictor(
            provider=system_config.provider,
            model=system_config.model,
            api_key=extra.get("api_key"),
            base_url=extra.get("base_url"),
            extra=extra,
        )

    if kind == "external":
        if not system_config.provider:
            raise ValueError("External system requires provider")
        extra = system_config.extra or {}
        return BaseLLMPredictor(
            provider=system_config.provider,
            model=system_config.model,
            api_key=extra.get("api_key"),
            base_url=extra.get("base_url"),
            extra=extra,
        )

    if kind == "graph_rag":
        if not system_config.rag_config:
            raise ValueError("Graph RAG system requires rag_config")
        return GraphRAGPredictor(rag_config=system_config.rag_config)

    raise ValueError(f"Unknown system kind: {kind}")
