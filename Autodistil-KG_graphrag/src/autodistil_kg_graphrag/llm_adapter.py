"""LlamaIndex-compatible LLM and Embedding adapters.

Uses plain HTTP requests to call any OpenAI-compatible API (OpenRouter,
vLLM, Ollama, etc.) — avoids the need for ``llama-index-llms-openai``
and ``llama-index-embeddings-openai`` packages.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import requests
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(CustomLLM):
    """LlamaIndex LLM that calls any OpenAI-compatible chat/completions API."""

    model_name: str = Field(default="gpt-4", description="Model identifier")
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the API (e.g. https://openrouter.ai/api/v1)",
    )
    _api_key: str = PrivateAttr(default="")
    _temperature: float = PrivateAttr(default=0.7)
    _max_tokens: int = PrivateAttr(default=4096)

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4",
        api_base: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model, api_base=api_base.rstrip("/"), **kwargs)
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            is_chat_model=True,
        )

    def _call_api(self, messages: list[dict], **kwargs: Any) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            # Disable thinking/reasoning for models that support it (Qwen3+).
            # Graph RAG needs structured outputs (Cypher, JSON), not reasoning traces.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        # Log the request
        last_msg = messages[-1]["content"] if messages else ""
        logger.info(
            "LLM request: model=%s, url=%s, messages=%d, last_msg=%.150s...",
            self.model_name, url, len(messages), last_msg,
        )

        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        try:
            body = resp.json()
        except ValueError:
            logger.error(
                "LLM API returned non-JSON response (status=%d, body=%.500s)",
                resp.status_code, resp.text,
            )
            resp.raise_for_status()
            raise ValueError(f"LLM API returned empty/non-JSON response (status {resp.status_code})")
        if "error" in body:
            logger.error("LLM API error: %s", body["error"])
            raise RuntimeError(f"LLM API error: {body['error']}")
        resp.raise_for_status()

        message = body["choices"][0]["message"]
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        usage = body.get("usage", {})

        # Log raw response details
        logger.info(
            "LLM response: content_len=%s, reasoning_len=%s, "
            "finish_reason=%s, tokens(in=%s, out=%s)",
            len(content) if content else "null",
            len(reasoning) if reasoning else "null",
            body["choices"][0].get("finish_reason"),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )

        # Qwen3/3.5 thinking mode: content may be null, actual text in reasoning_content
        if content is None:
            if reasoning:
                logger.info("LLM content was null, using reasoning_content (len=%d)", len(reasoning))
            content = reasoning or ""

        if content:
            logger.info("LLM output preview: %.300s", content)
        else:
            logger.warning("LLM returned empty content (all fields null)")

        return content

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        text = self._call_api([{"role": "user", "content": prompt}], **kwargs)
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs: Any):
        # Non-streaming fallback
        yield self.complete(prompt, **kwargs)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        formatted = [{"role": m.role.value, "content": m.content} for m in messages]
        text = self._call_api(formatted, **kwargs)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text)
        )

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        # Non-streaming fallback
        yield self.chat(messages, **kwargs)


class OpenAICompatibleEmbedding(BaseEmbedding):
    """LlamaIndex embedding that calls any OpenAI-compatible embeddings API."""

    _api_key: str = PrivateAttr(default="")
    _api_base: str = PrivateAttr(default="https://api.openai.com/v1")
    _dimensions: int = PrivateAttr(default=1536)

    def __init__(
        self,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        api_base: str = "https://api.openai.com/v1",
        dimensions: int = 1536,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model, embed_batch_size=10, **kwargs)
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._dimensions = dimensions

    def _get_embedding(self, text: str) -> list[float]:
        url = f"{self._api_base}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "input": text,
            "model": self.model_name,
            "dimensions": self._dimensions,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_embedding(query)
