"""OpenAI (ChatGPT) LLM client implementation.

Works with the official OpenAI API as well as any OpenAI-compatible
endpoint (OpenRouter, vLLM, Together, etc.) via the ``base_url`` parameter.
"""
from typing import List, Optional
import logging
from openai import OpenAI

from .interface import LLMClient, LLMMessage

logger = logging.getLogger(__name__)


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Ensure the base URL ends with ``/v1`` for OpenAI-compatible APIs.

    The OpenAI Python SDK appends ``/chat/completions`` to whatever
    ``base_url`` you provide.  Many users pass just the host
    (``https://openrouter.ai/api``) and end up hitting a non-standard
    endpoint.  This helper auto-appends ``/v1`` when it looks like it
    was forgotten.
    """
    if not base_url:
        return None
    url = base_url.rstrip("/")
    # Already has /v1 — fine
    if url.endswith("/v1"):
        return url
    # Known providers where /v1 is required
    known_patterns = ("openrouter.ai",)
    for pattern in known_patterns:
        if pattern in url:
            corrected = f"{url}/v1"
            logger.info("Auto-corrected base_url: %s -> %s", base_url, corrected)
            return corrected
    return url


class OpenAIClient(LLMClient):
    """OpenAI ChatGPT implementation of LLMClient."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: Optional base URL for custom endpoints
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=_normalize_base_url(base_url))

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from OpenAI."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        logger.debug("LLM request: model=%s, messages=%d, temp=%.1f, max_tokens=%s", self.model, len(messages), temperature, max_tokens)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            # Defensive: some non-standard endpoints may return a raw string
            # instead of a ChatCompletion object.
            if isinstance(response, str):
                logger.debug("LLM response (raw string): %d chars", len(response))
                return response
            content = response.choices[0].message.content or ""
            logger.debug("LLM response: %d chars", len(content))
            return content
        except Exception as e:
            logger.error("OpenAI generation error: %s", e)
            raise
    
    def stream_generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate a streaming response from OpenAI."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
