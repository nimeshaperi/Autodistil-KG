"""vLLM local LLM client implementation."""
from typing import List, Optional
import logging
import requests

from .interface import LLMClient, LLMMessage

logger = logging.getLogger(__name__)


class VLLMClient(LLMClient):
    """vLLM local LLM implementation of LLMClient."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: Optional[str] = None,
        chat_path: str = "/v1/chat/completions",
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM API base URL (e.g. http://localhost:8000)
            model: Model name (required for most vLLM servers)
            chat_path: API path for chat completions (default /v1/chat/completions)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
    
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from vLLM."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        url = f"{self.base_url}{self.chat_path}"
        logger.debug(
            "[LLM_REQ] vLLM POST %s | model=%s | messages=%d | temp=%.2f | max_tokens=%s | prompt_chars=%d",
            url, self.model, len(formatted_messages), temperature,
            max_tokens, sum(len(m["content"]) for m in formatted_messages),
        )
        try:
            response = requests.post(url, json=payload, timeout=1800)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"] or ""
            usage = response.json().get("usage", {})
            logger.debug(
                "[LLM_RES] vLLM | status=%d | response_chars=%d | usage=%s",
                response.status_code, len(content), usage,
            )
            return content
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(
                    "vLLM 404: %s not found. Ensure vLLM is started with: vllm serve <model> --host 0.0.0.0. "
                    "Test with: curl %s/v1/models",
                    url, self.base_url,
                )
            logger.error("vLLM generation error: %s", e)
            raise
        except Exception as e:
            logger.error("vLLM generation error: %s", e)
            raise
    
    def stream_generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate a streaming response from vLLM."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.base_url}{self.chat_path}",
                json=payload,
                stream=True,
                timeout=1800
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    import json
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
        except Exception as e:
            logger.error(f"vLLM streaming error: {e}")
            raise
