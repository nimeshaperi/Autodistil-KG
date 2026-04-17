"""Anthropic Claude LLM client implementation."""
from typing import List, Optional
import logging
from anthropic import Anthropic

from .interface import LLMClient, LLMMessage

logger = logging.getLogger(__name__)


class ClaudeClient(LLMClient):
    """Anthropic Claude implementation of LLMClient."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229"
    ):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key
            model: Model name (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229")
        """
        self.api_key = api_key
        self.model = model
        self.client = Anthropic(api_key=api_key)
    
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from Claude."""
        # Claude uses a different message format
        system_message = None
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                system=system_message,
                messages=formatted_messages,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            raise
    
    def stream_generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate a streaming response from Claude."""
        system_message = None
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                system=system_message,
                messages=formatted_messages,
                **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            raise
