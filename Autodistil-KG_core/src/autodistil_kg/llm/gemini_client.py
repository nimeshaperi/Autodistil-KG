"""Google Gemini (Vertex AI) LLM client implementation."""
from typing import List, Optional
import logging

from .interface import LLMClient, LLMMessage

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Google Gemini (Vertex AI) implementation of LLMClient."""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "gemini-pro",
        credentials_path: Optional[str] = None
    ):
        """
        Initialize Gemini client.
        
        Args:
            project_id: Google Cloud project ID
            location: GCP location (e.g., "us-central1")
            model: Model name (e.g., "gemini-pro")
            credentials_path: Optional path to service account credentials
        """
        self.project_id = project_id
        self.location = location
        self.model = model
        self.credentials_path = credentials_path
        
        # Lazy import to avoid requiring vertexai if not used
        self._vertexai = None
        self._model = None
    
    def _initialize_client(self):
        """Lazy initialization of Vertex AI client."""
        if self._model is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
                
                if self.credentials_path:
                    import os
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
                
                vertexai.init(project=self.project_id, location=self.location)
                self._model = GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "vertexai package is required for GeminiClient. "
                    "Install with: pip install google-cloud-aiplatform"
                )
    
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from Gemini."""
        self._initialize_client()
        
        # Convert messages to Gemini format
        # Gemini uses a different message format
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        full_prompt = "\n".join(prompt_parts)
        
        try:
            generation_config = {
                "temperature": temperature,
                **kwargs
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            
            response = self._model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def stream_generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate a streaming response from Gemini."""
        self._initialize_client()
        
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        full_prompt = "\n".join(prompt_parts)
        
        try:
            generation_config = {
                "temperature": temperature,
                **kwargs
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            
            response = self._model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
