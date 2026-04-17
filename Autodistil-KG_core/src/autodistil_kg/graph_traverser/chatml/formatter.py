"""CHATML formatter for converting graph data to conversations."""
from typing import Dict, Any, Optional
import logging

from .dataset import ChatMLConversation, ChatMLMessage

logger = logging.getLogger(__name__)


class ChatMLFormatter:
    """Formats graph data into CHATML conversations."""
    
    @staticmethod
    def format_node_prompt(
        node_data: Dict[str, Any],
        seed_prompt_template: Optional[str] = None
    ) -> str:
        """
        Format node data into a prompt.
        
        Args:
            node_data: Dictionary containing node information
            seed_prompt_template: Optional template string for the prompt
        """
        if seed_prompt_template:
            # Use template with node data
            try:
                return seed_prompt_template.format(**node_data)
            except KeyError as e:
                logger.warning(f"Missing key in template: {e}, using default format")
        
        # Default format
        labels = node_data.get("labels", [])
        properties = node_data.get("properties", {})
        
        prompt = f"Node Information:\n"
        prompt += f"Labels: {', '.join(labels)}\n"
        prompt += f"Properties:\n"
        for key, value in properties.items():
            prompt += f"  {key}: {value}\n"
        
        return prompt
    
    @staticmethod
    def create_conversation_pair(
        prompt: str,
        response: str,
        system_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMLConversation:
        """
        Create a CHATML conversation from prompt and response.
        
        Args:
            prompt: User prompt
            response: Assistant response
            system_message: Optional system message
            metadata: Optional metadata to attach
        """
        messages = []
        
        if system_message:
            messages.append(ChatMLMessage(role="system", content=system_message))
        
        messages.append(ChatMLMessage(role="user", content=prompt))
        messages.append(ChatMLMessage(role="assistant", content=response))
        
        return ChatMLConversation(messages=messages, metadata=metadata)
