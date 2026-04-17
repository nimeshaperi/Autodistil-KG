"""CHATML dataset generation and management."""
from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ChatMLMessage(BaseModel):
    """Represents a message in CHATML format."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatMLConversation(BaseModel):
    """Represents a complete conversation in CHATML format."""
    messages: List[ChatMLMessage]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "messages": [msg.model_dump() for msg in self.messages]
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ChatMLDataset:
    """Manages a collection of CHATML conversations."""

    def __init__(self) -> None:
        self.conversations: List[ChatMLConversation] = []

    def add_conversation(self, conversation: ChatMLConversation) -> None:
        self.conversations.append(conversation)

    def add_from_messages(
        self,
        messages: List[ChatMLMessage],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        conversation = ChatMLConversation(messages=messages, metadata=metadata)
        self.add_conversation(conversation)

    def to_list(self) -> List[Dict[str, Any]]:
        return [conv.to_dict() for conv in self.conversations]

    def to_jsonl(self) -> str:
        lines = []
        for conv in self.conversations:
            lines.append(json.dumps(conv.to_dict()))
        return "\n".join(lines)

    def save_jsonl(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_jsonl())
        logger.info(f"Saved {len(self.conversations)} conversations to {filepath}")

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2)
        logger.info(f"Saved {len(self.conversations)} conversations to {filepath}")

    def load_jsonl(self, filepath: str) -> None:
        self.conversations = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    messages = [
                        ChatMLMessage(role=msg["role"], content=msg["content"])
                        for msg in data["messages"]
                    ]
                    metadata = data.get("metadata")
                    self.add_conversation(
                        ChatMLConversation(messages=messages, metadata=metadata)
                    )
        logger.info(f"Loaded {len(self.conversations)} conversations from {filepath}")

    def __len__(self) -> int:
        return len(self.conversations)

    def __iter__(self):
        return iter(self.conversations)
