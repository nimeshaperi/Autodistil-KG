"""Local embedding model using sentence-transformers.

Falls back to this when no external embedding API key is configured,
keeping the vector retriever functional without requiring an OpenAI key.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"


class LocalSentenceTransformerEmbedding(BaseEmbedding):
    """LlamaIndex embedding backed by a local sentence-transformers model."""

    _model: Any = PrivateAttr(default=None)

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, embed_batch_size=32, **kwargs)
        self._model = None  # lazy-loaded

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_text_embedding(self, text: str) -> list[float]:
        model = self._load_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)


def create_local_embedding(model_name: Optional[str] = None) -> Optional[BaseEmbedding]:
    """Try to create a local embedding model.

    Returns ``None`` if sentence-transformers is not installed.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.info(
            "sentence-transformers not installed — cannot use local embeddings. "
            "Install with: pip install sentence-transformers"
        )
        return None

    name = model_name or DEFAULT_LOCAL_MODEL
    logger.info("Using local embedding model: %s", name)
    return LocalSentenceTransformerEmbedding(model_name=name)
