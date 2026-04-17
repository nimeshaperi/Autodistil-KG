"""Retriever construction for Graph RAG.

Each retriever strategy is independently toggleable via
:class:`~autodistil_kg_graphrag.config.RetrieverConfig`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, TYPE_CHECKING

from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    TextToCypherRetriever,
    VectorContextRetriever,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from .config import RetrieverConfig, RetrieverType

if TYPE_CHECKING:
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

logger = logging.getLogger(__name__)


_CYPHER_KEYWORDS = re.compile(
    r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|RETURN|CALL|UNWIND|CREATE|MERGE|"
    r"DELETE|REMOVE|SET|UNION|ORDER\s+BY|LIMIT|SKIP|WHERE)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_thinking(text: str) -> str:
    """Strip reasoning preamble emitted by Qwen3 / DeepSeek-R1 and similar models.

    Handles:
    * ``<think>...</think>`` XML blocks
    * Orphaned ``</think>`` — everything before it is a reasoning preamble
    * Plain-text ``Thinking:`` / ``**Thinking**:`` lines without XML tags
    * Any other preamble before the first Cypher keyword line
    """
    # XML-tagged thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]

    # Plain-text "Thinking:" header lines (Qwen3 non-XML variant)
    text = re.sub(r"(?i)^\*{0,2}thinking\*{0,2}\s*:.*?(\n\n|\Z)", "", text, flags=re.DOTALL)

    text = text.strip()

    # Last-resort: if the text still doesn't start with a Cypher keyword,
    # scan forward to the first line that does.
    if text and not _CYPHER_KEYWORDS.match(text):
        match = _CYPHER_KEYWORDS.search(text)
        if match:
            text = text[match.start():]

    return text.strip()


# Neo4j 3.x compatible Cypher prompt — avoids modern syntax the LLM might generate
_NEO4J3_CYPHER_TEMPLATE = """\
Generate a Cypher query to answer the following question using the provided graph schema.

IMPORTANT: You are targeting Neo4j 3.x. Do NOT use any of the following:
- IF NOT EXISTS
- CREATE OR REPLACE
- CALL {{ }} subqueries
- EXISTS {{ }} subqueries
- SHOW INDEXES / SHOW CONSTRAINTS
- vector.similarity.cosine or any vector functions
- db.index.vector.queryNodes
- USE database

Only use basic Cypher: MATCH, WHERE, RETURN, WITH, UNWIND, ORDER BY, LIMIT, OPTIONAL MATCH, UNION.

Cypher syntax rules:
- Each WHERE must directly follow a MATCH, OPTIONAL MATCH, or WITH clause.
- Do NOT place two WHERE clauses in a row without an intervening MATCH/OPTIONAL MATCH/WITH.
- When filtering after OPTIONAL MATCH, place the condition in the WHERE of that OPTIONAL MATCH, not as a separate WHERE.

Schema:
{schema}

Question: {question}

Respond with ONLY the Cypher query, no explanation.
"""

_CYPHER_RETRY_TEMPLATE = PromptTemplate("""\
The following Cypher query failed with a syntax error on the Neo4j server.

Failed query:
{failed_query}

Error:
{error}

Schema:
{schema}

Original question: {question}

Please generate a corrected Cypher query that fixes the syntax error. \
Respond with ONLY the Cypher query, no explanation.
""")

# Patterns that indicate Neo4j 4/5+ syntax
_NEO4J_MODERN_PATTERNS = re.compile(
    r"\bIF\s+NOT\s+EXISTS\b"
    r"|\bCREATE\s+OR\s+REPLACE\b"
    r"|\bCALL\s*\{"
    r"|\bEXISTS\s*\{"
    r"|\bSHOW\s+(?:INDEX|CONSTRAINT)"
    r"|\bvector\s*\.\s*similarity"
    r"|\bdb\s*\.\s*index\s*\.\s*vector"
    r"|\bUSE\s+\w+",
    re.IGNORECASE,
)


def _cypher_validator_neo4j3(cypher: str) -> str:
    """Strip or reject Cypher that uses Neo4j 4/5+ features."""
    if _NEO4J_MODERN_PATTERNS.search(cypher):
        logger.warning("Rejecting LLM-generated Cypher with modern syntax: %.120s", cypher)
        raise ValueError("Generated Cypher uses Neo4j 4+ syntax not supported by this server")
    return cypher


_MAX_CYPHER_RETRIES = 2


class _RetryTextToCypherRetriever(TextToCypherRetriever):
    """TextToCypherRetriever that retries on Neo4j syntax errors.

    When the generated Cypher fails execution, this retriever feeds the
    error back to the LLM and asks for a corrected query, up to
    ``_MAX_CYPHER_RETRIES`` times.
    """

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        schema = self._graph_store.get_schema_str()
        question = query_bundle.query_str

        response = _strip_thinking(self.llm.predict(
            self.text_to_cypher_template,
            schema=schema,
            question=question,
        ))
        parsed_cypher = self._parse_generated_cypher(response)

        # Guard: if LLM returned empty/whitespace Cypher, return empty results
        if not parsed_cypher or not parsed_cypher.strip():
            logger.warning("LLM returned empty Cypher query — skipping Cypher retriever for this question")
            return []

        query_output = None
        for attempt in range(_MAX_CYPHER_RETRIES + 1):
            try:
                query_output = self._graph_store.structured_query(parsed_cypher)
                break
            except Exception as exc:
                err_str = str(exc)
                is_syntax = "SyntaxError" in err_str or "syntax" in err_str.lower()
                if not is_syntax or attempt == _MAX_CYPHER_RETRIES:
                    logger.warning("Cypher retriever failed after %d attempts: %s", attempt + 1, err_str[:200])
                    return []
                logger.warning(
                    "Cypher syntax error (attempt %d/%d), retrying: %s",
                    attempt + 1,
                    _MAX_CYPHER_RETRIES + 1,
                    err_str[:200],
                )
                retry_response = _strip_thinking(self.llm.predict(
                    _CYPHER_RETRY_TEMPLATE,
                    failed_query=parsed_cypher,
                    error=err_str[:500],
                    schema=schema,
                    question=question,
                ))
                parsed_cypher = self._parse_generated_cypher(retry_response)
                if not parsed_cypher or not parsed_cypher.strip():
                    logger.warning("LLM returned empty Cypher on retry — giving up")
                    return []

        cleaned = self._clean_query_output(query_output)

        if self.summarize_response:
            node_text = self.llm.predict(
                self.summarization_template,
                context=str(cleaned),
                question=parsed_cypher,
            )
        else:
            node_text = self.response_template.format(
                query=parsed_cypher,
                response=str(cleaned),
            )

        return [
            NodeWithScore(
                node=TextNode(
                    text=node_text,
                    metadata=(
                        {"query": parsed_cypher, "response": cleaned}
                        if self.include_raw_response_as_metadata
                        else {}
                    ),
                ),
                score=1.0,
            )
        ]


def build_retrievers(
    graph_store: Neo4jPropertyGraphStore,
    llm: LLM,
    embed_model: BaseEmbedding | None,
    config: RetrieverConfig,
    neo4j_supports_vector: bool = True,
) -> List[BaseRetriever]:
    """Instantiate the enabled retriever strategies.

    Parameters
    ----------
    graph_store:
        The Neo4j property graph store to query against.
    llm:
        LLM used by the synonym and Cypher retrievers.
    embed_model:
        Embedding model used by the vector retriever.
    config:
        Controls which retrievers are enabled and their parameters.

    Returns
    -------
    list[BaseRetriever]
        One retriever per enabled strategy, ready to be passed as
        ``sub_retrievers`` to a :class:`PropertyGraphIndex`.
    """
    retrievers: list[BaseRetriever] = []

    if RetrieverType.VECTOR in config.enabled:
        logger.info(
            "Enabling VectorContextRetriever (top_k=%d, path_depth=%d)",
            config.similarity_top_k,
            config.path_depth,
        )
        retrievers.append(
            VectorContextRetriever(
                graph_store=graph_store,
                embed_model=embed_model,
                similarity_top_k=config.similarity_top_k,
                path_depth=config.path_depth,
                include_text=config.include_text,
            )
        )

    if RetrieverType.SYNONYM in config.enabled:
        logger.info(
            "Enabling LLMSynonymRetriever (max_keywords=%d, path_depth=%d)",
            config.synonym_max_keywords,
            config.path_depth,
        )
        retrievers.append(
            LLMSynonymRetriever(
                graph_store=graph_store,
                llm=llm,
                max_keywords=config.synonym_max_keywords,
                path_depth=config.path_depth,
                include_text=config.include_text,
            )
        )

    if RetrieverType.CYPHER in config.enabled:
        logger.info("Enabling TextToCypherRetriever (with retry)")
        cypher_kwargs: dict = dict(graph_store=graph_store, llm=llm)
        if not neo4j_supports_vector:
            # Neo4j 3.x — use restricted prompt and validator
            cypher_kwargs["text_to_cypher_template"] = _NEO4J3_CYPHER_TEMPLATE
            cypher_kwargs["cypher_validator"] = _cypher_validator_neo4j3
        retrievers.append(_RetryTextToCypherRetriever(**cypher_kwargs))

    if not retrievers:
        raise ValueError(
            "No retrievers enabled. Set GRAPHRAG_RETRIEVERS to at least one of: "
            "vector, cypher, synonym"
        )

    logger.info("Built %d retriever(s): %s", len(retrievers), config.enabled)
    return retrievers
