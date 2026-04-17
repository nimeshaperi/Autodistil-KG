"""Graph RAG query engine for schema-agnostic Neo4j knowledge graphs.

Retrieval pipeline (three strategies, run in parallel):
  1. Keyword / substring matching with 2-hop neighbourhood expansion —
     works on any graph schema without requiring LlamaIndex's internal
     ``__Entity__`` / ``__Node__`` label conventions.
  2. Vector similarity (when node embeddings are present).
  3. Text-to-Cypher via LlamaIndex's ``TextToCypherRetriever`` — the LLM
     generates a Cypher query from the question and executes it directly.
     This is the primary deep-retrieval path and is the strategy that
     scales to multi-hop relational questions.

The synthesis step uses a biomedical expert prompt that encourages the
model to ground answers in the retrieved context while drawing on domain
knowledge to fill gaps — rather than enumerating what is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import GraphRAGConfig, load_config
from .graph_store import create_graph_store
from .local_embedding import create_local_embedding

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    """Remove chain-of-thought blocks emitted by reasoning models (Qwen3, DeepSeek-R1, etc.).

    Handles:
    * ``<think>...</think>`` XML blocks
    * Orphaned ``</think>`` — everything before it is a reasoning preamble
    * Plain-text ``Thinking:`` / ``**Thinking**:`` lines without XML tags
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    text = re.sub(r"(?i)^\*{0,2}thinking\*{0,2}\s*:.*?(\n\n|\Z)", "", text, flags=re.DOTALL)
    return text.strip()


# ── Prompt templates ─────────────────────────────────────────────────

_ANSWER_SYSTEM = """\
/no_think
You are a biomedical expert. Answer the user's question using the graph \
context provided as your primary evidence base. Synthesise the relationships \
and node properties into a thorough, well-structured answer that names \
specific entities and their connections.

Where the graph context establishes clear relationships, cite them explicitly. \
Where the context is sparse, draw on your biomedical knowledge to contextualise \
and extend the evidence—grounding any inferences in the graph data where \
possible. Do not enumerate missing data; instead focus on building a \
coherent, complete answer from what is available."""

_ANSWER_USER = """\
## Graph Context
{context}

## Question
{question}

## Answer"""

# Used as the text_to_cypher_template for TextToCypherRetriever.
# Variables: {schema}, {question}  (LlamaIndex PromptTemplate convention)
_CYPHER_GENERATION_TEMPLATE = """\
/no_think
Generate a Cypher query to retrieve relevant information from a Neo4j \
knowledge graph to help answer the following biomedical question.

IMPORTANT — use only Neo4j 3.x compatible Cypher:
- Allowed: MATCH, WHERE, RETURN, WITH, UNWIND, ORDER BY, LIMIT, \
OPTIONAL MATCH, UNION
- Forbidden: IF NOT EXISTS, CREATE OR REPLACE, CALL {{}} subqueries, \
EXISTS {{}} subqueries, SHOW INDEXES, vector functions, \
db.index.vector.queryNodes

Focus the query on retrieving nodes, their properties, and their \
relationships that are most relevant to the question.

Schema:
{schema}

Question: {question}

Respond with ONLY the Cypher query, no explanation or markdown fences."""

_EXTRACT_ENTITIES_PROMPT = """\
/no_think
Extract the key biomedical entities (genes, diseases, compounds, proteins, \
biological processes, etc.) from the following question. Return ONLY a \
comma-separated list of entity names, nothing else.

Question: {question}

Entities:"""

_SYNONYM_PROMPT = """\
/no_think
For each biomedical entity below, list up to 3 alternative names, gene \
identifiers, full gene names, or common aliases. Focus on database identifiers \
and full names (e.g. SHTN1 → Shootin-1, Shootin1; CALD1 → Caldesmon 1, \
caldesmon; LYZ → Lysozyme, LYZ1). Return ONLY a flat comma-separated list of \
all alternatives, nothing else.

Entities: {entities}

Alternatives:"""

_FULLTEXT_INDEX_NAME = "entityNameIndex"


@dataclass
class GraphRAGResponse:
    """Container for a Graph RAG query result."""

    answer: str
    source_nodes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphRAGEngine:
    """Schema-agnostic Graph RAG engine for external Neo4j knowledge graphs.

    Uses three complementary retrieval strategies:

    * **Keyword retrieval** — CONTAINS substring matching + 2-hop neighbourhood
      expansion, schema-agnostic and fast.
    * **Vector retrieval** — optional, requires node embeddings in the graph.
    * **Text-to-Cypher** — LlamaIndex ``TextToCypherRetriever`` with one
      syntax-error retry; generates a targeted Cypher query from the question
      and executes it directly against the graph.

    The keyword and vector paths are necessary because LlamaIndex's native
    ``LLMSynonymRetriever`` and ``VectorContextRetriever`` rely on its own
    ``__Entity__`` / ``__Node__`` label conventions which are absent in
    externally-imported graphs.  The Cypher path uses LlamaIndex's
    ``TextToCypherRetriever`` abstraction, which generates arbitrary Cypher
    and therefore works with any graph schema.
    """

    def __init__(self, config: Optional[GraphRAGConfig] = None) -> None:
        self._config = config or load_config()
        self._llm = None
        self._graph_store = None
        self._embed_fn = None
        self._has_vector_index = False
        self._has_fulltext_index = False
        self._cypher_retriever: Optional[_RetryTextToCypherRetriever] = None

    def initialise(self) -> None:
        """Connect to Neo4j, initialise LLM, and build retriever pipeline."""
        from llama_index.llms.openai_like import OpenAILike
        from llama_index.core.prompts import PromptTemplate
        from .retrievers import _RetryTextToCypherRetriever

        cfg = self._config

        logger.info("Initialising LLM: model=%s", cfg.llm.model)
        self._llm = OpenAILike(
            model=cfg.llm.model,
            api_base=cfg.llm.base_url or "https://api.openai.com/v1",
            api_key=cfg.llm.api_key or "none",
            is_chat_model=True,
            max_tokens=2048,
            # Disable chain-of-thought thinking for Qwen3 / reasoning models.
            # vLLM honours this via the extra_body field on the OpenAI-compatible
            # endpoint; non-reasoning models silently ignore it.
            model_kwargs={"extra_body": {"enable_thinking": False}},
        )

        self._graph_store, supports_vector = create_graph_store(cfg.neo4j)
        logger.info("Neo4j PropertyGraphStore connected (vector_support=%s)", supports_vector)

        if supports_vector:
            self._embed_fn = create_local_embedding()
            if self._embed_fn:
                try:
                    result = self._graph_store.structured_query(
                        "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS cnt LIMIT 1"
                    )
                    count = result[0].get("cnt", 0) if result else 0
                    if count > 0:
                        self._has_vector_index = True
                        logger.info("Vector index available (%d nodes with embeddings)", count)
                    else:
                        logger.info("No node embeddings found — vector retrieval disabled")
                except Exception:
                    logger.info("Could not check for embeddings — vector retrieval disabled")

        # LlamaIndex TextToCypherRetriever (Neo4j 3.x compatible, with retry)
        self._cypher_retriever = _RetryTextToCypherRetriever(
            graph_store=self._graph_store,
            llm=self._llm,
            text_to_cypher_template=PromptTemplate(_CYPHER_GENERATION_TEMPLATE),
        )

        try:
            schema = self._graph_store.get_schema_str()
            logger.info("Graph schema: %s", schema[:300])
        except Exception:
            pass

        self._ensure_fulltext_index()

        logger.info(
            "GraphRAGEngine ready (path_depth=%d)",
            cfg.retriever.path_depth,
        )

    # ── Retrieval strategies ─────────────────────────────────────────

    def _extract_entities(self, question: str) -> List[str]:
        """Use the LLM to extract biomedical entity names from the question."""
        import re
        prompt = _EXTRACT_ENTITIES_PROMPT.format(question=question)
        response = self._llm.complete(prompt)
        raw = _strip_thinking(response.text or "")
        if not raw:
            logger.warning("Entity extraction returned empty — falling back to heuristics")
            raw = ", ".join(re.findall(r'\*([^*]+)\*', question))
            if not raw:
                raw = ", ".join(w for w in question.split() if w[0].isupper() and len(w) > 2)
        entities = []
        for e in raw.split(","):
            e = e.strip().strip("*").strip("'").strip('"').strip("`").strip()
            if e and len(e) > 1:
                entities.append(e)
        logger.info("Extracted entities: %s", entities)
        return entities

    def _retrieve_by_keywords(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Substring node matching with configurable-depth neighbourhood expansion.

        Uses ``path_depth`` from config to switch between 2-hop and 3-hop
        traversal.  Node cap is raised to 10 per entity; neighbour caps are
        50 / 30 / 20 for hops 1-3 to surface cross-gene associations that the
        original 5 / 20 / 15 limits truncated.
        """
        if not entities:
            return []

        depth = self._config.retriever.path_depth
        if depth >= 3:
            # Intermediate WITH … LIMIT guards prevent the Cartesian-product
            # explosion that occurs when chaining OPTIONAL MATCHes on a highly
            # connected biomedical graph.  Without them, 5 seeds × 100 hop-1
            # neighbors × 50 hop-2 neighbors × 30 hop-3 neighbors = 750 000
            # intermediate rows before any collect() runs, easily hitting the
            # Neo4j transaction timeout.  The guards cap the row count at each
            # stage while still producing representative neighbour samples.
            cypher = (
                "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) "
                "WITH n LIMIT 5 "
                "OPTIONAL MATCH (n)-[r1]-(m) "
                "WITH n, r1, m LIMIT 150 "
                "OPTIONAL MATCH (m)-[r2]-(o) WHERE o <> n "
                "WITH n, r1, m, r2, o LIMIT 500 "
                "OPTIONAL MATCH (o)-[r3]-(p) WHERE p <> n AND p <> m "
                "WITH n, "
                "collect(DISTINCT {neighbor: m.name, labels: labels(m), "
                "rel: type(r1)})[0..50] AS hop1, "
                "collect(DISTINCT {neighbor: o.name, labels: labels(o), "
                "rel: type(r2)})[0..30] AS hop2, "
                "collect(DISTINCT {neighbor: p.name, labels: labels(p), "
                "rel: type(r3)})[0..20] AS hop3 "
                "RETURN n.name AS name, labels(n) AS labels, "
                "properties(n) AS props, hop1 + hop2 + hop3 AS neighbours"
            )
            hop_label = "3-hop"
        else:
            # 2-hop: guard after hop-1 to bound the cross-join into hop-2.
            cypher = (
                "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) "
                "WITH n LIMIT 10 "
                "OPTIONAL MATCH (n)-[r1]-(m) "
                "WITH n, r1, m LIMIT 500 "
                "OPTIONAL MATCH (m)-[r2]-(o) WHERE o <> n "
                "WITH n, "
                "collect(DISTINCT {neighbor: m.name, labels: labels(m), "
                "rel: type(r1)})[0..50] AS hop1, "
                "collect(DISTINCT {neighbor: o.name, labels: labels(o), "
                "rel: type(r2)})[0..30] AS hop2 "
                "RETURN n.name AS name, labels(n) AS labels, "
                "properties(n) AS props, hop1 + hop2 AS neighbours"
            )
            hop_label = "2-hop"

        results = []
        for entity in entities:
            try:
                logger.info("Keyword search (%s): '%s'", hop_label, entity)
                records = self._graph_store.structured_query(
                    cypher,
                    param_map={"name": entity},
                )
                if records:
                    for rec in records:
                        logger.info(
                            "  Found: [%s] %s (%d neighbours)",
                            ":".join(rec.get("labels", [])),
                            rec.get("name", "?"),
                            len(rec.get("neighbours") or []),
                        )
                        results.append(rec)
                else:
                    logger.info("  No matches for '%s'", entity)
            except Exception as e:
                logger.warning("Keyword retrieval failed for '%s': %s", entity, e)

        logger.info("Keyword retrieval: %d results for %d entities", len(results), len(entities))
        return results

    def _retrieve_by_vector(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity retrieval (requires node embeddings)."""
        if not self._has_vector_index or not self._embed_fn:
            return []
        try:
            embedding = self._embed_fn._get_query_embedding(question)
            records = self._graph_store.structured_query(
                "CALL db.index.vector.queryNodes('entity', $top_k, $embedding) "
                "YIELD node, score "
                "OPTIONAL MATCH (node)-[r]-(m) "
                "WITH node, score, collect(DISTINCT {neighbor: m.name, "
                "labels: labels(m), rel: type(r)})[0..15] AS neighbours "
                "RETURN node.name AS name, labels(node) AS labels, "
                "properties(node) AS props, score, neighbours "
                "ORDER BY score DESC",
                param_map={"top_k": top_k, "embedding": embedding},
            )
            logger.info("Vector retrieval: %d results", len(records) if records else 0)
            return records or []
        except Exception as e:
            logger.warning("Vector retrieval failed: %s", e)
            return []

    def _retrieve_by_cypher(self, question: str) -> List[str]:
        """LlamaIndex TextToCypherRetriever: NL → Cypher → execute.

        Returns a list of result text blocks (one per successful query execution).
        Uses ``_RetryTextToCypherRetriever`` from the retrievers module, which
        automatically retries with error feedback on Cypher syntax failures.
        """
        if self._cypher_retriever is None:
            return []
        try:
            from llama_index.core.schema import QueryBundle
            nodes_with_scores = self._cypher_retriever.retrieve_from_graph(
                QueryBundle(query_str=question)
            )
            texts = [nws.node.text for nws in nodes_with_scores if nws.node.text.strip()]
            logger.info("Cypher retrieval: %d result block(s)", len(texts))
            return texts
        except Exception as e:
            logger.warning("Cypher retriever failed: %s", e)
            return []

    # ── Full-text fuzzy & synonym retrieval ──────────────────────────

    def _ensure_fulltext_index(self) -> None:
        """Create a Lucene full-text index on node ``name`` properties if absent.

        Requires Neo4j 3.5+.  Silently disables itself on older servers or if
        the index procedures are unavailable.
        """
        try:
            existing = self._graph_store.structured_query(
                "CALL db.indexes() YIELD description RETURN description"
            )
            if any(
                _FULLTEXT_INDEX_NAME in str(row.get("description", ""))
                for row in existing
            ):
                logger.info("Full-text index '%s' already exists", _FULLTEXT_INDEX_NAME)
                self._has_fulltext_index = True
                return

            labels = list(
                self._graph_store.structured_schema.get("node_props", {}).keys()
            )
            if not labels:
                labels = ["Gene", "Disease", "Pathway", "Anatomy", "Compound"]

            self._graph_store.structured_query(
                f"CALL db.index.fulltext.createNodeIndex("
                f"$idx_name, $labels, ['name'])",
                param_map={"idx_name": _FULLTEXT_INDEX_NAME, "labels": labels},
            )
            logger.info(
                "Created full-text index '%s' on labels: %s",
                _FULLTEXT_INDEX_NAME,
                labels,
            )
            self._has_fulltext_index = True
        except Exception as exc:
            logger.warning(
                "Full-text index unavailable (Neo4j <3.5 or procedure missing): %s",
                exc,
            )
            self._has_fulltext_index = False

    def _retrieve_by_fulltext_fuzzy(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Lucene fuzzy search via the full-text node index.

        Uses edit-distance-1 fuzzy (``term~1``) for short gene symbols and
        edit-distance-2 (``term~``) for longer names, then expands 1 hop to
        capture immediate neighbours.  Deduplicates against previously seen
        names via the ``seen`` set inside ``_build_context``.
        """
        if not self._has_fulltext_index:
            return []

        results: List[Dict[str, Any]] = []
        seen_names: set = set()
        for entity in entities:
            try:
                # Short identifiers (≤6 chars, e.g. gene symbols) use tighter
                # edit distance to avoid noisy matches.
                fuzzy_suffix = "~1" if len(entity) <= 6 else "~"
                lucene_query = f"{entity}{fuzzy_suffix}"
                logger.info("Fuzzy search: '%s'", lucene_query)
                records = self._graph_store.structured_query(
                    f"CALL db.index.fulltext.queryNodes($idx, $q) YIELD node, score "
                    "WHERE score > 0.3 "
                    "WITH node, score ORDER BY score DESC LIMIT 5 "
                    "OPTIONAL MATCH (node)-[r1]-(m) "
                    "WITH node, score, "
                    "collect(DISTINCT {neighbor: m.name, labels: labels(m), "
                    "rel: type(r1)})[0..30] AS hop1 "
                    "RETURN node.name AS name, labels(node) AS labels, "
                    "properties(node) AS props, score, hop1 AS neighbours",
                    param_map={"idx": _FULLTEXT_INDEX_NAME, "q": lucene_query},
                )
                for rec in records or []:
                    name = rec.get("name", "")
                    if name and name not in seen_names:
                        seen_names.add(name)
                        results.append(rec)
                        logger.info(
                            "  Fuzzy match: '%s' → '%s' (score=%.3f)",
                            entity,
                            name,
                            rec.get("score", 0),
                        )
            except Exception as exc:
                logger.warning(
                    "Full-text fuzzy retrieval failed for '%s': %s", entity, exc
                )

        logger.info(
            "Fuzzy retrieval: %d results for %d entities", len(results), len(entities)
        )
        return results

    def _retrieve_by_synonyms(self, entities: List[str]) -> List[Dict[str, Any]]:
        """LLM-generated alias expansion followed by keyword retrieval.

        Replaces LlamaIndex's ``LLMSynonymRetriever`` which requires the graph
        to carry ``__Entity__`` / ``__Node__`` labels from LlamaIndex's own
        ingestion pipeline.  This implementation generates synonyms via the
        same LLM used for querying, then falls back to the schema-agnostic
        substring CONTAINS search.
        """
        if not entities:
            return []

        try:
            prompt = _SYNONYM_PROMPT.format(entities=", ".join(entities))
            response = self._llm.complete(prompt)
            raw = _strip_thinking(response.text or "")
            synonyms: List[str] = []
            entity_lower = {e.lower() for e in entities}
            for s in raw.split(","):
                s = s.strip().strip("*").strip("'").strip('"').strip("`").strip()
                if s and len(s) > 1 and s.lower() not in entity_lower:
                    synonyms.append(s)
            logger.info("Generated %d synonym(s): %s", len(synonyms), synonyms)
        except Exception as exc:
            logger.warning("Synonym generation failed: %s", exc)
            return []

        if not synonyms:
            return []

        return self._retrieve_by_keywords(synonyms)

    # ── Context assembly ─────────────────────────────────────────────

    def _build_context(
        self,
        keyword_results: List[Dict],
        vector_results: List[Dict],
        cypher_texts: Optional[List[str]] = None,
    ) -> str:
        seen: set = set()
        lines: List[str] = []

        def _format_node(rec: Dict) -> None:
            name = rec.get("name", "unknown")
            if name in seen:
                return
            seen.add(name)
            labels = rec.get("labels", [])
            props = rec.get("props", {})
            score = rec.get("score")
            label_str = ":".join(labels) if labels else "Node"
            header = f"[{label_str}] {name}"
            if score is not None:
                header += f" (similarity: {score:.3f})"
            lines.append(header)
            skip_keys = {"name", "embedding", "id", "source", "license", "url"}
            for k, v in props.items():
                if k.lower() not in skip_keys and v:
                    val = str(v)
                    if len(val) > 200:
                        val = val[:200] + "..."
                    lines.append(f"  {k}: {val}")
            for nb in (rec.get("neighbours") or []):
                if nb and nb.get("neighbor"):
                    nb_labels = ":".join(nb.get("labels", []))
                    rel = nb.get("rel", "RELATED_TO")
                    lines.append(f"  --[{rel}]--> [{nb_labels}] {nb['neighbor']}")
            lines.append("")

        for rec in vector_results:
            _format_node(rec)
        for rec in keyword_results:
            _format_node(rec)

        if cypher_texts:
            lines.append("## Cypher Query Results")
            for text in cypher_texts:
                lines.append(text)
                lines.append("")

        context = "\n".join(lines)
        if not context.strip():
            context = "No relevant information was found in the knowledge graph."

        # ~8 000 tokens ≈ 32 000 chars — well within modern LLM context windows
        max_chars = 32000
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... (truncated)"

        return context

    # ── Main query entry point ───────────────────────────────────────

    def query(self, question: str) -> GraphRAGResponse:
        """Run a Graph RAG query and return a structured response."""
        if self._llm is None:
            raise RuntimeError("Engine not initialised. Call initialise() first.")

        logger.info("Query: %s", question)

        entities = self._extract_entities(question)
        keyword_results = self._retrieve_by_keywords(entities)
        fuzzy_results = self._retrieve_by_fulltext_fuzzy(entities)
        synonym_results = self._retrieve_by_synonyms(entities)
        vector_results = self._retrieve_by_vector(question)
        cypher_texts = self._retrieve_by_cypher(question)

        all_graph_results = keyword_results + fuzzy_results + synonym_results
        context = self._build_context(all_graph_results, vector_results, cypher_texts)
        node_count = len({rec.get("name", "") for rec in all_graph_results + vector_results})
        logger.info(
            "Context built: %d chars, %d unique nodes, %d cypher block(s) "
            "[keyword=%d fuzzy=%d synonym=%d vector=%d]",
            len(context), node_count, len(cypher_texts),
            len(keyword_results), len(fuzzy_results),
            len(synonym_results), len(vector_results),
        )

        from llama_index.core.llms import ChatMessage, MessageRole
        user_content = _ANSWER_USER.format(context=context, question=question)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_ANSWER_SYSTEM),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]
        response = self._llm.chat(messages)
        answer = _strip_thinking(response.message.content or "")
        if not answer:
            logger.warning("LLM returned empty answer")
            answer = "The Graph RAG system retrieved context but the LLM returned an empty response."
        else:
            logger.info("Answer: %d chars — %.200s", len(answer), answer)

        source_nodes = []
        seen_src: set = set()
        for rec in vector_results + all_graph_results:
            name = rec.get("name", "unknown")
            if name in seen_src:
                continue
            seen_src.add(name)
            source_nodes.append({
                "text": f"[{':'.join(rec.get('labels', []))}] {name}",
                "score": rec.get("score"),
                "metadata": {"labels": rec.get("labels", [])},
            })

        return GraphRAGResponse(
            answer=answer,
            source_nodes=source_nodes,
            metadata={
                "llm_model": self._config.llm.model,
                "entities_extracted": entities,
                "keyword_results": len(keyword_results),
                "fuzzy_results": len(fuzzy_results),
                "synonym_results": len(synonym_results),
                "vector_results": len(vector_results),
                "cypher_blocks": len(cypher_texts),
                "total_nodes": node_count,
                "path_depth": self._config.retriever.path_depth,
            },
        )
