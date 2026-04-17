"""
Graph Traverser Agent - A semantic-aware agent that traverses a Knowledge Graph
and creates CHATML compliant datasets.

This module defines the main ``GraphTraverserAgent`` class which orchestrates
graph traversal, Q&A generation, quality scoring, and dataset management.
The heavy lifting is delegated to focused sub-modules:

- ``traversal_strategies`` -- BFS, DFS, random, semantic, reasoning traversals
- ``qa_generator`` -- LLM-based path reasoning, synthesis, and Q&A generation
- ``quality`` -- quality scoring, alignment helpers, reference text loading
- ``utils`` -- event emission, ID formatting, reasoning-trace stripping
"""
import logging
import random
import threading
from typing import List, Dict, Any, Optional
import time

# Import from modular interfaces
from .graph_db import GraphDatabase
from autodistil_kg.llm import LLMClient, LLMMessage
from .state_storage import StateStorage, NodeMetadata, NodeState
from .chatml import ChatMLDataset, ChatMLFormatter
from .config import (
    GraphTraverserAgentConfig,
    TraversalStrategy,
)
from .prompts import (
    format_node_context,
    format_path_description,
)

# Sub-modules
from .utils import (
    emit_traversal_event as _emit_traversal_event,
    short_id as _short_id,
)
from .traversal_strategies import (
    traverse_bfs,
    traverse_dfs,
    traverse_random,
    traverse_semantic,
    traverse_reasoning,
)
from .qa_generator import (
    reason_through_path,
    reason_through_paths_batch,
    synthesize_subgraph,
    synthesize_and_generate_qa,
    parse_combined_response,
    generate_reasoning_qa,
)
from .quality import (
    score_qa_quality,
    load_reference_texts,
    build_system_message,
    rephrase_eval_questions,
)

logger = logging.getLogger(__name__)


class GraphTraverserAgent:
    """
    Semantic-aware agent that traverses a Knowledge Graph and creates
    CHATML compliant datasets.
    """

    def __init__(
        self,
        graph_db: GraphDatabase,
        llm_client: LLMClient,
        state_storage: StateStorage,
        config: GraphTraverserAgentConfig
    ):
        """
        Initialize the Graph Traverser Agent.

        Args:
            graph_db: Graph database instance
            llm_client: LLM client instance
            state_storage: State storage instance
            config: Configuration object
        """
        self.graph_db = graph_db
        self.llm_client = llm_client
        self.state_storage = state_storage
        self.config = config

        self.dataset = ChatMLDataset()
        self.visited_count = 0
        self.current_depth = 0
        self._checkpoint_path = self.config.dataset.output_path
        self._last_checkpoint_idx = 0  # index into dataset.conversations already flushed
        self._lock = threading.Lock()  # Guards dataset, visited_count, checkpoint
        self._num_workers = max(1, self.config.traversal.num_workers)
        # Separate eval dataset: clean Q&A pairs with rephrased questions,
        # suitable for judge-based evaluation after fine-tuning.
        self._eval_samples: List[Dict[str, str]] = []
        self._eval_sample_ratio = self.config.dataset.eval_sample_ratio
        self._eval_rephrase = self.config.dataset.eval_rephrase
        # Incremental eval checkpoint: mirrors _last_checkpoint_idx for eval samples
        self._last_eval_checkpoint_idx = 0
        # Per-node manifest: rich metadata about each processed node
        self._node_manifest: List[Dict[str, Any]] = []
        self._last_manifest_idx = 0

        # --- Alignment ---
        self._alignment = self.config.dataset.alignment
        self._reference_texts = load_reference_texts(self._alignment.reference_texts_path)
        self._quality_filtered = 0  # counter for quality-gate rejections

    # ------------------------------------------------------------------
    # Alignment helpers (delegate to quality module)
    # ------------------------------------------------------------------

    def _load_reference_texts(self) -> str:
        """Load reference material from *alignment.reference_texts_path*."""
        return load_reference_texts(self._alignment.reference_texts_path)

    def _build_system_message(self) -> Optional[str]:
        """Build an enhanced system message incorporating alignment context."""
        return build_system_message(self.config.dataset.system_message, self._alignment)

    def _score_qa_quality(
        self,
        question: str,
        answer: str,
    ) -> Dict[str, float]:
        """Score a Q&A pair on relevance, groundedness, and completeness."""
        return score_qa_quality(
            self.llm_client, question, answer, self._reference_texts,
        )

    # ------------------------------------------------------------------
    # Main traversal entry point
    # ------------------------------------------------------------------

    def traverse(self) -> ChatMLDataset:
        """
        Start the graph traversal and dataset generation process.

        Returns:
            ChatMLDataset containing all generated conversations
        """
        logger.info("Starting graph traversal...")

        # Resume from checkpoint if available
        self._load_checkpoint()

        # Connect to all services
        self._connect_services()

        try:
            # Get starting nodes
            start_nodes = self._get_start_nodes()

            if not start_nodes:
                logger.warning("No starting nodes found. Cannot traverse.")
                return self.dataset

            logger.info(
                "Starting traversal from %d nodes (strategy=%s, max_nodes=%s, max_depth=%s)",
                len(start_nodes),
                self.config.traversal.strategy.value,
                self.config.traversal.max_nodes,
                self.config.traversal.max_depth,
            )
            _emit_traversal_event(
                "traversal_start",
                strategy=self.config.traversal.strategy.value,
                seed_nodes=len(start_nodes),
                max_nodes=self.config.traversal.max_nodes,
                max_depth=self.config.traversal.max_depth,
                num_workers=self._num_workers,
            )

            # Traverse based on strategy -- delegate to traversal_strategies module
            if self.config.traversal.strategy == TraversalStrategy.BFS:
                traverse_bfs(self, start_nodes)
            elif self.config.traversal.strategy == TraversalStrategy.DFS:
                traverse_dfs(self, start_nodes)
            elif self.config.traversal.strategy == TraversalStrategy.RANDOM:
                traverse_random(self, start_nodes)
            elif self.config.traversal.strategy == TraversalStrategy.SEMANTIC:
                traverse_semantic(self, start_nodes)
            elif self.config.traversal.strategy == TraversalStrategy.REASONING:
                traverse_reasoning(self, start_nodes)
            else:
                raise ValueError(f"Unknown traversal strategy: {self.config.traversal.strategy}")

            logger.info(f"Traversal complete. Generated {len(self.dataset)} conversations.")
            alignment_summary: Dict[str, Any] = {}
            if self._alignment.quality_filter:
                alignment_summary["quality_filtered"] = self._quality_filtered
                alignment_summary["quality_threshold"] = self._alignment.quality_threshold
            if self._alignment.domain_focus:
                alignment_summary["domain_focus"] = self._alignment.domain_focus
            if self._alignment.reference_texts_path:
                alignment_summary["has_reference_texts"] = bool(self._reference_texts)

            _emit_traversal_event(
                "traversal_complete",
                visited=self.visited_count,
                dataset_size=len(self.dataset),
                strategy=self.config.traversal.strategy.value,
                **({"alignment": alignment_summary} if alignment_summary else {}),
            )

            # Save dataset if output path is specified
            if self.config.dataset.output_path:
                self._save_dataset()

            return self.dataset

        finally:
            self._disconnect_services()

    # ------------------------------------------------------------------
    # Service management
    # ------------------------------------------------------------------

    def _connect_services(self) -> None:
        """Connect to all required services."""
        logger.info("Connecting to Neo4j...")
        if hasattr(self.graph_db, 'connect'):
            self.graph_db.connect()
        logger.info("Connecting to Redis state storage...")
        if hasattr(self.state_storage, 'connect'):
            self.state_storage.connect()
        logger.info("Services connected")

    def _disconnect_services(self) -> None:
        """Disconnect from all required services."""
        logger.debug("Disconnecting services...")
        if hasattr(self.graph_db, 'disconnect'):
            self.graph_db.disconnect()
        if hasattr(self.state_storage, 'disconnect'):
            self.state_storage.disconnect()

    def _get_start_nodes(self) -> List[str]:
        """Get starting nodes for traversal."""
        if self.config.traversal.seed_node_ids:
            return self.config.traversal.seed_node_ids

        # Query for nodes matching criteria
        query = "MATCH (n)"
        conditions = []
        params = {}

        if self.config.traversal.node_labels:
            labels = ":".join(self.config.traversal.node_labels)
            query = f"MATCH (n:{labels})"

        # Use the graph_db's compatible ID expression
        id_expr = getattr(self.graph_db, '_node_id_expr', lambda v: f"toString(id({v}))")("n")
        query += f" RETURN {id_expr} as node_id LIMIT 100"

        try:
            results = self.graph_db.query(query, params)
            return [str(r["node_id"]) for r in results]
        except Exception as e:
            logger.error(f"Error getting start nodes: {e}")
            return []

    # ------------------------------------------------------------------
    # Node processing -- reasoning strategy
    # ------------------------------------------------------------------

    def _process_node_reasoning(
        self,
        node_id: str,
        reasoning_depth: int,
        max_paths: int,
    ) -> None:
        """
        Deep reasoning processing for a single node:
        1. Query depth-N subgraph
        2. Reason through each path
        3. Synthesize all path reasonings
        4. Generate distillation-ready QA pair
        """
        existing_state = self.state_storage.get_node_state(node_id)
        if existing_state and existing_state.state == NodeState.VISITED:
            logger.debug("Node %s already visited, skipping", _short_id(node_id))
            return

        logger.info(
            "[%d/%s] Deep reasoning on node %s (depth=%d, subgraph_depth=%d)...",
            self.visited_count + 1,
            self.config.traversal.max_nodes or "\u221e",
            _short_id(node_id),
            self.current_depth,
            reasoning_depth,
        )
        _emit_traversal_event(
            "node_start",
            node_id=_short_id(node_id),
            depth=self.current_depth,
            visited=self.visited_count + 1,
            total=self.config.traversal.max_nodes or 0,
            step="querying_subgraph",
            reasoning_depth=reasoning_depth,
        )

        self._mark_in_progress(node_id)

        # Step 1: Get the full subgraph around this node
        subgraph = self.graph_db.get_subgraph(
            node_id,
            depth=reasoning_depth,
            relationship_types=self.config.traversal.relationship_types,
        )

        center = subgraph["center"]
        paths = subgraph["paths"]
        nodes_map = subgraph["nodes"]
        edges = subgraph["edges"]

        if not center or (not center.get("labels") and not center.get("properties")):
            logger.warning("Node %s has no data, marking skipped", _short_id(node_id))
            self._mark_skipped(node_id)
            return

        logger.info(
            "  Subgraph: %d nodes, %d edges, %d paths",
            len(nodes_map), len(edges), len(paths),
        )

        # Build lightweight node summaries with key properties for the UI
        def _pick_properties(props: Dict[str, Any], max_keys: int = 6) -> Dict[str, str]:
            """Select the most useful properties and stringify values."""
            if not props:
                return {}
            # Prioritise common meaningful keys
            priority = ["name", "title", "label", "description", "type", "category", "id"]
            picked: Dict[str, str] = {}
            for key in priority:
                if key in props and len(picked) < max_keys:
                    picked[key] = str(props[key])[:120]
            for key, val in props.items():
                if key not in picked and len(picked) < max_keys:
                    picked[key] = str(val)[:120]
            return picked

        # Center node summary
        center_summary = {
            "id": _short_id(str(node_id)),
            "labels": center.get("labels", []),
            "properties": _pick_properties(center.get("properties", {})),
        }
        subgraph_nodes = [
            {
                "id": _short_id(str(nid)),
                "labels": ndata.get("labels", []),
                "properties": _pick_properties(ndata.get("properties", {})),
            }
            for nid, ndata in nodes_map.items()
            if str(nid) != str(node_id)
        ]
        subgraph_edges = [
            {
                "source": _short_id(str(e.get("source_id", ""))),
                "target": _short_id(str(e.get("target_id", ""))),
                "type": e.get("type", ""),
            }
            for e in edges
        ]

        _emit_traversal_event(
            "subgraph_loaded",
            node_id=_short_id(node_id),
            labels=center.get("labels", []),
            node_count=len(nodes_map),
            edge_count=len(edges),
            path_count=len(paths),
            visited=self.visited_count + 1,
            total=self.config.traversal.max_nodes or 0,
            center=center_summary,
            nodes=subgraph_nodes,
            edges=subgraph_edges,
        )

        # Deduplicate paths by their string representation to avoid redundant reasoning
        unique_paths = []
        seen_path_keys = set()
        for path in paths:
            key = format_path_description(path)
            if key not in seen_path_keys:
                seen_path_keys.add(key)
                unique_paths.append(path)

        # Limit paths
        if len(unique_paths) > max_paths:
            # Prefer longer paths (more multi-hop reasoning potential)
            unique_paths.sort(key=lambda p: len(p), reverse=True)
            unique_paths = unique_paths[:max_paths]

        logger.info("  Reasoning over %d unique paths...", len(unique_paths))

        # Step 2: Reason through paths in batches
        batch_size = self.config.traversal.path_batch_size
        path_analyses = []
        for batch_start in range(0, len(unique_paths), batch_size):
            if self._should_stop():
                logger.info("  Cancellation requested, aborting path reasoning for %s", _short_id(node_id))
                return
            batch = unique_paths[batch_start:batch_start + batch_size]
            batch_end = batch_start + len(batch)
            logger.info(
                "  Batch %d-%d/%d (%d paths)",
                batch_start + 1, batch_end, len(unique_paths), len(batch),
            )
            _emit_traversal_event(
                "path_reasoning",
                node_id=_short_id(node_id),
                path_index=batch_start + 1,
                total_paths=len(unique_paths),
                batch_size=len(batch),
                step="llm_reasoning",
            )
            if len(batch) == 1:
                analysis = self._reason_through_path(center, batch[0])
                if analysis:
                    path_analyses.append(analysis)
            else:
                analyses = self._reason_through_paths_batch(center, batch)
                path_analyses.extend(analyses)


        if not path_analyses:
            logger.warning("  No path analyses produced for %s, skipping", _short_id(node_id))
            self._mark_skipped(node_id)
            return

        # Step 3: Synthesize all path reasonings
        if self._should_stop():
            logger.info("  Cancellation requested, skipping synthesis for %s", _short_id(node_id))
            return
        logger.info("  Synthesizing %d path analyses...", len(path_analyses))
        _emit_traversal_event(
            "synthesis",
            node_id=_short_id(node_id),
            path_analyses_count=len(path_analyses),
            step="llm_synthesizing",
        )
        synthesis = self._synthesize_subgraph(
            center, path_analyses, len(nodes_map), len(edges),
        )

        # Guard: skip dataset entry if synthesis failed (e.g. LLM timeout)
        if synthesis.startswith("Error during synthesis:"):
            logger.warning(
                "Skipping node %s -- synthesis failed: %s",
                _short_id(node_id), synthesis,
            )
            with self._lock:
                try:
                    self.state_storage.mark_visited(node_id, metadata={"processed": False, "strategy": "reasoning", "error": synthesis})
                except Exception:
                    pass
                self.visited_count += 1
                visited = self.visited_count
                ds_size = len(self.dataset)
            _emit_traversal_event(
                "node_done",
                node_id=_short_id(node_id),
                status="error",
                error=synthesis,
                visited=visited,
                total=self.config.traversal.max_nodes or 0,
                dataset_size=ds_size,
                labels=center.get("labels", []),
            )
            return

        # Step 4: Generate distillation QA pair
        if self._should_stop():
            logger.info("  Cancellation requested, skipping QA generation for %s", _short_id(node_id))
            return
        logger.info("  Generating QA pair for distillation...")
        _emit_traversal_event(
            "qa_generation",
            node_id=_short_id(node_id),
            step="llm_generating_qa",
        )
        qa_pair = self._generate_reasoning_qa(center, synthesis)

        # --- Quality gate ---
        if self._should_stop():
            logger.info("  Cancellation requested, skipping quality check for %s", _short_id(node_id))
            return
        if self._alignment.quality_filter and qa_pair["question"] and qa_pair["answer"]:
            scores = self._score_qa_quality(qa_pair["question"], qa_pair["answer"])
            logger.info(
                "  Quality scores: relevance=%.2f groundedness=%.2f completeness=%.2f avg=%.2f",
                scores["relevance"], scores["groundedness"], scores["completeness"], scores["avg"],
            )
            _emit_traversal_event(
                "quality_scored",
                node_id=_short_id(node_id),
                scores=scores,
                threshold=self._alignment.quality_threshold,
                passed=scores["avg"] >= self._alignment.quality_threshold,
            )
            if scores["avg"] < self._alignment.quality_threshold:
                logger.info(
                    "  QA pair filtered (avg=%.2f < threshold=%.2f) -- node will NOT count toward max_nodes",
                    scores["avg"], self._alignment.quality_threshold,
                )
                with self._lock:
                    self._quality_filtered += 1
                    try:
                        self.state_storage.mark_visited(
                            node_id,
                            metadata={"processed": False, "strategy": "reasoning", "quality_filtered": True, "quality_scores": scores},
                        )
                    except Exception:
                        pass
                    # NOTE: Do NOT increment visited_count here.  Filtered
                    # nodes must not eat into the max_nodes budget -- otherwise
                    # the traversal stops early and produces fewer training
                    # samples than the user requested.
                    ds_size = len(self.dataset)
                _emit_traversal_event(
                    "node_done",
                    node_id=_short_id(node_id),
                    status="quality_filtered",
                    visited=self.visited_count,
                    total=self.config.traversal.max_nodes or 0,
                    dataset_size=ds_size,
                    labels=center.get("labels", []),
                    quality_scores=scores,
                    quality_filtered_total=self._quality_filtered,
                )
                return

        # Build a Cypher-like representation of the subgraph for explainability
        def _cypher_repr() -> str:
            """Generate a human-readable Cypher representation of the subgraph."""
            lines: List[str] = []
            # Center node
            c_labels = ":".join(center.get("labels", []))
            c_props = center.get("properties", {})
            c_props_str = ", ".join(f'{k}: "{v}"' for k, v in list(c_props.items())[:5])
            lines.append(f"// Center node")
            lines.append(f"(:{c_labels} {{{c_props_str}}})")
            # Relationships
            lines.append(f"// {len(edges)} relationships")
            for e in edges[:20]:  # Cap for readability
                src_node = nodes_map.get(e.get("source_id", ""), {})
                tgt_node = nodes_map.get(e.get("target_id", ""), {})
                src_label = ":".join(src_node.get("labels", ["?"]))
                tgt_label = ":".join(tgt_node.get("labels", ["?"]))
                rel_type = e.get("type", "RELATED_TO")
                lines.append(f"(:{src_label})-[:{rel_type}]->(:{tgt_label})")
            if len(edges) > 20:
                lines.append(f"// ... and {len(edges) - 20} more relationships")
            return "\n".join(lines)

        subgraph_cypher = _cypher_repr()

        # Create CHATML conversations
        metadata = {
            "node_id": node_id,
            "labels": center.get("labels", []),
            "depth": self.current_depth,
            "subgraph_nodes": len(nodes_map),
            "subgraph_edges": len(edges),
            "paths_analyzed": len(path_analyses),
            "strategy": "reasoning",
            "timestamp": time.time(),
            "subgraph_cypher": subgraph_cypher,
        }

        # Use alignment-enhanced system message for training data
        system_msg = self._build_system_message()

        # Create the main distillation conversation from the QA pair
        conversation = ChatMLFormatter.create_conversation_pair(
            prompt=qa_pair["question"],
            response=qa_pair["answer"],
            system_message=system_msg,
            metadata=metadata if self.config.dataset.include_metadata else None,
        )

        # Also add the synthesis as a second training example (teach the model the deep reasoning)
        entity_name = center.get('properties', {}).get('name', center.get('labels', ['this entity'])[0] if center.get('labels') else 'this entity')
        synthesis_prompt = (
            f"Explain everything you know about {entity_name}, "
            f"including how it relates to other concepts in this domain, "
            f"key mechanisms, associations, and any important implications."
        )
        synthesis_conv = ChatMLFormatter.create_conversation_pair(
            prompt=synthesis_prompt,
            response=synthesis,
            system_message=system_msg,
            metadata={**metadata, "type": "synthesis"} if self.config.dataset.include_metadata else None,
        )

        # Add to dataset (thread-safe)
        import random as _random
        with self._lock:
            self.dataset.add_conversation(conversation)
            self.dataset.add_conversation(synthesis_conv)

            # Collect eval samples: same knowledge, rephrased questions.
            # The QA pair is always in the training data; the eval dataset
            # gets a rephrased version of the question so it tests
            # comprehension rather than memorisation.
            if (qa_pair["question"] and qa_pair["answer"]
                    and _random.random() < self._eval_sample_ratio):
                self._eval_samples.append({
                    "question": qa_pair["question"],
                    "reference": qa_pair["answer"],
                })

            try:
                self.state_storage.mark_visited(node_id, metadata={"processed": True, "strategy": "reasoning"})
            except Exception as e:
                logger.warning("Failed to persist visited state for %s: %s", _short_id(node_id), e)
            self.visited_count += 1

            # Record rich per-node metadata in the manifest
            self._node_manifest.append({
                "node_id": node_id,
                "short_id": _short_id(node_id),
                "labels": center.get("labels", []),
                "center_properties": _pick_properties(center.get("properties", {})),
                "subgraph": {
                    "node_count": len(nodes_map),
                    "edge_count": len(edges),
                    "path_count": len(paths),
                    "unique_paths_used": len(unique_paths),
                    "paths_analyzed": len(path_analyses),
                    "cypher_repr": subgraph_cypher,
                },
                "quality_scores": scores if (self._alignment.quality_filter and qa_pair["question"] and qa_pair["answer"]) else None,
                "qa_question": qa_pair["question"][:200] if qa_pair.get("question") else None,
                "conversations_added": 2,
                "dataset_index": len(self.dataset) - 2,
                "visited_count": self.visited_count,
                "depth": self.current_depth,
                "timestamp": time.time(),
            })

            self._checkpoint()

            visited = self.visited_count
            ds_size = len(self.dataset)

        logger.info(
            "  Node %s done -> 2 conversations (visited=%d)",
            _short_id(node_id), visited,
        )
        _emit_traversal_event(
            "node_done",
            node_id=_short_id(node_id),
            visited=visited,
            total=self.config.traversal.max_nodes or 0,
            dataset_size=ds_size,
            labels=center.get("labels", []),
            paths_analyzed=len(path_analyses),
            conversations=2,
        )

    # ------------------------------------------------------------------
    # QA generation helpers (delegate to qa_generator module)
    # ------------------------------------------------------------------

    def _reason_through_path(
        self,
        center_node: Dict[str, Any],
        path: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Use LLM to reason through a single path from the subgraph."""
        return reason_through_path(
            self.llm_client, center_node, path,
            self.config.dataset.system_message,
            self._alignment, self._reference_texts,
        )

    def _reason_through_paths_batch(
        self,
        center_node: Dict[str, Any],
        paths: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """Reason through multiple paths in a single LLM call."""
        return reason_through_paths_batch(
            self.llm_client, center_node, paths,
            self.config.dataset.system_message,
            self._alignment, self._reference_texts,
        )

    def _synthesize_subgraph(
        self,
        center_node: Dict[str, Any],
        path_analyses: List[str],
        num_nodes: int,
        num_edges: int,
    ) -> str:
        """Synthesize multiple path-level analyses into a comprehensive understanding."""
        return synthesize_subgraph(
            self.llm_client, center_node, path_analyses, num_nodes, num_edges,
            self.config.dataset.system_message,
            self._alignment, self._reference_texts,
        )

    def _synthesize_and_generate_qa(
        self,
        center_node: Dict[str, Any],
        path_analyses: List[str],
        num_nodes: int,
        num_edges: int,
    ) -> Dict[str, Any]:
        """Synthesize path analyses AND generate a QA pair in one LLM call."""
        return synthesize_and_generate_qa(
            self.llm_client, center_node, path_analyses, num_nodes, num_edges,
            self.config.dataset.system_message,
            self._alignment, self._reference_texts,
        )

    def _parse_combined_response(
        self,
        center_node: Dict[str, Any],
        response: str,
    ) -> Dict[str, Any]:
        """Parse the combined synthesis+QA response into its two parts."""
        return parse_combined_response(center_node, response)

    def _generate_reasoning_qa(
        self,
        center_node: Dict[str, Any],
        synthesis: str,
    ) -> Dict[str, str]:
        """Generate a high-quality QA pair from the synthesis."""
        return generate_reasoning_qa(
            self.llm_client, center_node, synthesis,
            self.config.dataset.system_message,
            self._alignment, self._reference_texts,
        )

    # ------------------------------------------------------------------
    # Node processing -- simple strategy (BFS/DFS/Random/Semantic)
    # ------------------------------------------------------------------

    def _process_node(self, node_id: str) -> None:
        """
        Process a single node: generate prompt/response pair and add to dataset.
        """
        existing_state = self.state_storage.get_node_state(node_id)
        if existing_state and existing_state.state == NodeState.VISITED:
            logger.debug("Node %s already visited (Redis), skipping", _short_id(node_id))
            return

        logger.info(
            "[%d/%s] Processing node %s (depth=%d)...",
            self.visited_count + 1,
            self.config.traversal.max_nodes or "\u221e",
            _short_id(node_id),
            self.current_depth,
        )
        _emit_traversal_event(
            "node_start",
            node_id=_short_id(node_id),
            depth=self.current_depth,
            visited=self.visited_count + 1,
            total=self.config.traversal.max_nodes or 0,
            step="querying",
        )

        self._mark_in_progress(node_id)

        node = self.graph_db.get_node(node_id)
        if not node:
            logger.warning("Node %s not found in graph, marking skipped", _short_id(node_id))
            self._mark_skipped(node_id)
            return

        # Get neighbors for context
        neighbors = self.graph_db.get_neighbors(node_id, limit=5)

        # Emit neighbor data so the UI can render relationships
        neighbor_summaries = [
            {
                "id": _short_id(str(n["id"])),
                "labels": n.get("labels", []),
                "relationship_type": n.get("relationship_type", ""),
            }
            for n in neighbors
        ]
        _emit_traversal_event(
            "neighbors_loaded",
            node_id=_short_id(node_id),
            labels=node.get("labels", []),
            neighbors=neighbor_summaries,
        )

        # Generate prompt using seed prompts or default
        prompt = self._generate_prompt(node, neighbors)

        _emit_traversal_event(
            "node_start",
            node_id=_short_id(node_id),
            depth=self.current_depth,
            visited=self.visited_count + 1,
            total=self.config.traversal.max_nodes or 0,
            step="llm_generating",
            labels=node.get("labels", []),
        )
        response = self._generate_response(node, neighbors, prompt)

        # Create CHATML conversation
        metadata = {
            "node_id": node_id,
            "labels": node.get("labels", []),
            "depth": self.current_depth,
            "timestamp": time.time()
        }

        conversation = ChatMLFormatter.create_conversation_pair(
            prompt=prompt,
            response=response,
            system_message=self.config.dataset.system_message,
            metadata=metadata if self.config.dataset.include_metadata else None
        )

        # Add to dataset (thread-safe)
        with self._lock:
            self.dataset.add_conversation(conversation)

            try:
                self.state_storage.mark_visited(node_id, metadata={"processed": True})
            except Exception as e:
                logger.warning("Failed to persist visited state for %s: %s", _short_id(node_id), e)
            self.visited_count += 1

            self._checkpoint()

            visited = self.visited_count
            ds_size = len(self.dataset)

        logger.info(
            "  Node %s done -> Redis (visited=%d, labels=%s)",
            _short_id(node_id),
            visited,
            node.get("labels", []),
        )
        _emit_traversal_event(
            "node_done",
            node_id=_short_id(node_id),
            visited=visited,
            total=self.config.traversal.max_nodes or 0,
            dataset_size=ds_size,
            labels=node.get("labels", []),
        )

    def _generate_prompt(
        self,
        node: Dict[str, Any],
        neighbors: List[Dict[str, Any]]
    ) -> str:
        """Generate a prompt for the node."""
        # Use seed prompts if available
        if self.config.dataset.seed_prompts:
            template = random.choice(self.config.dataset.seed_prompts)
            try:
                return template.format(
                    labels=node.get("labels", []),
                    properties=node.get("properties", {}),
                    neighbors=neighbors
                )
            except KeyError:
                pass

        # Use custom template if provided
        if self.config.dataset.prompt_template:
            try:
                return self.config.dataset.prompt_template.format(
                    labels=node.get("labels", []),
                    properties=node.get("properties", {}),
                    neighbors=neighbors
                )
            except KeyError:
                pass

        # Default prompt
        return ChatMLFormatter.format_node_prompt(node)

    def _generate_response(
        self,
        node: Dict[str, Any],
        neighbors: List[Dict[str, Any]],
        prompt: str
    ) -> str:
        """Generate a response using the LLM."""
        messages = []

        if self.config.dataset.system_message:
            messages.append(LLMMessage(role="system", content=self.config.dataset.system_message))

        # Add context about the node using versioned template
        context = format_node_context(
            labels=node.get('labels', []),
            properties=node.get('properties', {}),
            neighbors_count=len(neighbors) if neighbors else 0,
            version="current"
        )

        messages.append(LLMMessage(role="user", content=f"{context}\n\n{prompt}"))

        try:
            response = self.llm_client.generate(
                messages,
                temperature=0.7,
                max_tokens=500
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response for node: {e}"

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------

    def _mark_in_progress(self, node_id: str) -> None:
        """Persist node to Redis as in-progress before expensive LLM call."""
        try:
            existing = self.state_storage.get_node_state(node_id)
            metadata = NodeMetadata(
                node_id=node_id,
                state=NodeState.IN_PROGRESS,
                visit_count=existing.visit_count if existing else 0,
                last_visited=time.time(),
                metadata={"processing": True},
            )
            self.state_storage.set_node_state(node_id, metadata)
        except Exception as e:
            logger.debug(f"Could not persist in-progress state for {node_id}: {e}")

    def _mark_skipped(self, node_id: str) -> None:
        """Persist node as skipped (e.g. not found) to Redis."""
        try:
            existing = self.state_storage.get_node_state(node_id)
            metadata = NodeMetadata(
                node_id=node_id,
                state=NodeState.SKIPPED,
                visit_count=existing.visit_count if existing else 0,
                last_visited=time.time(),
                metadata={"reason": "not_found"},
            )
            self.state_storage.set_node_state(node_id, metadata)
        except Exception as e:
            logger.debug(f"Could not persist skipped state for {node_id}: {e}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Flush new conversations and eval samples to disk since the last checkpoint.

        This ensures that work is not lost if the process crashes mid-traversal.
        Uses append mode so each call only writes new data.
        """
        if not self._checkpoint_path:
            return
        new_convs = self.dataset.conversations[self._last_checkpoint_idx:]
        if new_convs:
            self.dataset.append_jsonl(self._checkpoint_path, new_convs)
            self._last_checkpoint_idx = len(self.dataset.conversations)
            logger.debug(
                "Checkpoint: appended %d conversations to %s (total %d)",
                len(new_convs), self._checkpoint_path, self._last_checkpoint_idx,
            )
        # Also flush eval samples incrementally
        new_eval = self._eval_samples[self._last_eval_checkpoint_idx:]
        if new_eval:
            self._append_eval_checkpoint(new_eval)
            self._last_eval_checkpoint_idx = len(self._eval_samples)
        # Flush node manifest incrementally
        new_manifest = self._node_manifest[self._last_manifest_idx:]
        if new_manifest:
            self._append_manifest_checkpoint(new_manifest)
            self._last_manifest_idx = len(self._node_manifest)

    def _eval_checkpoint_path(self) -> "Optional[str]":
        """Derive eval checkpoint path from the main checkpoint path."""
        if not self._checkpoint_path:
            return None
        from pathlib import Path as _Path
        p = _Path(self._checkpoint_path)
        return str(p.parent / f"{p.stem}_eval{p.suffix}")

    def _append_eval_checkpoint(self, samples: List[Dict[str, str]]) -> None:
        """Append eval samples to the eval checkpoint file."""
        import json as _json
        eval_path = self._eval_checkpoint_path()
        if not eval_path:
            return
        from pathlib import Path as _Path
        _Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "a", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "messages": [
                        {"role": "user", "content": sample["question"]},
                        {"role": "assistant", "content": sample["reference"]},
                    ]
                }
                f.write(_json.dumps(record, ensure_ascii=False) + "\n")

    def _manifest_checkpoint_path(self) -> "Optional[str]":
        """Derive manifest checkpoint path from the main checkpoint path."""
        if not self._checkpoint_path:
            return None
        from pathlib import Path as _Path
        p = _Path(self._checkpoint_path)
        return str(p.parent / f"{p.stem}_manifest{p.suffix}")

    def _append_manifest_checkpoint(self, entries: List[Dict[str, Any]]) -> None:
        """Append node manifest entries to the manifest checkpoint file."""
        import json as _json
        manifest_path = self._manifest_checkpoint_path()
        if not manifest_path:
            return
        from pathlib import Path as _Path
        _Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug(
            "Manifest: appended %d entries to %s (total %d)",
            len(entries), manifest_path, self._last_manifest_idx + len(entries),
        )

    def _load_checkpoint(self) -> None:
        """Resume from an existing checkpoint file if one exists.

        Loads previously saved conversations and eval samples so they are
        included in the final dataset. The Redis state storage already tracks
        visited nodes, so those nodes will be skipped during traversal
        automatically.
        """
        if not self._checkpoint_path:
            return
        from pathlib import Path
        path = Path(self._checkpoint_path)
        if not path.exists() or path.stat().st_size == 0:
            return
        try:
            prev = ChatMLDataset()
            prev.load_jsonl(str(path))
            self.dataset = prev
            self._last_checkpoint_idx = len(self.dataset.conversations)
            logger.info(
                "Resumed from checkpoint: loaded %d conversations from %s",
                len(self.dataset), self._checkpoint_path,
            )
        except Exception as e:
            logger.warning("Could not load checkpoint %s: %s -- starting fresh", self._checkpoint_path, e)

        # Restore eval samples from checkpoint
        import json as _json
        eval_path = self._eval_checkpoint_path()
        if eval_path:
            eval_p = Path(eval_path)
            if eval_p.exists() and eval_p.stat().st_size > 0:
                try:
                    restored = []
                    with open(eval_p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            record = _json.loads(line)
                            msgs = record.get("messages", [])
                            if len(msgs) >= 2:
                                restored.append({
                                    "question": msgs[0]["content"],
                                    "reference": msgs[1]["content"],
                                })
                    if restored:
                        self._eval_samples = restored
                        self._last_eval_checkpoint_idx = len(restored)
                        logger.info(
                            "Resumed eval checkpoint: loaded %d eval samples from %s",
                            len(restored), eval_path,
                        )
                except Exception as e:
                    logger.warning("Could not load eval checkpoint %s: %s -- starting fresh", eval_path, e)

        # Restore node manifest from checkpoint
        manifest_path = self._manifest_checkpoint_path()
        if manifest_path:
            manifest_p = Path(manifest_path)
            if manifest_p.exists() and manifest_p.stat().st_size > 0:
                try:
                    restored_manifest = []
                    with open(manifest_p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            restored_manifest.append(_json.loads(line))
                    if restored_manifest:
                        self._node_manifest = restored_manifest
                        self._last_manifest_idx = len(restored_manifest)
                        logger.info(
                            "Resumed manifest checkpoint: loaded %d node entries from %s",
                            len(restored_manifest), manifest_path,
                        )
                except Exception as e:
                    logger.warning("Could not load manifest checkpoint %s: %s -- starting fresh", manifest_path, e)

    # ------------------------------------------------------------------
    # Stopping logic
    # ------------------------------------------------------------------

    def _should_stop(self) -> bool:
        """Check if traversal should stop (max nodes reached or cancelled)."""
        token = getattr(self, '_cancel_token', None)
        if token and token.is_cancelled:
            return True
        if self.config.traversal.max_nodes:
            return self.visited_count >= self.config.traversal.max_nodes
        return False

    # ------------------------------------------------------------------
    # Eval rephrasing (delegate to quality module)
    # ------------------------------------------------------------------

    def _rephrase_eval_questions(
        self, samples: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Rephrase eval questions using the LLM so they test comprehension."""
        return rephrase_eval_questions(self.llm_client, samples)

    # ------------------------------------------------------------------
    # Dataset saving
    # ------------------------------------------------------------------

    def _save_dataset(self) -> None:
        """Save the final dataset to file.

        If incremental checkpointing was active and all conversations have
        already been flushed, this rewrites the file to ensure a clean
        output (no duplicate lines from appending to a resumed checkpoint).

        Also writes a separate eval dataset (clean Q&A pairs with rephrased
        questions) alongside the training data for judge-based evaluation.
        """
        import json as _json
        from pathlib import Path as _Path

        path = self.config.dataset.output_path
        if self.config.dataset.output_format == "jsonl":
            self.dataset.save_jsonl(path)
        else:
            self.dataset.save_json(path)
        logger.info("Dataset saved: %s (%d conversations)", path, len(self.dataset))

        # Write eval dataset as a sibling file (e.g. output.jsonl -> output_eval.jsonl)
        if self._eval_samples:
            # Rephrase questions so eval tests comprehension, not memorisation
            if self._eval_rephrase:
                self._eval_samples = self._rephrase_eval_questions(self._eval_samples)

            p = _Path(path)
            eval_path = str(p.parent / f"{p.stem}_eval{p.suffix}")
            with open(eval_path, "w", encoding="utf-8") as f:
                for sample in self._eval_samples:
                    # Format as CHATML so the evaluator's _extract_eval_sample works
                    record = {
                        "messages": [
                            {"role": "user", "content": sample["question"]},
                            {"role": "assistant", "content": sample["reference"]},
                        ]
                    }
                    f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                "Eval dataset saved: %s (%d samples, rephrased=%s)",
                eval_path, len(self._eval_samples), self._eval_rephrase,
            )
