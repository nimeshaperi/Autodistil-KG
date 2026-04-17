"""
Traversal strategy implementations for the Graph Traverser Agent.

Contains standalone functions for BFS, DFS, random, semantic, and reasoning
traversal strategies, as well as the parallel execution helpers that drive
multi-worker traversal.
"""
import logging
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Deque, TYPE_CHECKING

from .utils import short_id

if TYPE_CHECKING:
    from .graph_traverser_agent import GraphTraverserAgent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# BFS
# ------------------------------------------------------------------

def traverse_bfs(agent: "GraphTraverserAgent", start_nodes: List[str]) -> None:
    """Breadth-First Search traversal."""
    queue: Deque[tuple[str, int]] = deque()
    visited_set: set = set()

    for node_id in start_nodes:
        if node_id not in visited_set:
            queue.append((node_id, 0))
            visited_set.add(node_id)

    logger.info("BFS initialized: queue=%d nodes, workers=%d", len(queue), agent._num_workers)

    if agent._num_workers <= 1:
        while queue:
            if agent._should_stop():
                logger.info("Stopping: reached max_nodes=%d", agent.config.traversal.max_nodes)
                break

            node_id, depth = queue.popleft()

            if agent.config.traversal.max_depth and depth > agent.config.traversal.max_depth:
                continue

            agent.current_depth = depth
            agent._process_node(node_id)

            neighbors = agent.graph_db.get_neighbors(
                node_id,
                relationship_types=agent.config.traversal.relationship_types
            )
            new_count = 0
            for neighbor in neighbors:
                neighbor_id = neighbor["id"]
                if neighbor_id not in visited_set:
                    visited_set.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
                    new_count += 1

            logger.debug(
                "BFS: processed %s (depth=%d), +%d neighbors, queue=%d, processed=%d",
                short_id(node_id), depth, new_count, len(queue), agent.visited_count,
            )
    else:
        _traverse_parallel(agent, queue, visited_set, "BFS")


# ------------------------------------------------------------------
# DFS
# ------------------------------------------------------------------

def traverse_dfs(agent: "GraphTraverserAgent", start_nodes: List[str]) -> None:
    """Depth-First Search traversal."""
    stack: Deque[tuple[str, int]] = deque()
    visited_set: set = set()

    for node_id in reversed(start_nodes):
        if node_id not in visited_set:
            stack.append((node_id, 0))
            visited_set.add(node_id)

    logger.info("DFS initialized: stack=%d nodes, workers=%d", len(stack), agent._num_workers)

    if agent._num_workers <= 1:
        while stack:
            if agent._should_stop():
                logger.info("Stopping: reached max_nodes=%d", agent.config.traversal.max_nodes)
                break

            node_id, depth = stack.pop()

            if agent.config.traversal.max_depth and depth > agent.config.traversal.max_depth:
                continue

            agent.current_depth = depth
            agent._process_node(node_id)

            neighbors = agent.graph_db.get_neighbors(
                node_id,
                relationship_types=agent.config.traversal.relationship_types
            )
            new_count = 0
            for neighbor in neighbors:
                neighbor_id = neighbor["id"]
                if neighbor_id not in visited_set:
                    visited_set.add(neighbor_id)
                    stack.append((neighbor_id, depth + 1))
                    new_count += 1

            logger.debug(
                "DFS: processed %s (depth=%d), +%d neighbors, stack=%d, processed=%d",
                short_id(node_id), depth, new_count, len(stack), agent.visited_count,
            )
    else:
        _traverse_parallel(agent, stack, visited_set, "DFS")


# ------------------------------------------------------------------
# Random
# ------------------------------------------------------------------

def traverse_random(agent: "GraphTraverserAgent", start_nodes: List[str]) -> None:
    """Random traversal."""
    visited_set: set = set()
    unvisited: Deque[tuple[str, int]] = deque()
    for nid in start_nodes:
        unvisited.append((nid, 0))
    visited_set.update(start_nodes)

    logger.info("Random traversal initialized: %d nodes to process, workers=%d", len(unvisited), agent._num_workers)

    if agent._num_workers <= 1:
        while unvisited:
            if agent._should_stop():
                break

            # Shuffle by converting to list, shuffling, and putting back
            items = list(unvisited)
            random.shuffle(items)
            unvisited = deque(items)

            node_id, depth = unvisited.pop()

            agent.current_depth = depth
            agent._process_node(node_id)

            neighbors = agent.graph_db.get_neighbors(
                node_id,
                relationship_types=agent.config.traversal.relationship_types
            )
            for neighbor in neighbors:
                neighbor_id = neighbor["id"]
                if neighbor_id not in visited_set:
                    visited_set.add(neighbor_id)
                    unvisited.append((neighbor_id, depth + 1))

            logger.debug("Random: processed %s, remaining=%d", short_id(node_id), len(unvisited))
    else:
        _traverse_parallel(agent, unvisited, visited_set, "Random")


# ------------------------------------------------------------------
# Semantic
# ------------------------------------------------------------------

def traverse_semantic(agent: "GraphTraverserAgent", start_nodes: List[str]) -> None:
    """Semantic-aware traversal using LLM to decide which nodes to visit next."""
    from .prompts import format_semantic_selection_prompt
    from autodistil_kg.llm import LLMMessage

    visited_set: set = set()
    candidates = list(start_nodes)

    logger.info("Semantic traversal initialized: %d candidates", len(candidates))

    while candidates:
        if agent._should_stop():
            break

        if len(candidates) > 1:
            selected_node = _select_semantic_node(agent, candidates, visited_set)
        else:
            selected_node = candidates[0]

        candidates.remove(selected_node)
        visited_set.add(selected_node)

        agent._process_node(selected_node)

        neighbors = agent.graph_db.get_neighbors(
            selected_node,
            relationship_types=agent.config.traversal.relationship_types
        )
        for neighbor in neighbors:
            neighbor_id = neighbor["id"]
            if neighbor_id not in visited_set and neighbor_id not in candidates:
                candidates.append(neighbor_id)

        logger.debug("Semantic: processed %s, candidates=%d", short_id(selected_node), len(candidates))


def _select_semantic_node(
    agent: "GraphTraverserAgent",
    candidates: List[str],
    visited: set,
) -> str:
    """Use LLM to select the most semantically relevant node from candidates."""
    from .prompts import format_semantic_selection_prompt
    from autodistil_kg.llm import LLMMessage

    # Get information about candidate nodes
    candidate_info = []
    for node_id in candidates[:10]:  # Limit to 10 for LLM context
        node = agent.graph_db.get_node(node_id)
        if node:
            candidate_info.append({
                "id": node_id,
                "labels": node.get("labels", []),
                "properties": node.get("properties", {})
            })

    # Create prompt for LLM using versioned prompt
    prompt = format_semantic_selection_prompt(
        candidate_info, version="current", alignment=agent._alignment,
    )

    messages = [
        LLMMessage(role="user", content=prompt)
    ]

    try:
        response = agent.llm_client.generate(messages, temperature=0.3, max_tokens=10)
        # Parse response to get node index
        try:
            index = int(response.strip()) - 1
            if 0 <= index < len(candidate_info):
                return candidate_info[index]["id"]
        except ValueError:
            pass
    except Exception as e:
        logger.warning(f"Error in semantic selection: {e}")

    # Fallback to first candidate
    return candidates[0]


# ------------------------------------------------------------------
# Reasoning
# ------------------------------------------------------------------

def traverse_reasoning(agent: "GraphTraverserAgent", start_nodes: List[str]) -> None:
    """Deep reasoning traversal: for each node, extract a depth-N subgraph,
    reason through each path, synthesize understanding, then generate
    rich QA pairs for SLM distillation.
    """
    visited_set: set = set()
    queue: Deque[tuple[str, int]] = deque()

    for node_id in start_nodes:
        if node_id not in visited_set:
            queue.append((node_id, 0))
            visited_set.add(node_id)

    reasoning_depth = agent.config.traversal.reasoning_depth
    max_paths = agent.config.traversal.max_paths_per_node

    logger.info(
        "Reasoning traversal initialized: %d seed nodes, subgraph_depth=%d, max_paths=%d, workers=%d",
        len(queue), reasoning_depth, max_paths, agent._num_workers,
    )

    if agent._num_workers <= 1:
        # Single-worker path (original sequential behavior)
        while queue:
            if agent._should_stop():
                logger.info("Stopping: reached max_nodes=%d", agent.config.traversal.max_nodes)
                break

            node_id, depth = queue.popleft()

            if agent.config.traversal.max_depth and depth > agent.config.traversal.max_depth:
                continue

            agent.current_depth = depth
            agent._process_node_reasoning(node_id, reasoning_depth, max_paths)

            # After processing, add unvisited subgraph nodes to the queue
            subgraph = agent.graph_db.get_subgraph(
                node_id,
                depth=1,
                relationship_types=agent.config.traversal.relationship_types,
            )
            for nid in subgraph.get("nodes", {}):
                if nid not in visited_set:
                    visited_set.add(nid)
                    queue.append((nid, depth + 1))

            logger.debug(
                "Reasoning: processed %s (depth=%d), queue=%d, processed=%d",
                short_id(node_id), depth, len(queue), agent.visited_count,
            )
    else:
        _traverse_reasoning_parallel(
            agent, queue, visited_set, reasoning_depth, max_paths,
        )


def _traverse_reasoning_parallel(
    agent: "GraphTraverserAgent",
    queue: Deque[tuple[str, int]],
    visited_set: set,
    reasoning_depth: int,
    max_paths: int,
) -> None:
    """Multi-worker reasoning traversal using a thread pool."""

    def _worker(node_id: str, depth: int) -> tuple[str, int]:
        """Process a single node and return (node_id, depth) for neighbor expansion."""
        agent._process_node_reasoning(node_id, reasoning_depth, max_paths)
        return node_id, depth

    with ThreadPoolExecutor(
        max_workers=agent._num_workers,
        thread_name_prefix="traverser",
    ) as executor:
        while queue or False:  # outer loop keeps going as long as there's work
            if agent._should_stop():
                logger.info("Stopping: reached max_nodes=%d", agent.config.traversal.max_nodes)
                break

            # Collect a batch of nodes to process in parallel
            batch: list[tuple[str, int]] = []
            while queue and len(batch) < agent._num_workers:
                if agent._should_stop():
                    break
                node_id, depth = queue.popleft()
                if agent.config.traversal.max_depth and depth > agent.config.traversal.max_depth:
                    continue
                batch.append((node_id, depth))

            if not batch:
                break

            # Submit all nodes in the batch for parallel processing
            futures = {
                executor.submit(_worker, nid, d): (nid, d)
                for nid, d in batch
            }

            # Wait for workers, but check cancellation between completions
            for future in as_completed(futures):
                nid, depth = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(
                        "Worker error processing %s: %s", short_id(nid), e
                    )
                    continue

                if agent._should_stop():
                    # Cancel any remaining futures and break
                    for f in futures:
                        f.cancel()
                    logger.info("Cancellation requested, stopping after current workers finish")
                    break

                # Expand neighbors into the queue
                try:
                    subgraph = agent.graph_db.get_subgraph(
                        nid,
                        depth=1,
                        relationship_types=agent.config.traversal.relationship_types,
                    )
                    for neighbor_id in subgraph.get("nodes", {}):
                        if neighbor_id not in visited_set:
                            visited_set.add(neighbor_id)
                            queue.append((neighbor_id, depth + 1))
                except Exception as e:
                    logger.warning(
                        "Error expanding neighbors for %s: %s", short_id(nid), e
                    )

                logger.debug(
                    "Reasoning: processed %s (depth=%d), queue=%d, processed=%d",
                    short_id(nid), depth, len(queue), agent.visited_count,
                )

            # If the queue is empty after processing, we're done
            if not queue:
                break


# ------------------------------------------------------------------
# Generic parallel traversal (BFS / DFS / Random)
# ------------------------------------------------------------------

def _traverse_parallel(
    agent: "GraphTraverserAgent",
    queue: Deque[tuple[str, int]],
    visited_set: set,
    label: str,
) -> None:
    """Generic parallel traversal for BFS/DFS strategies.

    Pops batches of nodes from *queue*, processes them concurrently via
    a thread pool, and expands neighbours back into *queue*.
    """

    def _worker(node_id: str) -> str:
        agent._process_node(node_id)
        return node_id

    with ThreadPoolExecutor(
        max_workers=agent._num_workers,
        thread_name_prefix="traverser",
    ) as executor:
        while queue:
            if agent._should_stop():
                logger.info("Stopping: reached max_nodes=%d", agent.config.traversal.max_nodes)
                break

            batch: list[tuple[str, int]] = []
            while queue and len(batch) < agent._num_workers:
                if agent._should_stop():
                    break
                nid, depth = queue.popleft()
                if agent.config.traversal.max_depth and depth > agent.config.traversal.max_depth:
                    continue
                batch.append((nid, depth))

            if not batch:
                break

            futures = {
                executor.submit(_worker, nid): (nid, d)
                for nid, d in batch
            }

            for future in as_completed(futures):
                nid, depth = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error("Worker error processing %s: %s", short_id(nid), e)
                    continue

                try:
                    neighbors = agent.graph_db.get_neighbors(
                        nid,
                        relationship_types=agent.config.traversal.relationship_types,
                    )
                    for neighbor in neighbors:
                        neighbor_id = neighbor["id"]
                        if neighbor_id not in visited_set:
                            visited_set.add(neighbor_id)
                            queue.append((neighbor_id, depth + 1))
                except Exception as e:
                    logger.warning("Error expanding neighbors for %s: %s", short_id(nid), e)

                logger.debug(
                    "%s: processed %s (depth=%d), queue=%d, processed=%d",
                    label, short_id(nid), depth, len(queue), agent.visited_count,
                )
