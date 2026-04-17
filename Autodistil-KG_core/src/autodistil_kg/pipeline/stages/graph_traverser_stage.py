"""
Graph Traverser stage: context extraction + CHATML dataset generation.
Runnable standalone or as part of the pipeline.
"""
import logging
from pathlib import Path
from typing import Optional

from ..interfaces import Stage, StageResult, PipelineContext
from ..config import GraphTraverserStageConfig

logger = logging.getLogger(__name__)


class GraphTraverserStage(Stage):
    """Stage that runs the graph traverser and produces a CHATML dataset."""

    name = "graph_traverser"  # matches StageId.GRAPH_TRAVERSER

    def __init__(self, config: GraphTraverserStageConfig):
        self.config = config

    def run(self, context: PipelineContext) -> StageResult:
        from ...graph_traverser import GraphTraverser
        try:
            agent_config = self.config.get_agent_config()
            traverser = GraphTraverser(agent_config)
            # Pass cancellation token to the traverser agent
            if context.cancel_token and hasattr(traverser, 'agent'):
                traverser.agent._cancel_token = context.cancel_token
            dataset = traverser.traverse()

            context.chatml_dataset = dataset
            if self.config.output_path:
                path = Path(self.config.output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                dataset.save_jsonl(str(path))
                context.chatml_dataset_path = str(path)
                logger.info("Saved CHATML dataset to %s", path)

            # Set the eval dataset path if one was generated
            if traverser.eval_dataset_path:
                context.eval_dataset_path = traverser.eval_dataset_path
                logger.info("Eval dataset available at %s", traverser.eval_dataset_path)

            stats = traverser.get_statistics()
            logger.info(
                "Graph traverser stats: processed=%d, dataset=%d, Redis visited=%d",
                stats.get("visited_count", 0),
                stats.get("dataset_size", 0),
                stats.get("total_visited", 0),
            )
            return StageResult(
                success=True,
                output=dataset,
                metadata={"statistics": stats, "conversations_count": len(dataset)},
            )
        except Exception as e:
            logger.exception("Graph traverser stage failed")
            return StageResult(success=False, error=str(e))
