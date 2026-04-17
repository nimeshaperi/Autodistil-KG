"""
Pipeline and per-stage configuration.

Each stage has its own config model so it can be configured separately and
linked into a :class:`PipelineConfig`.  Graph traverser types are imported
lazily so the pipeline module can be used without ``neo4j`` when that stage
is not selected.
"""
from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional


class StageId(str, Enum):
    """Canonical identifiers for pipeline stages.

    Using a ``StrEnum`` eliminates magic strings -- compare stage names with
    ``StageId.GRAPH_TRAVERSER`` instead of raw ``"graph_traverser"`` strings.

    The string values are JSON-serializable and match the keys used in
    ``PipelineConfig``, the API request models, and the frontend constants.
    """
    GRAPH_TRAVERSER = "graph_traverser"
    CHATML_CONVERTER = "chatml_converter"
    FINETUNER = "finetuner"
    EVALUATOR = "evaluator"


# Keep a tuple for ordered iteration (stages run in this order)
STAGE_ORDER = (
    StageId.GRAPH_TRAVERSER,
    StageId.CHATML_CONVERTER,
    StageId.FINETUNER,
    StageId.EVALUATOR,
)


class GraphTraverserStageConfig(BaseModel):
    """Configuration for the Graph Traverser stage (context extraction + CHATML generation)."""
    agent_config: Optional[Any] = None  # GraphTraverserAgentConfig when set
    graph_db: Optional[Any] = None  # GraphDatabaseConfig
    llm_config: Optional[Any] = None  # LLMConfig
    state_storage: Optional[Any] = None  # StateStorageConfig
    traversal: Optional[Any] = None  # TraversalConfig
    dataset: Optional[Any] = None  # DatasetGenerationConfig
    output_path: Optional[str] = None

    def get_agent_config(self) -> Any:
        from ..graph_traverser.config import GraphTraverserAgentConfig
        if self.agent_config is not None:
            return self.agent_config
        if all([self.graph_db, self.llm_config, self.state_storage, self.traversal, self.dataset]):
            return GraphTraverserAgentConfig(
                graph_db=self.graph_db,
                llm=self.llm_config,
                state_storage=self.state_storage,
                traversal=self.traversal,
                dataset=self.dataset,
            )
        raise ValueError(
            "GraphTraverserStageConfig: provide either agent_config or "
            "graph_db, llm_config, state_storage, traversal, dataset"
        )


class ChatMLConverterStageConfig(BaseModel):
    """Configuration for the ChatML Converter stage (normalize / prepare format)."""
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    prepare_for_finetuning: bool = True
    chat_template: Optional[str] = None
    additional_params: Dict[str, Any] = {}


class FineTunerStageConfig(BaseModel):
    """Configuration for the FineTuner stage. Delegates to finetuner module config."""
    finetuner_config: Optional[Any] = None  # UnslothFineTunerConfig
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    output_dir: Optional[str] = None
    max_seq_length: int = 2048
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    additional_params: Dict[str, Any] = {}


class EvaluatorStageConfig(BaseModel):
    """Configuration for the Evaluator stage."""
    model_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    output_report_path: Optional[str] = None
    metrics: Optional[List[str]] = None
    evalg_mode: Literal["internal", "cli", "noop"] = "internal"
    # Inference settings for local models (finetuned + local base)
    max_new_tokens: int = 2048
    max_seq_length: int = 4096
    # Base model for comparison
    base_model_provider: Optional[str] = None
    base_model_name: Optional[str] = None
    base_model_api_key: Optional[str] = None
    base_model_base_url: Optional[str] = None
    # Graph RAG config
    graph_rag_config: Optional[Dict[str, Any]] = None
    # Comparison systems toggle (False = skip even if model/config is available)
    eval_distilled: Optional[bool] = None  # None → include if model_path exists
    eval_base: Optional[bool] = None       # None → include if base_model_provider is set
    eval_graph_rag: Optional[bool] = None  # None → include if graph_rag_config is set
    # LLM Judge config
    judge_provider: Optional[str] = None
    judge_model: Optional[str] = None
    judge_api_key: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_max_tokens: Optional[int] = None
    # vLLM serving — run base + LoRA on a single vLLM server
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.8
    vllm_max_model_len: Optional[int] = None
    # Limits
    max_eval_samples: Optional[int] = None
    additional_params: Dict[str, Any] = {}


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class PipelineConfig(BaseModel):
    """
    Full pipeline configuration with optional config per stage.
    Stages can be enabled/disabled and configured independently.
    """
    graph_traverser: Optional[GraphTraverserStageConfig] = None
    chatml_converter: Optional[ChatMLConverterStageConfig] = None
    finetuner: Optional[FineTunerStageConfig] = None
    evaluator: Optional[EvaluatorStageConfig] = None
    output_dir: Optional[str] = None
    run_stages: Optional[List[StageId]] = None
    log_level: LogLevel = "INFO"
    log_dir: Optional[str] = None
