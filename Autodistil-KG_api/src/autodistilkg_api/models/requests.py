"""Strictly typed Pydantic request models for the KG Pipeline API.

Every model in this file corresponds to a JSON request body accepted by one or
more API endpoints.  Field descriptions and constraints are surfaced
automatically in the OpenAPI / Swagger documentation.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# Literal union types
# ---------------------------------------------------------------------------
StageId = Literal["graph_traverser", "chatml_converter", "finetuner", "evaluator"]
TraversalStrategy = Literal["bfs", "dfs", "random", "semantic", "reasoning"]
LLMProviderType = Literal["openai", "openai_compatible", "claude", "gemini", "ollama", "vllm"]
EvalMode = Literal["internal", "cli", "noop"]
RetrieverType = Literal["vector", "cypher", "synonym"]
OutputFormat = Literal["jsonl", "json"]

# ---------------------------------------------------------------------------
# Sub-config request models
# ---------------------------------------------------------------------------


class Neo4jConfigRequest(BaseModel):
    """Neo4j graph database connection settings."""
    uri: str = Field("bolt://localhost:7687", description="Bolt URI for the Neo4j instance.")
    username: str = Field("neo4j", description="Neo4j username.")
    password: str = Field("", description="Neo4j password.")
    database: Optional[str] = Field(None, description="Neo4j database name. Uses default if omitted.")


class RedisConfigRequest(BaseModel):
    """Redis state storage connection settings."""
    host: str = Field("localhost", description="Redis host.")
    port: int = Field(6379, ge=1, le=65535, description="Redis port.")
    db: int = Field(0, ge=0, le=15, description="Redis database number.")
    password: Optional[str] = Field(None, description="Redis password.")
    key_prefix: Optional[str] = Field(None, description="Key prefix for namespacing traversal state.")


class LLMConfigRequest(BaseModel):
    """LLM provider configuration. Fields vary by provider."""
    provider: LLMProviderType = Field(..., description="LLM provider to use.")
    api_key: Optional[str] = Field(None, description="API key (required for openai, claude).")
    model: Optional[str] = Field(None, description="Model name (e.g. 'gpt-4o', 'claude-sonnet-4-20250514').")
    base_url: Optional[str] = Field(None, description="Custom API base URL (for openai_compatible, vllm).")
    project_id: Optional[str] = Field(None, description="GCP project ID (Gemini only).")
    location: Optional[str] = Field(None, description="GCP region (Gemini only).")
    credentials_path: Optional[str] = Field(None, description="Path to GCP credentials JSON (Gemini only).")


class TraversalConfigRequest(BaseModel):
    """Graph traversal strategy and parameters."""
    strategy: TraversalStrategy = Field("bfs", description="Traversal algorithm. 'reasoning' enables deep multi-hop analysis.")
    max_nodes: int = Field(500, ge=1, le=10000, description="Maximum number of nodes to visit.")
    max_depth: int = Field(5, ge=1, le=20, description="Maximum traversal depth from seed nodes.")
    reasoning_depth: int = Field(2, ge=1, le=5, description="Hops to follow for reasoning strategy path analysis.")
    max_paths_per_node: int = Field(15, ge=1, le=100, description="Max relationship paths to analyse per node.")
    path_batch_size: int = Field(5, ge=1, le=50, description="Number of paths to reason about in a single LLM call.")
    num_workers: int = Field(1, ge=1, le=16, description="Parallel worker threads for node processing.")
    relationship_types: Optional[List[str]] = Field(None, description="Restrict traversal to these relationship types.")
    node_labels: Optional[List[str]] = Field(None, description="Restrict traversal to nodes with these labels.")
    seed_node_ids: Optional[List[str]] = Field(None, description="Start traversal from these specific node IDs.")


class AlignmentConfigRequest(BaseModel):
    """Align generated Q&A data with target domain, style, and quality."""
    domain_focus: Optional[str] = Field(None, description="Target domain for generated content (e.g. 'neuroscience').")
    domain_keywords: List[str] = Field(default_factory=list, description="Keywords that should appear in generated content.")
    style_guide: Optional[str] = Field(None, description="Prose style instructions for the LLM.")
    target_audience: Optional[str] = Field(None, description="Target audience level (e.g. 'graduate students').")
    max_answer_length: Optional[int] = Field(None, ge=10, description="Maximum answer length in tokens.")
    min_answer_length: Optional[int] = Field(None, ge=1, description="Minimum answer length in tokens.")
    quality_filter: bool = Field(False, description="Enable post-generation quality scoring gate.")
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum quality score to keep a generated pair.")
    reference_texts_path: Optional[str] = Field(None, description="Path to reference texts for alignment scoring.")


class DatasetConfigRequest(BaseModel):
    """Dataset generation settings."""
    seed_prompts: List[str] = Field(
        default=["What can you tell me about this node? Describe: {properties}"],
        description="Prompt templates for Q&A generation. Use {properties} as placeholder.",
    )
    system_message: Optional[str] = Field(None, description="System message prepended to every conversation.")
    prompt_template: Optional[str] = Field(None, description="Custom prompt template override.")
    include_metadata: bool = Field(True, description="Include source node metadata in CHATML output.")
    output_format: OutputFormat = Field("jsonl", description="Output file format.")
    output_path: Optional[str] = Field(None, description="Custom output file path.")


# ---------------------------------------------------------------------------
# Stage config request models
# ---------------------------------------------------------------------------


class GraphTraverserConfigRequest(BaseModel):
    """Configuration for the Graph Traverser stage."""
    output_path: Optional[str] = Field(None, description="Output path for the generated CHATML dataset.")
    traversal: TraversalConfigRequest = Field(default_factory=TraversalConfigRequest, description="Traversal strategy parameters.")
    dataset: DatasetConfigRequest = Field(default_factory=DatasetConfigRequest, description="Dataset generation settings.")
    alignment: Optional[AlignmentConfigRequest] = Field(None, description="Alignment and quality filtering settings.")
    neo4j: Optional[Neo4jConfigRequest] = Field(None, description="Neo4j connection (falls back to .env).")
    redis: Optional[RedisConfigRequest] = Field(None, description="Redis connection for state storage (falls back to .env).")
    llm: Optional[LLMConfigRequest] = Field(None, description="LLM provider config (falls back to .env).")
    llm_provider: Optional[str] = Field(None, description="Shorthand LLM provider name (alternative to full llm config).")


class ChatMLConverterConfigRequest(BaseModel):
    """Configuration for the ChatML Converter stage."""
    input_path: Optional[str] = Field(None, description="Input CHATML dataset path. Uses previous stage output if omitted.")
    output_path: Optional[str] = Field(None, description="Output path for the prepared fine-tuning dataset.")
    prepare_for_finetuning: bool = Field(True, description="Strip reasoning traces and format for fine-tuning.")
    chat_template: Optional[str] = Field(None, description="Chat template name for tokenizer (e.g. 'gemma').")


class FineTunerConfigRequest(BaseModel):
    """Configuration for the FineTuner stage (Unsloth-based)."""
    model_name: str = Field("unsloth/gemma-2-2b-it", description="HuggingFace model ID or path to fine-tune.")
    model_type: Optional[str] = Field(None, description="Model type hint (auto-inferred from model_name if omitted).")
    train_data_path: Optional[str] = Field(None, description="Training data JSONL path. Uses previous stage output if omitted.")
    eval_data_path: Optional[str] = Field(None, description="Eval data JSONL path for loss tracking.")
    output_dir: Optional[str] = Field(None, description="Directory to save the fine-tuned LoRA adapter.")
    max_seq_length: int = Field(2048, ge=128, le=32768, description="Maximum sequence length for tokenization.")
    num_train_epochs: int = Field(1, ge=1, le=100, description="Number of training epochs.")
    per_device_train_batch_size: int = Field(2, ge=1, le=64, description="Training batch size per device.")
    learning_rate: float = Field(2e-4, gt=0, le=1.0, description="Learning rate for the optimizer.")


class GraphRAGConfigRequest(BaseModel):
    """GraphRAG system configuration for evaluation comparison."""
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j Bolt URI.")
    neo4j_user: str = Field("neo4j", description="Neo4j username.")
    neo4j_password: str = Field("", description="Neo4j password.")
    neo4j_database: str = Field("neo4j", description="Neo4j database name.")
    llm_provider: Optional[LLMProviderType] = Field(None, description="LLM provider for GraphRAG retrieval.")
    llm_api_key: Optional[str] = Field(None, description="LLM API key.")
    llm_model: str = Field("gpt-4", description="LLM model name for GraphRAG.")
    llm_base_url: Optional[str] = Field(None, description="Custom LLM base URL.")
    embedding_api_key: Optional[str] = Field(None, description="Embedding model API key.")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model for vector retrieval.")
    embedding_base_url: Optional[str] = Field(None, description="Custom embedding base URL.")
    retrievers: List[RetrieverType] = Field(default=["vector", "cypher", "synonym"], description="Retriever types to combine.")
    num_agents: int = Field(1, ge=1, le=8, description="Number of parallel retrieval agents.")
    similarity_top_k: int = Field(5, ge=1, le=100, description="Number of top results from vector similarity search.")


class EvaluatorConfigRequest(BaseModel):
    """Configuration for the Evaluator stage."""
    model_path: Optional[str] = Field(None, description="Path to the fine-tuned model/adapter. Uses previous stage output if omitted.")
    eval_dataset_path: Optional[str] = Field(None, description="Eval dataset JSONL path.")
    output_report_path: Optional[str] = Field(None, description="Path to write the evaluation report JSON.")
    metrics: List[str] = Field(default=["answer_relevancy"], description="Evaluation metrics to compute.")
    evalg_mode: EvalMode = Field("internal", description="Evaluation execution mode.")
    max_new_tokens: int = Field(2048, ge=1, le=8192, description="Max tokens to generate per prediction.")
    max_seq_length: int = Field(4096, ge=128, le=32768, description="Max sequence length for model input.")
    base_model_provider: Optional[LLMProviderType] = Field(None, description="LLM provider for the base model comparison system.")
    base_model_name: Optional[str] = Field(None, description="Base model name/ID for comparison.")
    base_model_api_key: Optional[str] = Field(None, description="API key for the base model provider.")
    base_model_base_url: Optional[str] = Field(None, description="Custom base URL for the base model.")
    graph_rag_enabled: bool = Field(False, description="Include GraphRAG as a comparison system.")
    graph_rag_config: Optional[GraphRAGConfigRequest] = Field(None, description="GraphRAG system settings (when enabled).")
    judge_provider: Optional[LLMProviderType] = Field(None, description="LLM provider for judge-based evaluation.")
    judge_model: Optional[str] = Field(None, description="Judge model name.")
    judge_api_key: Optional[str] = Field(None, description="API key for the judge model.")
    judge_base_url: Optional[str] = Field(None, description="Custom base URL for the judge model.")
    use_vllm: bool = Field(False, description="Serve models via vLLM for faster inference.")
    vllm_gpu_memory_utilization: float = Field(0.9, ge=0.1, le=1.0, description="GPU memory fraction for vLLM serving.")
    vllm_max_model_len: Optional[int] = Field(None, ge=128, description="Max model length override for vLLM.")
    max_eval_samples: Optional[int] = Field(None, ge=1, description="Limit number of evaluation samples (for debugging).")


# ---------------------------------------------------------------------------
# Top-level pipeline request
# ---------------------------------------------------------------------------

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class PipelineRunRequest(BaseModel):
    """Top-level request to run a pipeline."""
    output_dir: Optional[str] = Field(None, description="Base output directory. Auto-generated per run if omitted.")
    run_stages: List[StageId] = Field(..., description="Stages to execute, in order.")
    log_level: LogLevel = Field("INFO", description="Logging verbosity for this run.")
    graph_traverser: Optional[GraphTraverserConfigRequest] = Field(None, description="Graph Traverser stage config (required if stage is selected).")
    chatml_converter: Optional[ChatMLConverterConfigRequest] = Field(None, description="ChatML Converter stage config.")
    finetuner: Optional[FineTunerConfigRequest] = Field(None, description="FineTuner stage config.")
    evaluator: Optional[EvaluatorConfigRequest] = Field(None, description="Evaluator stage config.")


# ---------------------------------------------------------------------------
# Inference request models
# ---------------------------------------------------------------------------


class InferenceLLMRequest(BaseModel):
    """Send a prompt to any supported LLM provider."""
    provider: LLMProviderType = Field(..., description="LLM provider to use.")
    model: Optional[str] = Field(None, description="Model name.")
    api_key: Optional[str] = Field(None, description="API key.")
    base_url: Optional[str] = Field(None, description="Custom base URL.")
    project_id: Optional[str] = Field(None, description="GCP project ID (Gemini).")
    location: Optional[str] = Field(None, description="GCP region (Gemini).")
    credentials_path: Optional[str] = Field(None, description="Path to GCP credentials JSON.")
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages (role + content dicts).")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate.")


class InferenceFinetunedRequest(BaseModel):
    """Run inference on a registered fine-tuned model."""
    model_id: str = Field(..., description="Registered model ID from the model registry.")
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages (role + content dicts).")
    max_new_tokens: int = Field(512, ge=1, le=8192, description="Maximum new tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")


class RegisterModelRequest(BaseModel):
    """Register a fine-tuned model in the model registry."""
    model_id: str = Field(..., description="Human-readable model ID (e.g. 'adhd-kg-v1').")
    model_path: str = Field(..., description="Absolute path to the LoRA adapter directory.")
    base_model: Optional[str] = Field(None, description="Base model name (e.g. 'unsloth/gemma-2-2b-it').")
    description: Optional[str] = Field(None, description="Brief description of the model.")


class UploadToHubRequest(BaseModel):
    """Upload a fine-tuned model to HuggingFace Hub."""
    model_path: str = Field(..., description="Local path to the LoRA adapter directory.")
    repo_id: str = Field(..., description="HuggingFace repo ID (e.g. 'username/model-name').")
    hf_token: str = Field(..., description="HuggingFace API token with write access.")
    private: bool = Field(True, description="Create a private repository.")
    base_model: Optional[str] = Field(None, description="Base model name (for model card). Auto-detected from adapter_config.json if omitted.")
    commit_message: Optional[str] = Field(None, description="Custom commit message.")
    merge_before_upload: bool = Field(False, description="Merge LoRA weights into base model before uploading (creates a larger standalone model).")


class InferenceGraphRAGRequest(BaseModel):
    """Query the knowledge graph via GraphRAG retrieval-augmented generation."""
    question: str = Field(..., description="Natural language question to answer.")
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j Bolt URI.")
    neo4j_user: str = Field("neo4j", description="Neo4j username.")
    neo4j_password: str = Field("password", description="Neo4j password.")
    neo4j_database: str = Field("neo4j", description="Neo4j database name.")
    llm_api_key: Optional[str] = Field(None, description="LLM API key.")
    llm_model: str = Field("gpt-4", description="LLM model for answer generation.")
    llm_base_url: Optional[str] = Field(None, description="Custom LLM base URL.")
    embedding_api_key: Optional[str] = Field(None, description="Embedding model API key.")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model name.")
    embedding_base_url: Optional[str] = Field(None, description="Custom embedding base URL.")
    retrievers: List[RetrieverType] = Field(default=["vector", "cypher", "synonym"], description="Retriever types to combine.")
    similarity_top_k: int = Field(5, ge=1, le=100, description="Top-K results from vector similarity.")
