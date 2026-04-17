// ===== Literal union types (mirroring API Pydantic models) =====
export type StageId = 'graph_traverser' | 'chatml_converter' | 'finetuner' | 'evaluator'
export type TraversalStrategy = 'bfs' | 'dfs' | 'random' | 'semantic' | 'reasoning'
export type LLMProviderType = 'openai' | 'openai_compatible' | 'claude' | 'gemini' | 'ollama' | 'vllm'
export type EvalMode = 'internal' | 'cli' | 'noop'
export type RetrieverType = 'vector' | 'cypher' | 'synonym'
export type OutputFormat = 'jsonl' | 'json'
export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'

// ===== Sub-config request types =====

export interface Neo4jConfigRequest {
  uri: string
  username: string
  password: string
  database?: string
}

export interface RedisConfigRequest {
  host: string
  port: number
  db: number
  password?: string
  key_prefix?: string
}

export interface LLMConfigRequest {
  provider: LLMProviderType
  api_key?: string
  model?: string
  base_url?: string
  project_id?: string
  location?: string
  credentials_path?: string
}

export interface TraversalConfigRequest {
  strategy: TraversalStrategy
  max_nodes: number
  max_depth: number
  reasoning_depth: number
  max_paths_per_node: number
  path_batch_size: number
  num_workers: number
  relationship_types?: string[]
  node_labels?: string[]
  seed_node_ids?: string[]
}

export interface AlignmentConfigRequest {
  domain_focus?: string
  domain_keywords?: string[]
  style_guide?: string
  target_audience?: string
  max_answer_length?: number
  min_answer_length?: number
  quality_filter?: boolean
  quality_threshold?: number
  reference_texts_path?: string
}

export interface DatasetConfigRequest {
  seed_prompts: string[]
  system_message?: string
  prompt_template?: string
  include_metadata: boolean
  output_format: OutputFormat
  output_path?: string
  eval_sample_ratio?: number
  eval_rephrase?: boolean
}

// ===== Stage config request types =====

export interface GraphTraverserConfigRequest {
  output_path?: string
  traversal: TraversalConfigRequest
  dataset: DatasetConfigRequest
  alignment?: AlignmentConfigRequest
  neo4j?: Neo4jConfigRequest
  redis?: RedisConfigRequest
  llm?: LLMConfigRequest
  llm_provider?: string
}

export interface ChatMLConverterConfigRequest {
  input_path?: string
  output_path?: string
  prepare_for_finetuning: boolean
  chat_template?: string
}

export interface FineTunerConfigRequest {
  model_name: string
  model_type?: string
  train_data_path?: string
  eval_data_path?: string
  output_dir?: string
  max_seq_length: number
  num_train_epochs: number
  per_device_train_batch_size: number
  learning_rate: number
}

export interface GraphRAGConfigRequest {
  neo4j_uri: string
  neo4j_user: string
  neo4j_password: string
  neo4j_database: string
  llm_provider?: LLMProviderType
  llm_api_key?: string
  llm_model: string
  llm_base_url?: string
  embedding_api_key?: string
  embedding_model: string
  embedding_base_url?: string
  retrievers: RetrieverType[]
  num_agents: number
  similarity_top_k: number
}

export interface EvaluatorConfigRequest {
  model_path?: string
  eval_dataset_path?: string
  output_report_path?: string
  metrics: string[]
  evalg_mode: EvalMode
  max_new_tokens?: number
  max_seq_length?: number
  base_model_provider?: LLMProviderType | 'local'
  base_model_name?: string
  base_model_api_key?: string
  base_model_base_url?: string
  graph_rag_enabled: boolean
  graph_rag_config?: GraphRAGConfigRequest
  judge_provider?: LLMProviderType
  judge_model?: string
  judge_api_key?: string
  judge_base_url?: string
  judge_max_tokens?: number
  // Comparison system toggles
  eval_distilled?: boolean
  eval_base?: boolean
  eval_graph_rag?: boolean
  // vLLM serving
  use_vllm?: boolean
  vllm_gpu_memory_utilization?: number
  vllm_max_model_len?: number
  max_eval_samples?: number
}

// ===== Top-level pipeline request =====

export interface PipelineConfigPayload {
  output_dir?: string
  run_stages: StageId[]
  log_level?: LogLevel
  graph_traverser?: GraphTraverserConfigRequest
  chatml_converter?: ChatMLConverterConfigRequest
  finetuner?: FineTunerConfigRequest
  evaluator?: EvaluatorConfigRequest
}

// ===== Response types =====

export interface StageResultResponse {
  success: boolean
  error?: string
  metadata: Record<string, unknown>
}

export type RunStatus = 'running' | 'completed' | 'failed' | 'cancelled' | 'unknown'

export interface PipelineRunResultResponse {
  run_id: string
  status: RunStatus
  success: boolean
  context?: Record<string, unknown>
  results?: StageResultResponse[]
  error?: string
  stages?: string[]
  current_stage?: string
}

export interface InferenceLLMRequest {
  provider: LLMProviderType
  model?: string
  api_key?: string
  base_url?: string
  project_id?: string
  location?: string
  credentials_path?: string
  messages: Array<{ role: string; content: string }>
  temperature?: number
  max_tokens?: number
}

export interface InferenceLLMResponse {
  response: string
  provider: string
  model?: string
}

export interface InferenceGraphRAGRequest {
  question: string
  neo4j_uri?: string
  neo4j_user?: string
  neo4j_password?: string
  neo4j_database?: string
  llm_api_key?: string
  llm_model?: string
  llm_base_url?: string
  embedding_api_key?: string
  embedding_model?: string
  embedding_base_url?: string
  retrievers?: RetrieverType[]
  num_agents?: number
  similarity_top_k?: number
}

export interface InferenceGraphRAGResponse {
  answer: string
  source_nodes: unknown[]
  metadata: Record<string, unknown>
}

export interface AvailableModel {
  run_id: string
  model_path: string
  label: string
}

export interface RegisteredModel {
  model_id: string
  model_path: string
  base_model?: string
  description?: string
}

export interface InferenceFinetunedRequest {
  model_id: string
  messages: Array<{ role: string; content: string }>
  max_new_tokens?: number
  temperature?: number
}

export interface InferenceFinetunedResponse {
  response: string
  model_id: string
}

export interface RegisterModelRequest {
  model_id: string
  model_path: string
  base_model?: string
  description?: string
}

// ===== Form-state types (CSV fields as strings for UI inputs) =====
export interface TraversalConfig {
  strategy: TraversalStrategy
  max_nodes: number
  max_depth: number
  reasoning_depth?: number
  max_paths_per_node?: number
  path_batch_size?: number
  num_workers?: number
  relationship_types?: string  // CSV in form, converted to string[] in payload
  node_labels?: string
  seed_node_ids?: string
}

export interface AlignmentConfig {
  domain_focus?: string
  domain_keywords?: string   // CSV in form, converted to string[] in payload
  style_guide?: string
  target_audience?: string
  max_answer_length?: number
  min_answer_length?: number
  quality_filter: boolean
  quality_threshold: number
  reference_texts_path?: string
}

export interface DatasetConfig {
  seed_prompts: string[]
  system_message?: string
  include_metadata: boolean
  output_format?: OutputFormat
  output_path?: string
  eval_sample_ratio?: number
  eval_rephrase?: boolean
}

export type Neo4jConfig = Neo4jConfigRequest
export type RedisConfig = RedisConfigRequest
export type LLMConfig = LLMConfigRequest

export interface GraphTraverserConfig {
  output_path?: string
  traversal: TraversalConfig
  dataset: DatasetConfig
  alignment?: AlignmentConfig
  neo4j?: Neo4jConfigRequest
  redis?: RedisConfigRequest
  llm?: LLMConfigRequest
  llm_provider?: string
}

export type ChatMLConverterConfig = ChatMLConverterConfigRequest

export interface FineTunerConfig {
  model_name: string
  model_type?: string
  train_data_path?: string
  eval_data_path?: string
  output_dir?: string
  save_as?: string  // Frontend-only: sets save folder name within outputDir
  max_seq_length: number
  num_train_epochs: number
  per_device_train_batch_size: number
  learning_rate: number
}

/** Form-state version of GraphRAGConfigRequest — all fields optional for incremental editing */
export interface GraphRAGConfigForm {
  neo4j_uri?: string
  neo4j_user?: string
  neo4j_password?: string
  neo4j_database?: string
  llm_provider?: LLMProviderType
  llm_api_key?: string
  llm_model?: string
  llm_base_url?: string
  embedding_api_key?: string
  embedding_model?: string
  embedding_base_url?: string
  retrievers?: RetrieverType[]
  num_agents?: number
  similarity_top_k?: number
}

export interface EvaluatorConfig {
  model_path?: string
  eval_dataset_path?: string
  output_report_path?: string
  metrics: string[]
  evalg_mode: EvalMode
  max_new_tokens?: number
  max_seq_length?: number
  base_model_provider?: LLMProviderType | 'local' | 'none'
  base_model_name?: string
  base_model_api_key?: string
  base_model_base_url?: string
  graph_rag_enabled: boolean
  graph_rag_config?: GraphRAGConfigForm
  judge_provider?: LLMProviderType
  judge_model?: string
  judge_api_key?: string
  judge_base_url?: string
  judge_max_tokens?: number
  // Comparison system toggles
  eval_distilled?: boolean
  eval_base?: boolean
  eval_graph_rag?: boolean
  // vLLM serving
  use_vllm: boolean
  vllm_gpu_memory_utilization: number
  vllm_max_model_len?: number
  max_eval_samples?: number
}

// ===== Constants =====

export const STAGE_ORDER: StageId[] = ['graph_traverser', 'chatml_converter', 'finetuner', 'evaluator']

export const STAGE_LABELS: Record<StageId, string> = {
  graph_traverser: 'Graph Traverser',
  chatml_converter: 'ChatML Converter',
  finetuner: 'FineTuner',
  evaluator: 'Evaluator',
}

export const STAGE_DESCRIPTIONS: Record<StageId, string> = {
  graph_traverser: 'Traverse Neo4j graph and generate conversations',
  chatml_converter: 'Convert and prepare ChatML datasets',
  finetuner: 'Fine-tune models with Unsloth',
  evaluator: 'Compare finetuned vs base models',
}
