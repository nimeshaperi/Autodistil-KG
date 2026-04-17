# Architecture 
The main purpose of this package is to abstract away the complexity of the following components
1. Graph Traversing
2. ChatML Dataset Creation 
3. Finetuner
4. Evaluator

The idea is to have interfaces that allow running each module independent to each other for other use cases. This will act as syntactic sugar and easy interface to work with.

The package should also allow the ability to define a set pipeline with each component configuration that can be run. This will probably be the most common usage.

Graph Traverser -> ChatML Converter -> FineTuner -> Evaluator

---

## Pipeline (modular, configurable stages)

- **`autodistil_kg.pipeline`**: Orchestrates stages; each stage is a modular class with its own config. Run the full pipeline or a single stage from the main runtime.
- **Stages**:
  - **Graph Traverser** (`GraphTraverserStage`): Context extraction + CHATML dataset generation. Config: `GraphTraverserStageConfig` (agent_config or graph_db, llm, state_storage, traversal, dataset; optional `output_path`).
  - **ChatML Converter** (`ChatMLConverterStage`): Normalize / prepare CHATML for finetuning. Input: `context.chatml_dataset_path` or `context.chatml_dataset`; output: prepared JSONL or in-memory. Config: `ChatMLConverterStageConfig` (input_path, output_path, prepare_for_finetuning, chat_template).
  - **FineTuner** (`FineTunerStage`): Unsloth-based finetuning. Config: `FineTunerStageConfig` (model_name, model_type, train_data_path, output_dir, or full `UnslothFineTunerConfig`). Model types: gemma, llama, qwen, mistral, phi, etc.
  - **Evaluator** (`EvaluatorStage`): Stub for future metrics. Config: `EvaluatorStageConfig`.
- **Main runtime**: `python -m autodistil_kg.run --stage <name> --input <path> [--output <path>]` or programmatic `Pipeline(config).run()` / `Pipeline(config).run_stage("chatml_converter", context)`.

---

## Graph Traverser
The graph traverser is a semantic-aware agent that goes through the Knowledge Graph (Should be able to configure through an interface that has several implementations for Neo4j, and other graph database providers) and creates a dataset that is CHATML compliant, some seed prompts should be set by the user to understand the structure the output should and so on as it creates Prompt & Response pairs. A strategy to keep the state of the Knowledge Graph as visited, not visited and metadata as such should be cached on memory storage (Maybe redis). The agent should be configured in a way that it'll be compatible on a LLM interface that can take the following LLMs into account chat-gpt, gemini (vertex AI), claude, local LLM (Ollama, vllm, ...)

---

## Finetuner
- **`autodistil_kg.finetuner`**: Unsloth-based fine-tuning, configurable for many model types (Gemma, Llama, Qwen, Mistral, Phi, etc.). Uses `UnslothFineTunerConfig` (model_name, model_type, LoRA and SFT params, train/eval paths, chat template). Expects JSONL with `messages` (list of `{role, content}`). Optional deps: `unsloth`, `trl`, `datasets`, `transformers`.

