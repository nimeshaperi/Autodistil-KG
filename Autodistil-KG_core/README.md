# Autodistil-KG

**Autodistil-KG** is a modular pipeline for building instruction-tuning datasets from knowledge graphs and fine-tuning small language models. It traverses a graph (e.g. Neo4j), uses an LLM to generate question–answer pairs in CHATML format, optionally converts and prepares that data, then fine-tunes a model (e.g. via Unsloth). Each stage can be run on its own or as part of the full pipeline.

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Stage Reference](#stage-reference)
- [Data Formats](#data-formats)
- [Scripts and Examples](#scripts-and-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline has four stages:

| Stage | Purpose |
|-------|--------|
| **Graph Traverser** | Connects to a knowledge graph (Neo4j), traverses nodes with an LLM-guided agent, and produces a CHATML dataset (conversations). |
| **ChatML Converter** | Loads CHATML (from file or previous stage), normalizes it, and optionally prepares JSONL with `messages` for fine-tuning. |
| **FineTuner** | Fine-tunes a model (e.g. Gemma, Llama, Qwen) on the prepared dataset using Unsloth. |
| **Evaluator** | Placeholder for future evaluation metrics. |

You can run the full sequence or any single stage (e.g. only the converter, or only the finetuner) with your own config and inputs.

---

## Pipeline Architecture

```
┌──────────────────┐    ┌─────────────────────┐    ┌──────────────┐    ┌────────────┐
│ Graph Traverser  │───▶│ ChatML Converter    │───▶│ FineTuner    │───▶│ Evaluator  │
│ (Neo4j + LLM)    │    │ (normalize/prepare) │    │ (Unsloth)    │    │ (stub)     │
└──────────────────┘    └─────────────────────┘    └──────────────┘    └────────────┘
       │                           │                        │
       ▼                           ▼                        ▼
  CHATML JSONL              Prepared JSONL           Model output
  (conversations)            (messages)                (checkpoints)
```

Stages share a **pipeline context**: paths and optional in-memory data (e.g. `chatml_dataset_path`, `prepared_dataset_path`, `model_output_path`). Running one stage can feed the next; you can also run a stage alone by providing input paths (and optionally a JSON or programmatic config).

---

## Requirements

- **Python**: 3.13+
- **Package manager**: [Poetry](https://python-poetry.org/) (recommended) or pip

**Per-stage dependencies:**

- **Graph Traverser**: Neo4j graph database, Redis (for traversal state), an LLM (OpenAI, Gemini, Claude, Ollama, or vLLM). See [example.env](example.env) for required env vars.
- **ChatML Converter**: No extra services; reads/writes JSONL.
- **FineTuner**: Unsloth and related ML stack (`unsloth`, `trl`, `datasets`, `transformers`). GPU recommended for training. **System requirement**: Python development headers (`python3-dev` or `python3.13-dev`) for Triton/CUDA extensions.

The project declares `neo4j` in `pyproject.toml`. Other providers (Redis, LLM SDKs, Unsloth) are used by the respective modules; install them as needed (see [Installation](#installation)).

---

## Installation

```bash
git clone <repository-url>
cd Autodistil-KG
poetry install   # or: make install
```

**Optional extras:**
- Graph Traverser (Neo4j, Redis, LLM): `poetry add python-dotenv`
- FineTuner (Unsloth): `poetry install -E finetune` or `pip install "autodistil-kg[finetune]"`  
  - **Required**: Python dev headers: `sudo apt install python3-dev` (Ubuntu/Debian) or `python3.13-dev` for Python 3.13

---

## Quick Start

### Run with no arguments

From the project root, run with the default config (ChatML converter on sample data):

```bash
make run
# or
poetry run python run.py
# or (after poetry install)
poetry run python -m autodistil_kg.run
```

If `config/default.json` exists, it is used. Otherwise a bundled default runs the ChatML converter on sample data—creating `data/sample_chatml.jsonl` if needed and writing `output/prepared.jsonl`. The root `run.py` script works without installing the package (run from project root).

### 1. ChatML Converter only (no graph, no training)

Convert an existing CHATML JSONL file into prepared format for fine-tuning:

```bash
poetry run python -m autodistil_kg.run --stage chatml_converter --input data/chatml.jsonl --output data/prepared.jsonl
```

### 2. FineTuner only (training from prepared JSONL)

Train on a prepared JSONL file (must contain `messages` per record):

```bash
poetry run python -m autodistil_kg.run --stage finetuner \
  --input data/prepared.jsonl \
  --output ./finetuner_output \
  --model-name unsloth/gemma-3-270m-it \
  --model-type gemma
```

### 3. Full pipeline or single stage via JSON config

```bash
poetry run python -m autodistil_kg.run --config config/pipeline.json
```

See [Configuration](#configuration) for the structure of `pipeline.json`.

### 4. Programmatic usage

Run only the ChatML converter from Python:

```python
from autodistil_kg.pipeline import Pipeline
from autodistil_kg.pipeline.config import PipelineConfig, ChatMLConverterStageConfig
from autodistil_kg.pipeline.interfaces import PipelineContext

config = PipelineConfig(
    chatml_converter=ChatMLConverterStageConfig(
        input_path="data/chatml.jsonl",
        output_path="data/prepared.jsonl",
        prepare_for_finetuning=True,
    ),
)
pipeline = Pipeline(config)
ctx = PipelineContext(chatml_dataset_path="data/chatml.jsonl")
result = pipeline.run_stage("chatml_converter", context=ctx)
```

Run only the FineTuner:

```python
from autodistil_kg.pipeline import Pipeline
from autodistil_kg.pipeline.config import PipelineConfig, FineTunerStageConfig
from autodistil_kg.pipeline.interfaces import PipelineContext

config = PipelineConfig(
    finetuner=FineTunerStageConfig(
        model_name="unsloth/gemma-3-270m-it",
        model_type="gemma",
        train_data_path="data/prepared.jsonl",
        output_dir="models/my_model",
    ),
)
pipeline = Pipeline(config)
result = pipeline.run_stage("finetuner")
```

Run the full pipeline (all configured stages in order):

```python
from autodistil_kg.pipeline import Pipeline
from autodistil_kg.pipeline.config import PipelineConfig, ...
from autodistil_kg.pipeline.interfaces import PipelineContext

config = PipelineConfig(
    graph_traverser=...,   # see Stage Reference
    chatml_converter=...,
    finetuner=...,
    run_stages=["graph_traverser", "chatml_converter", "finetuner"],
)
pipeline = Pipeline(config)
context, results = pipeline.run()
```

---

## Configuration

### Environment variables (Graph Traverser)

For the Graph Traverser you can use a `.env` file. Copy the template and fill in your values:

```bash
cp example.env .env
# Edit .env with your Neo4j, Redis, and LLM credentials
```

Main variables:

| Variable | Description |
|----------|-------------|
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Neo4j connection |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`, `REDIS_KEY_PREFIX` | Redis state storage |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | OpenAI |
| `GEMINI_PROJECT_ID`, `GEMINI_LOCATION`, `GEMINI_MODEL`, `GEMINI_CREDENTIALS_PATH` | Google Gemini |
| `CLAUDE_API_KEY`, `CLAUDE_MODEL` | Anthropic Claude |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Ollama (local) |
| `VLLM_BASE_URL`, `VLLM_MODEL` | vLLM (local) |

Load env in code (e.g. before building the pipeline) with the graph_traverser helper:

```python
from autodistil_kg.graph_traverser.env_config import (
    load_env_file,
    get_graph_db_config_from_env,
    get_llm_config_from_env,
    get_state_storage_config_from_env,
)
from autodistil_kg.graph_traverser.config import (
    GraphTraverserAgentConfig,
    TraversalConfig,
    DatasetGenerationConfig,
    TraversalStrategy,
)

load_env_file()  # loads .env from project root

agent_config = GraphTraverserAgentConfig(
    graph_db=get_graph_db_config_from_env(),
    llm=get_llm_config_from_env(),
    state_storage=get_state_storage_config_from_env(),
    traversal=TraversalConfig(
        strategy=TraversalStrategy.BFS,
        max_nodes=500,
        max_depth=5,
    ),
    dataset=DatasetGenerationConfig(
        seed_prompts=["What can you tell me about this node? Describe: {properties}"],
        output_path="output/dataset.jsonl",
    ),
)
```

### JSON config file

The project includes `config/default.json` for quick runs. The CLI can load any JSON file for multi-stage config. Example shape:

```json
{
  "output_dir": "./output",
  "run_stages": ["graph_traverser", "chatml_converter", "finetuner"],
  "graph_traverser": {
    "output_path": "output/chatml.jsonl",
    "traversal": { "strategy": "bfs", "max_nodes": 500, "max_depth": 5 },
    "dataset": {
      "seed_prompts": ["What can you tell me about this node? Describe: {properties}"],
      "include_metadata": true
    }
  },
  "chatml_converter": {
    "input_path": "data/chatml.jsonl",
    "output_path": "data/prepared.jsonl",
    "prepare_for_finetuning": true
  },
  "finetuner": {
    "model_name": "unsloth/gemma-3-270m-it",
    "model_type": "gemma",
    "train_data_path": "data/prepared.jsonl",
    "eval_data_path": "data/eval.jsonl",
    "output_dir": "./finetuner_output",
    "max_seq_length": 2048,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "learning_rate": 2e-4
  },
  "evaluator": {
    "model_path": "./finetuner_output",
    "eval_dataset_path": "data/eval.jsonl",
    "output_report_path": "./eval_report.json",
    "metrics": []
  }
}
```

- Omit a stage key (e.g. `graph_traverser`) to leave that stage unconfigured.
- `run_stages` controls which stages run and in what order when you run the full pipeline with `--config`.
- **Graph traverser** requires a `.env` file (copy from `example.env`) with Neo4j, Redis, and LLM credentials. Use `config/pipeline_full.json` to run graph traverser + chatml converter.

---

## Running the Pipeline

### CLI

```bash
# Default: run with bundled config (ChatML converter on sample data)
poetry run python -m autodistil_kg.run
# or: make run

# Custom config
poetry run python run.py --config config/pipeline.json

# Full pipeline with Graph Traverser (requires Neo4j, Redis, LLM in .env)
poetry run python run.py --config config/pipeline_full.json

# Graph traverser only (requires .env and config with graph_traverser)
poetry run python run.py --config config/pipeline_full.json --stage graph_traverser

# Single stage: ChatML converter
poetry run python -m autodistil_kg.run --stage chatml_converter --input data/chatml.jsonl [--output data/prepared.jsonl]

# Single stage: FineTuner
poetry run python -m autodistil_kg.run --stage finetuner --input data/prepared.jsonl [--output ./finetuner_output]

# Help
poetry run python -m autodistil_kg.run --help
```

- **No arguments**: Uses `config/default.json` if it exists, otherwise runs a bundled default.
- **`--config`**: Config file defines inputs, outputs, and which stages run.
- **`--stage`**: Requires `--input` (and `--output` as needed).
- **`-v` / `--verbose`**: Enable DEBUG logging (e.g. Redis writes, queue size). Alternatively set `LOG_LEVEL=DEBUG` in the environment.

### Programmatic

- **Single stage**: `Pipeline(config).run_stage("stage_name", context=ctx)`  
  Pass a `PipelineContext` with the right paths (e.g. `chatml_dataset_path` or `prepared_dataset_path`) when the stage needs input from file.
- **Full pipeline**: `Pipeline(config).run(context=ctx)`  
  Returns updated `context` and a list of `StageResult`s. Context is updated after each stage (e.g. `chatml_dataset_path`, then `prepared_dataset_path`, then `model_output_path`).

---

## Stage Reference

### Graph Traverser

- **Input**: Neo4j graph; traversal and dataset config (strategy, seed prompts, output path).
- **Output**: CHATML dataset (in-memory and/or JSONL at `output_path`); context gets `chatml_dataset` and `chatml_dataset_path`.

Configure via `GraphTraverserStageConfig`: either pass a full `GraphTraverserAgentConfig` or pass `graph_db`, `llm_config`, `state_storage`, `traversal`, and `dataset` so the stage can build the agent config. Optional `output_path` for the JSONL file.

Traversal strategies: `bfs`, `dfs`, `random`, `semantic`. For more detail (LLM providers, Redis, Neo4j), see [src/autodistil_kg/graph_traverser/README.md](src/autodistil_kg/graph_traverser/README.md).

### ChatML Converter

- **Input**: CHATML JSONL path (`input_path` or `context.chatml_dataset_path`) or in-memory `context.chatml_dataset`.
- **Output**: Normalized/prepared JSONL (each line: object with `messages`) and/or in-memory; context gets `prepared_dataset_path` and `extra["prepared_messages"]`.

Options: `input_path`, `output_path`, `prepare_for_finetuning` (default `True`), `chat_template` (optional).

### FineTuner

- **Input**: Prepared JSONL path (`train_data_path` or `context.prepared_dataset_path` / `context.chatml_dataset_path`).
- **Output**: Trained model/adapters in `output_dir`; context gets `model_output_path`.

You can pass a full `UnslothFineTunerConfig` in `FineTunerStageConfig.finetuner_config`, or set: `model_name`, `model_type`, `train_data_path`, `eval_data_path`, `output_dir`, `max_seq_length`, `num_train_epochs`, `per_device_train_batch_size`, `learning_rate`, etc. Supported `model_type` values include: `gemma`, `gemma2`, `gemma3`, `llama`, `llama3`, `mistral`, `qwen`, `qwen2`, `qwen3`, `phi3`, `phi4`, `zephyr`, `chatml`, `alpaca`, `vicuna`.

### Evaluator

- **Input**: `model_path`, `eval_dataset_path`, optional `metrics`, `output_report_path`.
- **Output**: Placeholder; intended for future evaluation reports.

---

## Data Formats

### CHATML (Graph Traverser / Converter input)

JSONL where each line is a conversation object, e.g.:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {"node_id": "...", "labels": [...], "depth": 1}}
```

### Prepared JSONL (Converter output / FineTuner input)

JSONL where each line has at least `messages`:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The FineTuner (Unsloth) expects this structure and uses the appropriate chat template for the chosen `model_type`.

---

## Scripts and Examples

- **`scripts/dblp_loader_to_neo4j.py`** – Example script to load DBLP data into Neo4j (requires `scripts/dblp.xml.gz` or similar).
- **`scripts/graph_traverser/graph_traverser.ipynb`** – Notebook for the graph traverser agent.
- **`scripts/finetuning/unsloth_finetuner_script.ipynb`** – Unsloth fine-tuning notebook.

For graph traverser–specific configuration (Neo4j, Redis, LLMs, traversal strategies), see [src/autodistil_kg/graph_traverser/README.md](src/autodistil_kg/graph_traverser/README.md).

---

## Troubleshooting

- **“No CHATML dataset in context and no valid input_path”**  
  For the ChatML converter, set `context.chatml_dataset_path` or `ChatMLConverterStageConfig.input_path` to an existing JSONL file, or pass `--input` in the CLI.

- **“FineTunerStage: no train_data_path and none in context”**  
  Set `train_data_path` on `FineTunerStageConfig` or ensure the context has `prepared_dataset_path` or `chatml_dataset_path` pointing to a valid JSONL file.

- **Graph Traverser connection errors**  
  Ensure Neo4j and Redis are running and that `.env` (or your env) has the correct URIs, credentials, and (for LLM) API keys or base URLs.

- **Unsloth / CUDA errors**  
  Install the correct Unsloth and PyTorch stack for your GPU; see Unsloth documentation. The FineTuner stage expects the same JSONL format as described above.

- **Evaluator stage**  
  The evaluator is a stub; it does not produce real metrics yet.

---

## License and Contact

See the repository for license information. For questions or contributions, open an issue or contact the maintainers (e.g. author in [pyproject.toml](pyproject.toml)).
