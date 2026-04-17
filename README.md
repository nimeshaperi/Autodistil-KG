# Autodistil-KG

A research tool that traverses a biomedical knowledge graph (Hetionet), generates ChatML-formatted fine-tuning datasets from the traversal, and optionally fine-tunes a language model on those datasets.

The project is a monorepo with four submodules:

| Submodule | Language | Role |
|---|---|---|
| `Autodistil-KG_core` | Python 3.13 | Pipeline engine (traversal → ChatML → fine-tune → eval) |
| `Autodistil-KG_graphrag` | Python 3.13 | Graph RAG layer (LlamaIndex + Neo4j) used by the API |
| `Autodistil-KG_api` | Python 3.13 / FastAPI | REST + WebSocket server that drives the pipeline |
| `Autodistil-KG_client` | TypeScript / React + Vite | Browser UI for configuring and monitoring pipelines |

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Quick Start — Docker (recommended)](#quick-start--docker-recommended)
4. [Service Ports](#service-ports)
5. [Environment Variables](#environment-variables)
6. [Hetionet Database Setup](#hetionet-database-setup)
7. [Using the UI](#using-the-ui)
8. [REST API](#rest-api)
9. [Pipeline Stages](#pipeline-stages)
10. [Development Setup (no Docker)](#development-setup-no-docker)
11. [Workspace & Outputs](#workspace--outputs)
12. [Troubleshooting](#troubleshooting)

---

## Architecture

```
Browser (port 3000)
    │
    │  HTTP /api/*  →  strip /api prefix
    │  WS  /ws      →  forward as-is
    ▼
nginx (client container, port 80)
    │
    ▼
FastAPI (api container, port 8000)
    ├── REST  /pipelines/run        →  Core pipeline
    ├── REST  /pipelines/runs/{id}  →  Async job status
    ├── WS    /ws                   →  Real-time event stream
    └── REST  /inference/*          →  GraphRAG queries
         │
         ├── autodistil_kg (core)
         │     Graph Traverser → ChatML Converter → FineTuner → Evaluator
         │
         └── autodistil_kg_graphrag
               LlamaIndex PropertyGraph → Neo4j RAG
                    │
                    ▼
Neo4j 5 (port 7687 Bolt / 7474 HTTP)
    Hetionet dataset — 47,031 nodes, 2,250,197 relationships

Redis (port 6379)
    ├── Pipeline job queue (async runs)
    └── Traversal state / checkpoint cache
```

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Docker Engine | 24+ | With Compose V2 (`docker compose`) |
| Docker Compose | V2 (bundled) | `docker compose version` to verify |
| Python 3 | 3.x | Only needed for the Hetionet import script (Step 3) |
| 8 GB RAM | — | Neo4j needs ~3 GB; allow headroom for API + build |
| 15 GB disk | — | Neo4j data volume + Docker images |

> **Apple Silicon / ARM64**: All images (python:3.13-slim, neo4j:5, redis:7-alpine, node:20-alpine, nginx:alpine) ship multi-arch manifests and run natively.

---

## Quick Start — Docker (recommended)

### Step 1 — Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/nimeshaperi/Autodistil-KG.git
cd Autodistil-KG
```

If you cloned without `--recurse-submodules`, initialise them now:

```bash
git submodule update --init --recursive
```

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at least one LLM provider key. The pipeline will not run without one.

```bash
# Minimum required — pick the provider you use:
OPENAI_API_KEY=sk-...          # OpenAI or any OpenAI-compatible endpoint
# or
CLAUDE_API_KEY=sk-ant-...      # Anthropic Claude
# or
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3
```

The `NEO4J_PASSWORD` defaults to `password`. Change it if you prefer, but make sure `NEO4J_PASSWORD` in `.env` matches across the stack (docker compose reads it automatically).

### Step 3 — Import the Hetionet knowledge graph

This is a one-time step. It takes **3–8 minutes** depending on your machine.

```bash
# Start only the infrastructure (Neo4j must be stopped for the import)
docker compose up -d redis
# Neo4j must NOT be running during import — skip it for now

# Run the import script
bash scripts/setup-hetionet.sh
```

Expected output at the end:

```
Imported: 47031 nodes, 2250197 relationships, 6799401 properties
```

> The script starts the legacy `dhimmel/hetionet` (Neo4j 3.5) container on temporary ports,
> exports all nodes and relationships to CSV, then runs `neo4j-admin database import full`
> into the `autodistil-kg_neo4j_data` Docker volume. The legacy container is removed afterwards.

### Step 4 — Build and start all services

```bash
docker compose up --build -d
```

The first build downloads base images and compiles Python/Node dependencies. It takes **5–15 minutes**. Subsequent starts are fast.

### Step 5 — Open the UI

| Service | URL |
|---|---|
| **Web UI** | http://localhost:3000 |
| **API docs** (Swagger) | http://localhost:8000/docs |
| **Neo4j Browser** | http://localhost:7474 |

Log into Neo4j Browser with username `neo4j` and the password from your `.env` (default: `password`).

---

## Service Ports

| Service | Internal port | Host port | Protocol |
|---|---|---|---|
| Web client (nginx) | 80 | **3000** | HTTP |
| API (FastAPI) | 8000 | **8000** | HTTP + WebSocket |
| Neo4j Browser | 7474 | **7474** | HTTP |
| Neo4j Bolt | 7687 | **7687** | Bolt |
| Redis | 6379 | **6379** | TCP |

The client nginx container proxies `/api/*` → API (strips `/api` prefix) and `/ws` → API WebSocket, so the browser never talks to the API directly.

---

## Environment Variables

All variables live in the root `.env` file. Docker compose passes them to the `api` service; some are overridden inside compose for container networking (e.g. `NEO4J_URI` becomes `bolt://neo4j:7687`).

### LLM Providers

Set **one** provider. The pipeline selects whichever credentials are present at runtime.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI secret key (`sk-…`) |
| `OPENAI_MODEL` | Model name (default: `gpt-4o`) |
| `OPENAI_BASE_URL` | Custom endpoint — leave blank for official OpenAI |
| `CLAUDE_API_KEY` | Anthropic API key (`sk-ant-…`) |
| `CLAUDE_MODEL` | e.g. `claude-3-opus-20240229` |
| `GEMINI_PROJECT_ID` | GCP project for Vertex AI |
| `GEMINI_LOCATION` | GCP region (default: `us-central1`) |
| `GEMINI_MODEL` | e.g. `gemini-pro` |
| `GEMINI_CREDENTIALS_PATH` | Path to service-account JSON (inside the API container) |
| `OLLAMA_BASE_URL` | e.g. `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | e.g. `llama3` |
| `VLLM_BASE_URL` | vLLM server URL |
| `VLLM_MODEL` | Model identifier |

### Neo4j

| Variable | Default (Docker) | Description |
|---|---|---|
| `NEO4J_PASSWORD` | `password` | Shared password for Neo4j auth |
| `NEO4J_URI` | `bolt://neo4j:7687` | Overridden in compose; use `bolt://localhost:7687` for local dev |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_DATABASE` | `neo4j` | Database name |

### Redis

| Variable | Default (Docker) | Description |
|---|---|---|
| `REDIS_HOST` | `redis` | Overridden in compose; use `localhost` for local dev |
| `REDIS_PORT` | `6379` | |
| `REDIS_DB` | `0` | |
| `REDIS_PASSWORD` | *(empty)* | Leave blank if no auth |

### Pipeline Defaults (optional)

These can also be set per-run via the UI or API payload.

| Variable | Options | Description |
|---|---|---|
| `TRAVERSAL_STRATEGY` | `bfs`, `dfs`, `random`, `semantic` | Graph traversal algorithm |
| `TRAVERSAL_MAX_NODES` | integer | Max nodes to visit per run |
| `TRAVERSAL_MAX_DEPTH` | integer | Maximum hop depth from seed nodes |
| `DATASET_OUTPUT_PATH` | file path | Output JSONL file (relative to workspace) |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | |

---

## Hetionet Database Setup

Hetionet is a biomedical knowledge graph with:
- **47,031 nodes** across 11 biological entity types (Gene, Disease, Compound, …)
- **2,250,197 relationships** across 24 types (TREATS, ASSOCIATES, INTERACTS, …)

The import script (`scripts/setup-hetionet.sh`) handles the full process automatically. See [Step 3](#step-3--import-the-hetionet-knowledge-graph) above.

### Re-importing from scratch

If you need to wipe and re-import:

```bash
docker compose stop neo4j
docker volume rm autodistil-kg_neo4j_data
bash scripts/setup-hetionet.sh
docker compose up -d neo4j
```

### Recommended graph filters for dataset generation

For a well-connected neuro-metabolic subgraph (300–500 nodes, good for training):

**Node Labels:** `Disease, Gene, Compound, Symptom`

**Relationship Types:**
```
ASSOCIATES_DaG, INTERACTS_GiG, TREATS_CtD, PALLIATES_CpD,
PRESENTS_DpS, RESEMBLES_DrD, BINDS_CbG, UPREGULATES_DuG, DOWNREGULATES_DdG
```

**Seed node element IDs** (8 neuro-metabolic diseases — IDs are instance-specific, look up by name if you re-import):

| Disease | Element ID |
|---|---|
| Schizophrenia | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:23039` |
| Bipolar disorder | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:22950` |
| Epilepsy syndrome | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:22947` |
| Autistic disorder | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:23043` |
| Obesity | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:22957` |
| Type 2 diabetes mellitus | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:23046` |
| Alzheimer's disease | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:22982` |
| Parkinson's disease | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:23006` |

> **Note:** Element IDs are specific to this Neo4j instance. After a re-import, look them up by name:
> ```cypher
> MATCH (d:Disease) WHERE d.name IN ['Schizophrenia', 'Alzheimer disease'] RETURN d.name, elementId(d)
> ```

Set `max_nodes` to 400–500 to stay within the target range.

---

## Using the UI

1. Open http://localhost:3000
2. Click **Configure Pipeline** to set:
   - Neo4j connection (pre-filled from server defaults)
   - LLM provider and model
   - Traversal strategy, max nodes, max depth
   - Seed node IDs and graph filters
3. Click **Run Pipeline** — the Traversal Activity panel streams real-time events
4. The subgraph card shows node properties and neighbour counts as each node is visited
5. When the traversal is done, the ChatML Converter stage runs automatically
6. Download the output dataset from the **Outputs** tab

---

## REST API

Base URL: `http://localhost:8000`

Interactive docs: http://localhost:8000/docs

### Key endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/pipelines/run` | Run a pipeline synchronously (returns result) |
| `POST` | `/pipelines/run?async=true` | Enqueue pipeline run — returns `{"run_id": "..."}` |
| `GET` | `/pipelines/runs/{run_id}` | Poll async run status |
| `WebSocket` | `/ws` | Bidirectional event stream |

### WebSocket protocol

Connect to `ws://localhost:8000/ws`, then send:

```json
{
  "action": "run",
  "config": {
    "neo4j_uri": "bolt://neo4j:7687",
    "llm_provider": "openai",
    "traversal_strategy": "bfs",
    "max_nodes": 200,
    "seed_node_ids": ["4:dbc36dbe-...:23039"]
  }
}
```

Server sends a stream of events:

```json
{"event": "run_start",      "run_id": "abc123"}
{"event": "pipeline_start", "stage": "graph_traverser"}
{"event": "stage_start",    "stage": "graph_traverser"}
{"event": "log",            "level": "INFO", "message": "Visiting node: Schizophrenia"}
{"event": "stage_end",      "stage": "graph_traverser", "status": "success"}
{"event": "done",           "run_id": "abc123"}
```

---

## Pipeline Stages

The pipeline runs sequentially through four stages. Each can also run standalone.

### 1. Graph Traverser

Traverses the Neo4j knowledge graph using an LLM-guided agent. Produces a JSONL file where each line describes a node, its properties, and its neighbourhood context.

Traversal strategies:
- `bfs` — breadth-first from seed nodes
- `dfs` — depth-first
- `random` — random walk
- `semantic` — LLM ranks which neighbours to visit next

Config key: `graph_traverser`

### 2. ChatML Converter

Reads the traversal JSONL and converts each entry into ChatML-formatted training examples (`system`, `user`, `assistant` turns). The output is a JSONL file ready for fine-tuning.

Config key: `chatml_converter`

### 3. FineTuner

Fine-tunes a base language model on the ChatML dataset using the `trl` + `unsloth` stack. Requires GPU and the `INSTALL_FINETUNE=1` build arg (disabled by default in Docker).

Config key: `finetuner`

### 4. Evaluator

Runs evaluation metrics (via DeepEval) on the fine-tuned model outputs.

Config key: `evaluator`

---

## Development Setup (no Docker)

Use this if you want to iterate on the code without rebuilding Docker images.

### Requirements

- Python 3.13 (exact — core and API enforce `>=3.13,<3.14`)
- [Poetry](https://python-poetry.org/docs/#installation) (`pip install poetry`)
- Node.js 20+ and npm
- Docker (for Redis and Neo4j — run `docker compose up -d redis neo4j`)

### Core pipeline

```bash
cd Autodistil-KG_core
cp .env.example .env   # fill in your keys
poetry install
poetry run python -m autodistil_kg.run --help
```

### API server

```bash
# Install core in editable mode first
cd Autodistil-KG_core && pip install -e . && cd ..

# Install graphrag
cd Autodistil-KG_graphrag && pip install -e . && cd ..

# Install and run the API
cd Autodistil-KG_api
pip install -e .
export KG_PIPELINE_WORKSPACE="$(cd ../Autodistil-KG_core && pwd)"
export NEO4J_URI=bolt://localhost:7687
export REDIS_HOST=localhost
uvicorn autodistilkg_api.main:app --reload --host 0.0.0.0 --port 8000
```

### Web client (dev server with hot-reload)

```bash
cd Autodistil-KG_client
npm install
npm run dev
# Open http://localhost:5173
# Vite proxies /api/* and /ws to localhost:8000 automatically
```

### GraphRAG (standalone)

```bash
cd Autodistil-KG_graphrag
cp .env.example .env   # fill in NEO4J_* and OPENAI_API_KEY
poetry install
poetry run autodistil-kg-graphrag --query "What genes are associated with Alzheimer's disease?"
```

---

## Workspace & Outputs

The API container mounts a Docker volume at `/app/workspace`. When running locally, set `KG_PIPELINE_WORKSPACE` to point at the `Autodistil-KG_core` directory.

Default output paths (relative to workspace):

| Output | Path |
|---|---|
| Traversal JSONL | `output/traversal.jsonl` |
| ChatML dataset | `output/chatml.jsonl` |
| Prepared dataset | `output/prepared/` |
| Fine-tuned model | `output/model/` |
| Eval report | `output/eval_report.json` |

To access outputs from the Docker volume:

```bash
# Copy outputs to host
docker cp autodistil-kg-api:/app/workspace/output ./output

# Or mount a host directory instead of a volume (add to docker-compose.yml api service):
# volumes:
#   - ./workspace:/app/workspace
```

---

## Troubleshooting

### Services won't start

```bash
docker compose ps          # check status
docker compose logs api    # view API logs
docker compose logs neo4j  # view Neo4j logs
```

### Neo4j Browser shows 0 nodes after import

The import may have failed silently. Check the import log output for errors. Re-run:

```bash
docker compose stop neo4j
docker volume rm autodistil-kg_neo4j_data
bash scripts/setup-hetionet.sh
docker compose up -d neo4j
```

### `Unable to retrieve routing information` (bolt URI error)

You're using `neo4j://` instead of `bolt://`. Change the URI scheme:
- Docker stack: `bolt://neo4j:7687`
- Local dev: `bolt://localhost:7687`

### `Database name parameter not supported in Bolt Protocol 3.0`

You're connecting to the legacy Hetionet 3.5 container (port 7687) instead of the Neo4j 5 container (also 7687 in compose). Make sure the compose neo4j service is running and the URI points to it.

### API container exits immediately

Check for missing environment variables:

```bash
docker compose logs api | head -50
```

The most common causes are a missing `.env` file or an LLM provider key that hasn't been set.

### `Invalid credential` on Neo4j

The Neo4j container was started with `NEO4J_AUTH=none` at some point and now expects no auth, or vice versa. Delete the volume and restart:

```bash
docker compose down -v   # WARNING: deletes all volumes including Hetionet data
bash scripts/setup-hetionet.sh
docker compose up -d
```

### Port conflicts

If another service is using ports 3000, 7474, 7687, 8000, or 6379, edit the `ports` section in `docker-compose.yml` to remap:

```yaml
ports:
  - "3001:80"   # e.g. map client to host port 3001
```

### Build fails: `Python 3.13 not found`

The Dockerfiles use `python:3.13-slim` — Docker pulls this automatically. If you're building locally (without Docker), install Python 3.13 via pyenv or your OS package manager.

### Hetionet script fails: `Legacy Neo4j did not become ready`

The `dhimmel/hetionet` image can be slow to start (Neo4j 3.5 initialises its store). Increase the timeout by editing `setup-hetionet.sh`: change `seq 1 60` (3-minute timeout) to a higher value, e.g. `seq 1 100`.

### Fine-tuning stage fails

Fine-tuning requires GPU and the `INSTALL_FINETUNE=1` build arg. Without GPU, stages 1–2 (traversal + ChatML) work fully; stages 3–4 require a CUDA-capable host:

```bash
docker compose build --build-arg INSTALL_FINETUNE=1 api
```
