"""Stage definitions endpoint."""
from fastapi import APIRouter
from typing import Any, Dict

from ..state import STAGE_ORDER, ARTIFACT_KEYS

router = APIRouter()


@router.get("/stages")
def get_stages() -> Dict[str, Any]:
    """Return pipeline stage definitions, config schemas, and available options."""
    return {
        "stage_order": list(STAGE_ORDER),
        "stages": {
            "graph_traverser": {
                "label": "Graph Traverser",
                "description": "Traverse a Neo4j knowledge graph and generate ChatML conversation datasets.",
                "config": {
                    "traversal": {
                        "strategies": [
                            {"value": "bfs", "label": "Breadth-First Search", "description": "Explore graph layer by layer from seed nodes."},
                            {"value": "dfs", "label": "Depth-First Search", "description": "Explore graph depth-first along each branch."},
                            {"value": "random", "label": "Random Walk", "description": "Randomly select neighbours at each step."},
                            {"value": "semantic", "label": "Semantic (LLM-guided)", "description": "LLM selects the most relevant neighbour at each step based on context."},
                            {"value": "reasoning", "label": "Reasoning (multi-hop)", "description": "Deep multi-hop reasoning with subgraph exploration and path analysis."},
                        ],
                        "fields": {
                            "max_nodes": {"type": "number", "default": 500, "label": "Max Nodes"},
                            "max_depth": {"type": "number", "default": 5, "label": "Max Depth"},
                            "reasoning_depth": {"type": "number", "default": 2, "label": "Reasoning Depth", "show_when_strategy": ["reasoning"]},
                            "max_paths_per_node": {"type": "number", "default": 15, "label": "Max Paths per Node", "show_when_strategy": ["reasoning"]},
                            "relationship_types": {"type": "string[]", "default": None, "label": "Relationship Types"},
                            "node_labels": {"type": "string[]", "default": None, "label": "Node Labels"},
                            "seed_node_ids": {"type": "string[]", "default": None, "label": "Seed Node IDs"},
                        },
                    },
                    "llm_providers": [
                        {"value": "openai", "label": "OpenAI", "fields": ["api_key", "model", "base_url"]},
                        {"value": "claude", "label": "Claude (Anthropic)", "fields": ["api_key", "model"]},
                        {"value": "gemini", "label": "Gemini (Google)", "fields": ["project_id", "location", "model", "credentials_path"]},
                        {"value": "ollama", "label": "Ollama (local)", "fields": ["base_url", "model"]},
                        {"value": "vllm", "label": "vLLM (local)", "fields": ["base_url", "model"]},
                    ],
                },
            },
            "chatml_converter": {
                "label": "ChatML Converter",
                "description": "Normalize and prepare ChatML datasets for fine-tuning.",
                "config": {
                    "fields": {
                        "input_path": {"type": "string", "default": "output/dataset.jsonl", "label": "Input Path", "hint": "From: Graph Traverser output"},
                        "output_path": {"type": "string", "default": "output/prepared.jsonl", "label": "Output Path", "hint": "Used by: FineTuner"},
                        "prepare_for_finetuning": {"type": "boolean", "default": True, "label": "Prepare for Fine-tuning"},
                        "chat_template": {"type": "string", "default": "auto", "label": "Chat Template"},
                    },
                },
            },
            "finetuner": {
                "label": "FineTuner",
                "description": "Fine-tune language models using Unsloth with LoRA adapters.",
                "config": {
                    "model_types": [
                        {"value": "gemma3", "label": "Gemma 3"},
                        {"value": "llama3", "label": "Llama 3"},
                        {"value": "qwen2", "label": "Qwen 2"},
                        {"value": "qwen3", "label": "Qwen 3"},
                        {"value": "mistral", "label": "Mistral"},
                        {"value": "phi3", "label": "Phi 3"},
                        {"value": "phi4", "label": "Phi 4"},
                    ],
                    "suggested_models": [
                        {"value": "unsloth/gemma-3-270m-it", "label": "Gemma 3 270M (instruction-tuned)", "type": "gemma3"},
                        {"value": "unsloth/gemma-3-1b-it", "label": "Gemma 3 1B (instruction-tuned)", "type": "gemma3"},
                        {"value": "unsloth/gemma-3-4b-it", "label": "Gemma 3 4B (instruction-tuned)", "type": "gemma3"},
                    ],
                    "fields": {
                        "model_name": {"type": "string", "label": "Model Name"},
                        "model_type": {"type": "string", "label": "Model Type"},
                        "train_data_path": {"type": "string", "default": "output/prepared.jsonl", "label": "Train Data Path", "hint": "From: ChatML Converter output"},
                        "output_dir": {"type": "string", "default": "output/finetuned", "label": "Output Directory", "hint": "Used by: Evaluator"},
                        "max_seq_length": {"type": "number", "default": 2048, "label": "Max Sequence Length"},
                        "num_train_epochs": {"type": "number", "default": 1, "label": "Epochs"},
                        "per_device_train_batch_size": {"type": "number", "default": 2, "label": "Batch Size"},
                        "learning_rate": {"type": "number", "default": 2e-4, "label": "Learning Rate"},
                    },
                },
            },
            "evaluator": {
                "label": "Evaluator",
                "description": "Compare finetuned model against base models and Graph RAG using configurable metrics.",
                "config": {
                    "modes": [
                        {"value": "internal", "label": "Internal (in-process)"},
                        {"value": "cli", "label": "CLI (external command)"},
                        {"value": "noop", "label": "No-op (stub report)"},
                    ],
                    "metrics": [
                        {"value": "answer_relevancy", "label": "Answer Relevancy (DeepEval)"},
                        {"value": "correctness", "label": "Correctness (DeepEval G-Eval)"},
                        {"value": "faithfulness", "label": "Faithfulness (DeepEval)"},
                        {"value": "hallucination", "label": "Hallucination (DeepEval)"},
                        {"value": "grounding", "label": "Grounding (DeepEval G-Eval)"},
                    ],
                    "system_kinds": [
                        {"value": "distilled", "label": "Finetuned Model"},
                        {"value": "base", "label": "Base Model"},
                        {"value": "graph_rag", "label": "Graph RAG"},
                        {"value": "external", "label": "External Model"},
                    ],
                    "llm_providers": [
                        {"value": "openai", "label": "OpenAI"},
                        {"value": "claude", "label": "Claude"},
                        {"value": "gemini", "label": "Gemini"},
                        {"value": "ollama", "label": "Ollama"},
                        {"value": "vllm", "label": "vLLM"},
                    ],
                    "fields": {
                        "eval_dataset_path": {"type": "string", "default": "output/prepared_eval.jsonl", "label": "Eval Dataset Path", "hint": "From: ChatML Converter (clean Q&A pairs)"},
                        "output_report_path": {"type": "string", "default": "output/eval_report.json", "label": "Output Report Path"},
                        "model_path": {"type": "string", "default": "output/finetuned", "label": "Finetuned Model Path", "hint": "From: FineTuner output"},
                        "max_eval_samples": {"type": "number", "label": "Max Eval Samples (optional)"},
                    },
                },
            },
        },
        "artifact_keys": list(ARTIFACT_KEYS.keys()),
    }
