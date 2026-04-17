"""
Main runtime: run pipeline or a single stage from config.

Usage (CLI):
    # Run with default config (ChatML converter on sample data)
    python -m autodistil_kg.run

    # Run with custom config
    python -m autodistil_kg.run --config config/pipeline.json

    # Run single stage
    python -m autodistil_kg.run --stage chatml_converter --input data/chatml.jsonl --output data/prepared.jsonl

Usage (programmatic):
    from autodistil_kg.pipeline import Pipeline, PipelineConfig, PipelineContext
    from autodistil_kg.pipeline.config import ChatMLConverterStageConfig, FineTunerStageConfig

    config = PipelineConfig(chatml_converter=ChatMLConverterStageConfig(...))
    pipeline = Pipeline(config)
    result = pipeline.run_stage("chatml_converter", context=ctx)
"""
import argparse
import json
import logging
import os
from pathlib import Path

from .pipeline import Pipeline
from .pipeline.config import (
    PipelineConfig,
    GraphTraverserStageConfig,
    ChatMLConverterStageConfig,
    FineTunerStageConfig,
    EvaluatorStageConfig,
)
from .pipeline.interfaces import PipelineContext

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

# Default config paths (relative to project root / cwd)
DEFAULT_CONFIG_PATHS = ["config/default.json", "config/pipeline.json"]

# Inline default config when no file exists (runs ChatML converter on sample data)
_BUNDLED_DEFAULT_CONFIG = {
    "output_dir": "./output",
    "run_stages": ["chatml_converter"],
    "chatml_converter": {
        "input_path": "data/sample_chatml.jsonl",
        "output_path": "output/prepared.jsonl",
        "prepare_for_finetuning": True,
    },
}

_SAMPLE_CHATML = """{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What can you tell me about the paper 'Attention Is All You Need'?"}, {"role": "assistant", "content": "The paper 'Attention Is All You Need' (2017) introduced the Transformer architecture, which relies entirely on self-attention mechanisms and dispenses with recurrence and convolutions."}], "metadata": {"node_id": "paper_001", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "Who authored the Transformer paper?"}, {"role": "assistant", "content": "The Transformer was authored by Vaswani et al. from Google Brain and Google Research. Key authors include Ashish Vaswani, Noam Shazeer, and Niki Parmar."}], "metadata": {"node_id": "paper_002", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "Describe the BERT model and its main contribution."}, {"role": "assistant", "content": "BERT (Bidirectional Encoder Representations from Transformers) was introduced by Devlin et al. in 2018. It uses masked language modeling to pre-train a bidirectional Transformer, enabling context-aware representations."}], "metadata": {"node_id": "paper_003", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What is word2vec and why is it significant?"}, {"role": "assistant", "content": "Word2vec (Mikolov et al., 2013) learns dense vector representations of words from large text corpora using neural networks, enabling words with similar meanings to have similar vectors."}], "metadata": {"node_id": "paper_004", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "Explain the key idea behind ResNet."}, {"role": "assistant", "content": "ResNet (He et al., 2015) introduced skip connections that allow gradients to flow directly through the network, enabling training of very deep networks (100+ layers) without vanishing gradients."}], "metadata": {"node_id": "paper_005", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What is GPT and how does it differ from BERT?"}, {"role": "assistant", "content": "GPT uses a decoder-only Transformer with autoregressive language modeling. Unlike BERT's bidirectional encoding, GPT predicts the next token given previous context, making it well-suited for generation tasks."}], "metadata": {"node_id": "paper_006", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What are attention mechanisms in neural networks?"}, {"role": "assistant", "content": "Attention mechanisms allow a model to focus on relevant parts of the input. In Transformers, self-attention enables each token to attend to every other token and capture long-range dependencies."}], "metadata": {"node_id": "paper_007", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "Describe the LSTM architecture and its purpose."}, {"role": "assistant", "content": "LSTM (Hochreiter & Schmidhuber, 1997) uses gating mechanisms to control information flow and addresses the vanishing gradient problem in RNNs, enabling learning of long-term dependencies."}], "metadata": {"node_id": "paper_008", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What is the ImageNet dataset and why does it matter?"}, {"role": "assistant", "content": "ImageNet is a large-scale image dataset with millions of labeled images. The ImageNet challenge drove breakthroughs like AlexNet (2012) and the rise of deep learning in computer vision."}], "metadata": {"node_id": "paper_009", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What is contrastive learning?"}, {"role": "assistant", "content": "Contrastive learning trains representations by pulling similar samples together and pushing dissimilar ones apart. Methods like SimCLR and MoCo achieve strong unsupervised representation learning."}], "metadata": {"node_id": "paper_010", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "Explain transfer learning in deep learning."}, {"role": "assistant", "content": "Transfer learning reuses a model pre-trained on a large dataset for a different task. Fine-tuning pre-trained weights on a smaller target dataset achieves better performance with less data than training from scratch."}], "metadata": {"node_id": "paper_011", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What is RLHF?"}, {"role": "assistant", "content": "RLHF (Reinforcement Learning from Human Feedback) trains models using human preferences as a reward signal, aligning model behavior with human values."}], "metadata": {"node_id": "paper_012", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What are knowledge graphs and how are they used in NLP?"}, {"role": "assistant", "content": "Knowledge graphs represent entities and relationships as structured graphs. In NLP, they provide factual grounding, enable reasoning, support question answering, and augment language models with external knowledge."}], "metadata": {"node_id": "paper_013", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "Describe the architecture of a typical Transformer block."}, {"role": "assistant", "content": "A Transformer block consists of multi-head self-attention followed by a feed-forward network, each with residual connections and layer normalization."}], "metadata": {"node_id": "paper_014", "labels": ["Paper"], "depth": 1}}
{"messages": [{"role": "system", "content": "You are a helpful assistant with knowledge about academic publications."}, {"role": "user", "content": "What is instruction tuning?"}, {"role": "assistant", "content": "Instruction tuning fine-tunes language models on (instruction, response) pairs, teaching them to follow diverse user instructions and improving zero-shot generalization."}], "metadata": {"node_id": "paper_015", "labels": ["Paper"], "depth": 1}}
"""


def _ensure_sample_data(cwd: Path) -> Path:
    """Create data/sample_chatml.jsonl if missing. Returns path."""
    data_dir = cwd / "data"
    sample_path = data_dir / "sample_chatml.jsonl"
    if not sample_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(_SAMPLE_CHATML.strip(), encoding="utf-8")
        logger.info("Created sample data at %s", sample_path.relative_to(cwd))
    return sample_path


def _config_from_data(data: dict, base_dir: Path) -> PipelineConfig:
    """Build PipelineConfig from dict (used for bundled default)."""
    def _cc(d):
        if not d:
            return None
        return ChatMLConverterStageConfig(
            input_path=_resolve_path(d.get("input_path"), base_dir),
            output_path=_resolve_path(d.get("output_path"), base_dir),
            prepare_for_finetuning=d.get("prepare_for_finetuning", True),
            chat_template=d.get("chat_template"),
        )

    def _ft(d):
        if not d:
            return None
        return FineTunerStageConfig(
            model_name=d.get("model_name", "unsloth/gemma-2-2b-it"),
            model_type=d.get("model_type"),
            train_data_path=_resolve_path(d.get("train_data_path"), base_dir),
            eval_data_path=_resolve_path(d.get("eval_data_path"), base_dir),
            output_dir=_resolve_path(d.get("output_dir"), base_dir),
            max_seq_length=d.get("max_seq_length", 2048),
            num_train_epochs=d.get("num_train_epochs", 1),
            per_device_train_batch_size=d.get("per_device_train_batch_size", 2),
            learning_rate=d.get("learning_rate", 2e-4),
        )

    ev = data.get("evaluator")
    evaluator = None
    if ev:
        evaluator = EvaluatorStageConfig(
            model_path=_resolve_path(ev.get("model_path"), base_dir),
            eval_dataset_path=_resolve_path(ev.get("eval_dataset_path"), base_dir),
            output_report_path=_resolve_path(ev.get("output_report_path"), base_dir),
            metrics=ev.get("metrics"),
            additional_params=ev.get("additional_params", {}),
        )

    return PipelineConfig(
        graph_traverser=None,
        chatml_converter=_cc(data.get("chatml_converter")),
        finetuner=_ft(data.get("finetuner")),
        evaluator=evaluator,
        output_dir=_resolve_path(data.get("output_dir"), base_dir) if data.get("output_dir") else None,
        run_stages=data.get("run_stages"),
    )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Autodistil-KG: Build instruction datasets from knowledge graphs and fine-tune LLMs.",
        epilog="Examples:\n"
        "  %(prog)s                                    # Run with default config\n"
        "  %(prog)s --config config/my.json             # Custom config\n"
        "  %(prog)s --stage chatml_converter --input data/chatml.jsonl\n"
        "  %(prog)s --stage finetuner --input data/prepared.jsonl --output ./models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage",
        choices=["graph_traverser", "chatml_converter", "finetuner", "evaluator"],
        help="Run only this stage (requires --input)",
    )
    p.add_argument("--config", type=str, help="Path to JSON config file")
    p.add_argument("--input", type=str, help="Input path (JSONL for converter/finetuner)")
    p.add_argument("--output", type=str, help="Output path or directory")
    p.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gemma-2-2b-it",
        help="Model for finetuner (default: %(default)s)",
    )
    p.add_argument(
        "--model-type",
        type=str,
        default="gemma",
        help="Model type for finetuner (default: %(default)s)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return p.parse_args()


def _resolve_path(path: str | None, base: Path) -> str | None:
    """Resolve path relative to base if not absolute."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def _parse_graph_traverser_config(gt_data: dict, base: Path) -> GraphTraverserStageConfig:
    """
    Build GraphTraverserStageConfig from JSON. Uses .env for credentials (Neo4j, Redis, LLM).
    Requires: python-dotenv, .env file with NEO4J_*, REDIS_*, and LLM provider vars.
    """
    try:
        from .graph_traverser.env_config import (
            load_env_file,
            get_graph_db_config_from_env,
            get_llm_config_from_env,
            get_state_storage_config_from_env,
        )
        from .graph_traverser.config import TraversalConfig, DatasetGenerationConfig, TraversalStrategy
    except ImportError as e:
        raise ImportError(
            "Graph traverser requires python-dotenv and LLM SDKs (e.g. openai). "
            "Install with: poetry add python-dotenv openai"
        ) from e

    env_path = base / ".env"
    if env_path.exists():
        load_env_file(str(env_path))
    else:
        raise FileNotFoundError(
            f"Graph traverser requires .env at {env_path}. "
            "Copy example.env to .env and set NEO4J_*, REDIS_*, and LLM credentials."
        )

    # Traversal config from JSON
    t = gt_data.get("traversal") or {}
    strategy_str = (t.get("strategy") or "bfs").lower()
    try:
        strategy = TraversalStrategy(strategy_str)
    except ValueError:
        strategy = TraversalStrategy.BFS
    traversal = TraversalConfig(
        strategy=strategy,
        max_nodes=t.get("max_nodes", 500),
        max_depth=t.get("max_depth", 5),
        relationship_types=t.get("relationship_types"),
        node_labels=t.get("node_labels"),
        seed_node_ids=t.get("seed_node_ids"),
    )

    # Dataset config from JSON
    d = gt_data.get("dataset") or {}
    output_path = _resolve_path(gt_data.get("output_path") or d.get("output_path"), base)
    dataset = DatasetGenerationConfig(
        seed_prompts=d.get("seed_prompts", ["What can you tell me about this node? Describe: {properties}"]),
        system_message=d.get("system_message"),
        prompt_template=d.get("prompt_template"),
        include_metadata=d.get("include_metadata", True),
        output_format=d.get("output_format", "jsonl"),
        output_path=output_path,
    )

    return GraphTraverserStageConfig(
        graph_db=get_graph_db_config_from_env(),
        llm_config=get_llm_config_from_env(provider=gt_data.get("llm_provider")),
        state_storage=get_state_storage_config_from_env(),
        traversal=traversal,
        dataset=dataset,
        output_path=output_path,
    )


def _config_from_json(path: str, base_dir: Path | None = None) -> PipelineConfig:
    # Paths in config are relative to project root (cwd) for consistency
    base = base_dir or Path.cwd()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _cc(d):
        if not d:
            return None
        return ChatMLConverterStageConfig(
            input_path=_resolve_path(d.get("input_path"), base),
            output_path=_resolve_path(d.get("output_path"), base),
            prepare_for_finetuning=d.get("prepare_for_finetuning", True),
            chat_template=d.get("chat_template"),
        )

    def _ft(d):
        if not d:
            return None
        return FineTunerStageConfig(
            model_name=d.get("model_name", "unsloth/gemma-2-2b-it"),
            model_type=d.get("model_type"),
            train_data_path=_resolve_path(d.get("train_data_path"), base),
            eval_data_path=_resolve_path(d.get("eval_data_path"), base),
            output_dir=_resolve_path(d.get("output_dir"), base),
            max_seq_length=d.get("max_seq_length", 2048),
            num_train_epochs=d.get("num_train_epochs", 1),
            per_device_train_batch_size=d.get("per_device_train_batch_size", 2),
            learning_rate=d.get("learning_rate", 2e-4),
        )

    ev = data.get("evaluator")
    evaluator = None
    if ev:
        evaluator = EvaluatorStageConfig(
            model_path=_resolve_path(ev.get("model_path"), base),
            eval_dataset_path=_resolve_path(ev.get("eval_dataset_path"), base),
            output_report_path=_resolve_path(ev.get("output_report_path"), base),
            metrics=ev.get("metrics"),
            additional_params=ev.get("additional_params", {}),
        )

    gt_data = data.get("graph_traverser")
    graph_traverser = None
    if gt_data:
        try:
            graph_traverser = _parse_graph_traverser_config(gt_data, base)
        except Exception as e:
            logger.error("Failed to load graph_traverser config: %s", e)
            raise

    return PipelineConfig(
        graph_traverser=graph_traverser,
        chatml_converter=_cc(data.get("chatml_converter")),
        finetuner=_ft(data.get("finetuner")),
        evaluator=evaluator,
        output_dir=_resolve_path(data.get("output_dir"), base) if data.get("output_dir") else None,
        run_stages=data.get("run_stages"),
    )


def _find_default_config(cwd: Path) -> Path | None:
    """Find first existing default config in project."""
    for name in DEFAULT_CONFIG_PATHS:
        p = cwd / name
        if p.exists():
            return p
    return None


def _context_from_config(config: PipelineConfig) -> PipelineContext:
    """Build initial context from config stage inputs."""
    chatml_path = None
    prepared_path = None
    if config.chatml_converter:
        chatml_path = config.chatml_converter.input_path
        prepared_path = config.chatml_converter.output_path
    if config.finetuner:
        prepared_path = config.finetuner.train_data_path or prepared_path
    return PipelineContext(
        chatml_dataset_path=chatml_path,
        prepared_dataset_path=prepared_path,
    )


def main():
    args = _parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    elif (env_level := os.environ.get("LOG_LEVEL", "").upper()) in ("DEBUG", "INFO", "WARNING", "ERROR"):
        log_level = getattr(logging, env_level)
    if log_level == logging.DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("autodistil_kg").setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled (via -v or LOG_LEVEL=DEBUG)")

    cwd = Path.cwd()

    config: PipelineConfig
    used_config_path: str | None = None

    if args.config and Path(args.config).exists():
        used_config_path = args.config
        config = _config_from_json(args.config)
    elif args.stage == "graph_traverser":
        logger.error(
            "Graph traverser requires a config file with Neo4j, Redis, and LLM settings. "
            "Use: %s --config config/pipeline_full.json --stage graph_traverser",
            "python run.py" if Path("run.py").exists() else "python -m autodistil_kg.run",
        )
        raise SystemExit(1)
    elif args.stage == "chatml_converter" and args.input:
        config = PipelineConfig(
            chatml_converter=ChatMLConverterStageConfig(
                input_path=str(Path(args.input).resolve()),
                output_path=str(
                    Path(args.output).resolve()
                    if args.output
                    else Path(args.input).resolve().with_suffix(".prepared.jsonl")
                ),
                prepare_for_finetuning=True,
            ),
        )
    elif args.stage == "finetuner" and args.input:
        config = PipelineConfig(
            finetuner=FineTunerStageConfig(
                model_name=args.model_name,
                model_type=args.model_type,
                train_data_path=str(Path(args.input).resolve()),
                output_dir=str(Path(args.output or "./finetuner_output").resolve()),
            ),
        )
    else:
        # No args: try default config file, then bundled default
        default_cfg = _find_default_config(cwd)
        if default_cfg:
            used_config_path = str(default_cfg)
            config = _config_from_json(str(default_cfg), base_dir=cwd)
            logger.info("Using config: %s", default_cfg.relative_to(cwd))
        else:
            # Use bundled default: ensure sample data exists, build config from dict
            _ensure_sample_data(cwd)
            used_config_path = "(bundled default)"
            config = _config_from_data(_BUNDLED_DEFAULT_CONFIG, cwd)
            logger.info("Using bundled default config (ChatML converter on sample data)")

    pipeline = Pipeline(config)

    if args.stage:
        context = PipelineContext(
            chatml_dataset_path=args.input if args.stage != "graph_traverser" else None,
            prepared_dataset_path=args.input if args.stage != "graph_traverser" else None,
        )
        if args.stage not in pipeline.available_stages:
            logger.error(
                "Stage %s not configured. Available: %s",
                args.stage,
                pipeline.available_stages,
            )
            raise SystemExit(1)
        result = pipeline.run_stage(args.stage, context=context)
        if not result.success:
            logger.error("Stage failed: %s", result.error)
            raise SystemExit(1)
        logger.info("Stage %s completed: %s", args.stage, result.metadata)
        return

    context = _context_from_config(config)
    context, results = pipeline.run(context=context)

    for i, r in enumerate(results):
        if not r.success:
            stage_name = (
                pipeline.available_stages[i]
                if i < len(pipeline.available_stages)
                else "unknown"
            )
            logger.error("Pipeline failed at stage %s: %s", stage_name, r.error)
            raise SystemExit(1)

    logger.info("Pipeline completed successfully")
    if used_config_path:
        logger.info("Output: %s", context.prepared_dataset_path or context.model_output_path)
    logger.info("Context: %s", context.to_dict())
