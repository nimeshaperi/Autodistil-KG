"""CLI entry point for Autodistil-KG Graph RAG."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .query_engine import GraphRAGEngine


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        logging.getLogger("neo4j").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autodistil-kg-graphrag",
        description="Graph RAG over a Neo4j knowledge graph using LlamaIndex.",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single question to answer (omit for interactive REPL).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to a .env file (default: auto-detect).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print results as JSON (useful for scripting).",
    )
    return parser


def _print_response(response, *, as_json: bool) -> None:
    if as_json:
        payload = {
            "answer": response.answer,
            "source_nodes": response.source_nodes,
            "metadata": response.metadata,
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"\n{'=' * 60}")
        print("Answer:")
        print(f"{'=' * 60}")
        print(response.answer)
        if response.source_nodes:
            print(f"\n{'-' * 60}")
            print(f"Sources ({len(response.source_nodes)}):")
            print(f"{'-' * 60}")
            for i, node in enumerate(response.source_nodes, 1):
                score = node.get("score")
                score_str = f"  (score: {score:.4f})" if score is not None else ""
                print(f"  [{i}]{score_str}  {node['text'][:120]}")
        print()


def _interactive(engine: GraphRAGEngine, *, as_json: bool) -> None:
    print("Autodistil-KG Graph RAG  (type 'exit' or Ctrl-D to quit)\n")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        response = engine.query(question)
        _print_response(response, as_json=as_json)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    env_path = Path(args.env) if args.env else None
    config = load_config(env_path)
    engine = GraphRAGEngine(config)

    print("Initialising Graph RAG engine...")
    engine.initialise()

    if args.query:
        response = engine.query(args.query)
        _print_response(response, as_json=args.json_output)
    else:
        _interactive(engine, as_json=args.json_output)


if __name__ == "__main__":
    main()
