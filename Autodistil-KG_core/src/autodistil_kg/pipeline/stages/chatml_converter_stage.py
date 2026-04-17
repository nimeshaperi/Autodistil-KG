"""
ChatML Converter stage: normalize / prepare CHATML for finetuning.
Loads from context or path; outputs prepared JSONL or in-memory dataset.
Also generates a separate eval dataset (clean Q&A pairs) for evaluation.
Runnable standalone (e.g. load from file) or after graph traverser.
"""
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..interfaces import Stage, StageResult, PipelineContext
from ..config import ChatMLConverterStageConfig
from ...chatml import ChatMLDataset, ChatMLConversation, ChatMLMessage

logger = logging.getLogger(__name__)


def _strip_reasoning_traces(text: str) -> str:
    """Strip LLM reasoning traces for clean eval references."""
    # Strip leading "..." continuation artifacts from truncated LLM output
    text = re.sub(r"^\.{2,}\s*", "", text)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(
        r"^[\s]*Thinking(?:\s+Process)?:\s*\n.*?"
        r"(?=\*\*Question:\*\*|\*\*Answer:\*\*|\n##\s)",
        "", text, count=1, flags=re.DOTALL | re.IGNORECASE,
    )
    if re.match(r"^\s*Thinking(?:\s+Process)?:", text, re.IGNORECASE):
        text = re.sub(
            r"^[\s]*Thinking(?:\s+Process)?:\s*\n(?:.*\n)*?(?=\S)",
            "", text, count=1, flags=re.IGNORECASE,
        )
    text = re.sub(
        r"^\s*\d+\.\s+\*\*.*?(?=\*\*Question:\*\*|\*\*Answer:\*\*)",
        "", text, count=1, flags=re.DOTALL,
    )
    text = re.sub(r"<\|im_start\|>.*?(?:\n|$)", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<start_of_turn>.*?(?:\n|$)", "", text)
    text = re.sub(r"<end_of_turn>", "", text)
    return text.strip()


class ChatMLConverterStage(Stage):
    """Stage that converts/normalizes CHATML and optionally prepares for finetuning."""

    name = "chatml_converter"  # matches StageId.CHATML_CONVERTER

    def __init__(self, config: ChatMLConverterStageConfig):
        self.config = config

    def _load_dataset(self, context: PipelineContext) -> Optional[ChatMLDataset]:
        if context.chatml_dataset is not None:
            return context.chatml_dataset
        path = self.config.input_path or context.chatml_dataset_path
        if not path or not Path(path).exists():
            return None
        dataset = ChatMLDataset()
        dataset.load_jsonl(path)
        return dataset

    def _conversation_to_messages_dict(self, conv: ChatMLConversation) -> Dict[str, Any]:
        return {
            "messages": [{"role": m.role, "content": m.content} for m in conv.messages],
            **({"metadata": conv.metadata} if conv.metadata else {}),
        }

    @staticmethod
    def _is_valid_eval_text(text: str, min_length: int = 20) -> bool:
        """Check if text is a usable eval sample (not truncated garbage)."""
        if not text or len(text) < min_length:
            return False
        # Reject literal "..." artifacts from truncated LLM output
        if text.strip().startswith("..."):
            return False
        # Reject if it's mostly numbered reasoning steps (meta-thinking)
        if re.match(r"^\s*\d+\.\s+\*\*", text):
            return False
        return True

    def _build_eval_dataset(
        self, dataset: ChatMLDataset, eval_path: str,
        sample_ratio: float = 0.2,
    ) -> int:
        """Extract clean Q&A pairs from the dataset and write an eval JSONL.

        Skips synthesis-type conversations (metadata.type == "synthesis") since
        those are supplementary training data, not standalone Q&A pairs.
        Also skips broken records where the LLM output was truncated or
        consists entirely of reasoning traces.

        Only a random subset (default 20%) of valid samples is written to
        keep evaluation fast and representative.

        Returns the number of eval samples written.
        """
        # Collect all valid samples first
        all_samples: List[Dict[str, Any]] = []
        for conv in dataset.conversations:
            # Skip synthesis training conversations
            if conv.metadata and conv.metadata.get("type") == "synthesis":
                continue

            question = ""
            reference = ""
            for msg in conv.messages:
                if msg.role == "user":
                    question = _strip_reasoning_traces(msg.content)
                elif msg.role == "assistant":
                    reference = _strip_reasoning_traces(msg.content)

            # Skip broken/truncated/empty records
            if not self._is_valid_eval_text(question) or not self._is_valid_eval_text(reference):
                continue

            all_samples.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reference},
                ]
            })

        # Randomly sample a subset for evaluation
        if sample_ratio < 1.0 and len(all_samples) > 10:
            k = max(10, int(len(all_samples) * sample_ratio))
            selected = random.sample(all_samples, min(k, len(all_samples)))
            logger.info(
                "Sampled %d eval samples from %d valid (%.0f%%)",
                len(selected), len(all_samples), sample_ratio * 100,
            )
        else:
            selected = all_samples

        Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            for record in selected:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Wrote %d eval samples to %s", len(selected), eval_path)
        return len(selected)

    def run(self, context: PipelineContext) -> StageResult:
        try:
            dataset = self._load_dataset(context)
            if dataset is None or len(dataset) == 0:
                return StageResult(
                    success=False,
                    error="No CHATML dataset in context and no valid input_path.",
                )

            if not self.config.prepare_for_finetuning:
                output_path = self.config.output_path or context.prepared_dataset_path
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    dataset.save_jsonl(output_path)
                    context.prepared_dataset_path = output_path
                context.chatml_dataset = dataset
                return StageResult(
                    success=True,
                    output=dataset,
                    metadata={"conversations_count": len(dataset)},
                )

            # Prepare for finetuning: output list of dicts with "messages" key (and optionally "text" if tokenizer applied elsewhere)
            prepared: List[Dict[str, Any]] = []
            for conv in dataset.conversations:
                prepared.append(self._conversation_to_messages_dict(conv))

            output_path = self.config.output_path or context.prepared_dataset_path
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    for item in prepared:
                        f.write(json.dumps(item) + "\n")
                context.prepared_dataset_path = str(path)
                logger.info("Saved prepared dataset to %s", path)

            context.prepared_messages = prepared

            # Generate eval dataset (clean Q&A pairs) as a sibling file.
            # Skip if the graph traverser already produced one (dataset_eval.jsonl)
            # — that version uses controlled sampling and optional rephrasing.
            eval_count = 0
            if output_path and not context.eval_dataset_path:
                p = Path(output_path)
                eval_path = str(p.parent / f"{p.stem}_eval{p.suffix}")
                eval_count = self._build_eval_dataset(dataset, eval_path)
                if eval_count > 0:
                    context.eval_dataset_path = eval_path
            elif context.eval_dataset_path:
                logger.info(
                    "Skipping eval dataset generation — using existing %s from graph traverser",
                    context.eval_dataset_path,
                )

            return StageResult(
                success=True,
                output=prepared,
                metadata={
                    "conversations_count": len(prepared),
                    "eval_samples_count": eval_count,
                },
            )
        except Exception as e:
            logger.exception("ChatML converter stage failed")
            return StageResult(success=False, error=str(e))
