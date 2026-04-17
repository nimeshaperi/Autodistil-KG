"""
Quality scoring and alignment filtering for generated Q&A pairs.

Provides LLM-based quality assessment (relevance, groundedness, completeness)
and reference text loading for alignment-guided generation.
"""
import logging
from typing import List, Dict, Optional, TYPE_CHECKING

from autodistil_kg.llm import LLMMessage
from .prompts import format_quality_scoring_prompt

if TYPE_CHECKING:
    from .config import AlignmentConfig
    from autodistil_kg.llm import LLMClient

logger = logging.getLogger(__name__)


def score_qa_quality(
    llm_client: "LLMClient",
    question: str,
    answer: str,
    reference_texts: str,
) -> Dict[str, float]:
    """Score a Q&A pair on relevance, groundedness, and completeness.

    Returns a dict with ``relevance``, ``groundedness``, ``completeness``
    (each 0-1) and ``avg`` (mean of the three).  On LLM failure returns
    all ones so the pair is kept.
    """
    prompt = format_quality_scoring_prompt(
        question, answer, reference=reference_texts,
    )
    messages = [LLMMessage(role="user", content=prompt)]
    try:
        response = llm_client.generate(
            messages, temperature=0.0, max_tokens=60,
        )
        scores: Dict[str, float] = {}
        for line in response.strip().splitlines():
            for dim in ("relevance", "groundedness", "completeness"):
                if line.lower().startswith(dim):
                    val = float(line.split(":", 1)[1].strip())
                    scores[dim] = val / 10.0  # normalise 0-10 -> 0-1
        if len(scores) < 3:
            raise ValueError(f"Could not parse all three scores: {response!r}")
        scores["avg"] = sum(scores.values()) / 3.0
        return scores
    except Exception as exc:
        logger.warning("Quality scoring failed, keeping pair: %s", exc)
        return {"relevance": 1.0, "groundedness": 1.0, "completeness": 1.0, "avg": 1.0}


def load_reference_texts(path_str: Optional[str]) -> str:
    """Load reference material from a file path.

    Supports plain-text (.txt) and JSONL (.jsonl) files.  For JSONL each
    line should have a ``"text"`` field.  Returns the concatenated text
    (capped at ~8 000 chars to keep prompts manageable) or an empty
    string when no path is configured.
    """
    import json as _json
    from pathlib import Path as _Path

    if not path_str:
        return ""
    path = _Path(path_str)
    if not path.exists():
        logger.warning("Reference texts file not found: %s", path)
        return ""

    max_chars = 8_000
    try:
        if path.suffix == ".jsonl":
            chunks: List[str] = []
            total = 0
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    obj = _json.loads(line)
                    text = obj.get("text", "")
                    if total + len(text) > max_chars:
                        break
                    chunks.append(text)
                    total += len(text)
            return "\n\n".join(chunks)
        else:
            text = path.read_text(encoding="utf-8")
            return text[:max_chars]
    except Exception as exc:
        logger.warning("Failed to load reference texts from %s: %s", path, exc)
        return ""


def build_system_message(
    base_system_message: Optional[str],
    alignment: "AlignmentConfig",
) -> Optional[str]:
    """Build an enhanced system message incorporating alignment context."""
    base = base_system_message or ""
    parts: List[str] = [base] if base else []
    if alignment.target_audience:
        parts.append(f"Your target audience is: {alignment.target_audience}.")
    if alignment.domain_focus:
        parts.append(f"Focus on the domain of: {alignment.domain_focus}.")
    return " ".join(parts) if parts else None


def rephrase_eval_questions(
    llm_client: "LLMClient",
    samples: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Rephrase eval questions using the LLM so they test comprehension.

    Each question is rewritten to ask about the same knowledge but with
    different wording, structure, and emphasis.  The reference answer is
    kept unchanged.  This prevents the evaluated model from scoring high
    simply by pattern-matching the exact phrasing it was trained on.
    """
    import re

    prompt_template = (
        "Rephrase the following question so it asks about the same "
        "knowledge but uses substantially different wording, sentence "
        "structure, and emphasis. The rephrased question should be "
        "equally specific and challenging. Return ONLY the rephrased "
        "question, nothing else.\n\n"
        "Original question: {question}\n\n"
        "Rephrased question:"
    )

    rephrased = []
    for i, sample in enumerate(samples):
        try:
            prompt = prompt_template.format(question=sample["question"])
            messages = [LLMMessage(role="user", content=prompt)]
            response = llm_client.generate(
                messages, temperature=0.7, max_tokens=512,
            )
            new_q = response.strip()
            # Strip reasoning traces if the model produces them
            new_q = re.sub(r"<think>.*?</think>", "", new_q, flags=re.DOTALL).strip()
            if "</think>" in new_q:
                new_q = new_q.split("</think>", 1)[1].strip()
            if new_q and len(new_q) > 20:
                rephrased.append({
                    "question": new_q,
                    "reference": sample["reference"],
                })
                logger.info(
                    "Rephrased eval Q %d/%d (orig: %.60s... -> new: %.60s...)",
                    i + 1, len(samples),
                    sample["question"], new_q,
                )
            else:
                # Fallback: keep original if rephrase failed
                rephrased.append(sample)
                logger.warning("Rephrase failed for eval Q %d -- keeping original", i + 1)
        except Exception as e:
            logger.warning("Rephrase error for eval Q %d: %s -- keeping original", i + 1, e)
            rephrased.append(sample)

    logger.info(
        "Rephrased %d/%d eval questions successfully",
        sum(1 for r, s in zip(rephrased, samples) if r["question"] != s["question"]),
        len(samples),
    )
    return rephrased
