"""
Q&A generation logic for the Graph Traverser Agent.

Handles all LLM interactions for path reasoning, subgraph synthesis, and
question-answer pair generation from knowledge graph subgraphs.
"""
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from autodistil_kg.llm import LLMMessage
from .prompts import (
    format_path_reasoning_prompt,
    format_batched_path_reasoning_prompt,
    format_subgraph_synthesis_prompt,
    format_reasoning_qa_prompt,
    format_synthesis_qa_combined_prompt,
)
from .utils import strip_reasoning_traces

if TYPE_CHECKING:
    from .config import AlignmentConfig
    from autodistil_kg.llm import LLMClient

logger = logging.getLogger(__name__)


def reason_through_path(
    llm_client: "LLMClient",
    center_node: Dict[str, Any],
    path: List[Dict[str, Any]],
    system_message: Optional[str],
    alignment: "AlignmentConfig",
    reference_texts: str,
) -> Optional[str]:
    """Use LLM to reason through a single path from the subgraph.

    Returns the LLM's step-by-step reasoning analysis.
    """
    prompt = format_path_reasoning_prompt(
        center_node, path,
        alignment=alignment,
        reference_excerpts=reference_texts,
    )
    messages = []

    if system_message:
        messages.append(LLMMessage(
            role="system",
            content="You are a knowledge graph reasoning engine. Analyze paths deeply and extract multi-step knowledge.",
        ))

    messages.append(LLMMessage(role="user", content=prompt))

    logger.debug("[LLM_CALL] path_reasoning | prompt_len=%d | max_tokens=800", len(prompt))
    try:
        response = llm_client.generate(
            messages, temperature=0.4, max_tokens=800
        )
        logger.debug("[LLM_RESP] path_reasoning | response_len=%d | preview=%.200s", len(response), response)
        return strip_reasoning_traces(response)
    except Exception as e:
        logger.warning("Error reasoning through path: %s", e)
        return None


def reason_through_paths_batch(
    llm_client: "LLMClient",
    center_node: Dict[str, Any],
    paths: List[List[Dict[str, Any]]],
    system_message: Optional[str],
    alignment: "AlignmentConfig",
    reference_texts: str,
) -> List[str]:
    """Reason through multiple paths in a single LLM call.

    Returns list of analyses (one per path, skipping empty ones).
    """
    prompt = format_batched_path_reasoning_prompt(
        center_node, paths,
        alignment=alignment,
        reference_excerpts=reference_texts,
    )
    messages = []

    if system_message:
        messages.append(LLMMessage(
            role="system",
            content="You are a knowledge graph reasoning engine. Analyze paths deeply and extract multi-step knowledge.",
        ))

    messages.append(LLMMessage(role="user", content=prompt))

    # Scale max_tokens with batch size
    max_tokens = min(800 * len(paths), 4096)
    logger.debug(
        "[LLM_CALL] batched_path_reasoning | paths=%d | prompt_len=%d | max_tokens=%d",
        len(paths), len(prompt), max_tokens,
    )
    try:
        response = llm_client.generate(
            messages, temperature=0.4, max_tokens=max_tokens
        )
        logger.debug(
            "[LLM_RESP] batched_path_reasoning | response_len=%d | preview=%.200s",
            len(response), response,
        )
        # Split by separator
        raw_parts = response.split("---PATH_SEPARATOR---")
        analyses = [strip_reasoning_traces(part) for part in raw_parts if part.strip()]
        logger.debug(
            "[LLM_RESP] batched_path_reasoning | expected=%d | got=%d analyses",
            len(paths), len(analyses),
        )
        return analyses
    except Exception as e:
        logger.warning("Error in batched path reasoning: %s", e)
        return []


def synthesize_subgraph(
    llm_client: "LLMClient",
    center_node: Dict[str, Any],
    path_analyses: List[str],
    num_nodes: int,
    num_edges: int,
    system_message: Optional[str],
    alignment: "AlignmentConfig",
    reference_texts: str,
) -> str:
    """Synthesize multiple path-level analyses into a comprehensive understanding."""
    prompt = format_subgraph_synthesis_prompt(
        center_node, path_analyses, num_nodes, num_edges,
        alignment=alignment,
        reference_excerpts=reference_texts,
    )
    messages = []

    if system_message:
        messages.append(LLMMessage(
            role="system",
            content="You are a knowledge synthesis engine. Combine multiple analyses into comprehensive, educational summaries.",
        ))

    messages.append(LLMMessage(role="user", content=prompt))

    logger.debug("[LLM_CALL] synthesis | prompt_len=%d | max_tokens=3000", len(prompt))
    try:
        response = llm_client.generate(
            messages, temperature=0.5, max_tokens=3000
        )
        logger.debug("[LLM_RESP] synthesis | response_len=%d | preview=%.200s", len(response), response)
        return strip_reasoning_traces(response)
    except Exception as e:
        logger.error("Error synthesizing subgraph: %s", e)
        return "Error during synthesis: " + str(e)


def synthesize_and_generate_qa(
    llm_client: "LLMClient",
    center_node: Dict[str, Any],
    path_analyses: List[str],
    num_nodes: int,
    num_edges: int,
    system_message: Optional[str],
    alignment: "AlignmentConfig",
    reference_texts: str,
) -> Dict[str, Any]:
    """Synthesize path analyses AND generate a QA pair in one LLM call.

    Returns dict with 'synthesis' (str) and 'qa' (dict with 'question'/'answer').
    """
    prompt = format_synthesis_qa_combined_prompt(
        center_node, path_analyses, num_nodes, num_edges,
        alignment=alignment,
        reference_excerpts=reference_texts,
    )
    messages = []

    if system_message:
        messages.append(LLMMessage(
            role="system",
            content="You are a knowledge synthesis engine. Combine analyses into comprehensive summaries and generate training data.",
        ))

    messages.append(LLMMessage(role="user", content=prompt))

    logger.debug("[LLM_CALL] synthesis_qa_combined | prompt_len=%d | max_tokens=4096", len(prompt))
    try:
        response = llm_client.generate(
            messages, temperature=0.5, max_tokens=4096
        )
        logger.debug(
            "[LLM_RESP] synthesis_qa_combined | response_len=%d | preview=%.200s",
            len(response), response,
        )
        response = strip_reasoning_traces(response)
        return parse_combined_response(center_node, response)
    except Exception as e:
        logger.error("Error in combined synthesis+QA: %s", e)
        return {
            "synthesis": "Error during synthesis: " + str(e),
            "qa": {"question": "", "answer": ""},
        }


def parse_combined_response(
    center_node: Dict[str, Any],
    response: str,
) -> Dict[str, Any]:
    """Parse the combined synthesis+QA response into its two parts."""
    synthesis = ""
    question = ""
    answer = ""

    # Split on the ## QA header to separate synthesis from QA
    if "## QA" in response:
        parts = response.split("## QA", 1)
        synthesis = parts[0]
        qa_text = parts[1]
    elif "## Synthesis" in response:
        # Has synthesis header but no QA header — whole thing is synthesis
        synthesis = response
        qa_text = ""
    else:
        # No headers — treat the whole response as synthesis
        synthesis = response
        qa_text = ""

    # Clean the synthesis: strip the "## Synthesis" header if present
    synthesis = synthesis.replace("## Synthesis", "").strip()

    # Parse QA from qa_text
    if qa_text:
        if "**Question:**" in qa_text and "**Answer:**" in qa_text:
            qa_parts = qa_text.split("**Answer:**", 1)
            question = qa_parts[0].replace("**Question:**", "").strip()
            answer = qa_parts[1].strip()
        elif "Question:" in qa_text and "Answer:" in qa_text:
            qa_parts = qa_text.split("Answer:", 1)
            question = qa_parts[0].replace("Question:", "").strip()
            answer = qa_parts[1].strip()

    # Fallback if QA parsing failed
    if not question:
        name = (
            center_node.get("properties", {}).get("name")
            or center_node.get("properties", {}).get("title")
            or ", ".join(center_node.get("labels", ["this entity"]))
        )
        question = f"What can you tell me about {name} and its relationships, including any multi-step inferences?"
    if not answer and synthesis:
        answer = synthesis

    return {
        "synthesis": synthesis,
        "qa": {"question": question, "answer": answer},
    }


def generate_reasoning_qa(
    llm_client: "LLMClient",
    center_node: Dict[str, Any],
    synthesis: str,
    system_message: Optional[str],
    alignment: "AlignmentConfig",
    reference_texts: str,
) -> Dict[str, str]:
    """Generate a high-quality QA pair from the synthesis, suitable for SLM distillation.

    Returns dict with 'question' and 'answer' keys.
    """
    prompt = format_reasoning_qa_prompt(
        center_node, synthesis,
        alignment=alignment,
        reference_excerpts=reference_texts,
    )
    messages = []

    if system_message:
        messages.append(LLMMessage(
            role="system",
            content="You are a training data generator. Create high-quality question-answer pairs for language model training.",
        ))

    messages.append(LLMMessage(role="user", content=prompt))

    logger.debug("[LLM_CALL] qa_generation | prompt_len=%d | max_tokens=2000", len(prompt))
    try:
        response = llm_client.generate(
            messages, temperature=0.6, max_tokens=2000
        )
        logger.debug("[LLM_RESP] qa_generation | response_len=%d | preview=%.200s", len(response), response)

        # Strip reasoning/thinking traces before parsing Q&A markers
        response = strip_reasoning_traces(response)

        # Parse the QA pair from the response
        question = ""
        answer = ""

        if "**Question:**" in response and "**Answer:**" in response:
            parts = response.split("**Answer:**", 1)
            question = parts[0].replace("**Question:**", "").strip()
            answer = parts[1].strip()
        elif "Question:" in response and "Answer:" in response:
            parts = response.split("Answer:", 1)
            question = parts[0].replace("Question:", "").strip()
            answer = parts[1].strip()
        else:
            # Fallback: use the synthesis as answer and generate a simple question
            name = (
                center_node.get("properties", {}).get("name")
                or center_node.get("properties", {}).get("title")
                or ", ".join(center_node.get("labels", ["this entity"]))
            )
            question = f"What can you tell me about {name} and its relationships, including any multi-step inferences?"
            answer = response

        return {"question": question, "answer": answer}

    except Exception as e:
        logger.error("Error generating QA pair: %s", e)
        name = center_node.get("properties", {}).get("name", "this entity")
        return {
            "question": f"What do you know about {name}?",
            "answer": synthesis,
        }
