"""
Utility functions for the graph traverser module.

Contains helper functions used across the traverser subsystem: event emission,
node ID formatting, and LLM reasoning-trace stripping.
"""
import logging
import re

logger = logging.getLogger(__name__)


def strip_reasoning_traces(text: str) -> str:
    """Remove LLM reasoning/thinking traces from generated text.

    Many reasoning models (Gemini 2.5, Qwen3, DeepSeek-R1, etc.) emit
    chain-of-thought blocks before the useful output.  These must be stripped
    before the text enters training data or evaluation datasets.
    """
    # 1. Strip <think>...</think> blocks (models like Qwen3 / DeepSeek-R1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Strip orphaned </think> — everything before it is reasoning that
    #    leaked without a matching opening <think> tag.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # 3. Strip leading "Thinking Process:" / "Thinking:" meta-reasoning blocks.
    #    These run until a known content marker: **Question:**, **Answer:**, or
    #    a markdown ## header that signals actual output has begun.
    text = re.sub(
        r"^[\s]*Thinking(?:\s+Process)?:\s*\n.*?"
        r"(?=\*\*Question:\*\*|\*\*Answer:\*\*|\n##\s)",
        "",
        text,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Aggressive fallback if no content markers found
    if re.match(r"^\s*Thinking(?:\s+Process)?:", text, re.IGNORECASE):
        text = re.sub(
            r"^[\s]*Thinking(?:\s+Process)?:\s*\n(?:.*\n)*?(?=\S)",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    # 4. Strip numbered meta-reasoning blocks (e.g. "1.  **Analyze the
    #    Request:**  ...  3.  **Drafting the Question:** ...") that precede
    #    the actual **Question:**/**Answer:** markers.
    text = re.sub(
        r"^\s*\d+\.\s+\*\*.*?(?=\*\*Question:\*\*|\*\*Answer:\*\*)",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # 5. Strip chat-template markers that leak through some providers
    text = re.sub(r"<\|im_start\|>.*?(?:\n|$)", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<start_of_turn>.*?(?:\n|$)", "", text)
    text = re.sub(r"<end_of_turn>", "", text)

    return text.strip()


def short_id(node_id: str, max_len: int = 24) -> str:
    """Shorten node ID for logging while keeping it unique.

    Neo4j element IDs look like ``4:<db-uuid>:<element-id>``.  The DB UUID
    is shared by every node in the same database, so naively truncating the
    front collapses all IDs into one string.  We extract the unique element
    ID suffix and prepend the DB prefix for context.
    """
    s = str(node_id)
    # Neo4j format: "N:uuid:element_id" — keep the unique element_id part
    parts = s.split(":")
    if len(parts) >= 3:
        prefix = parts[0]
        element_id = parts[-1]
        short = f"{prefix}:{element_id}"
        return short if len(short) <= max_len else f"{short[:max_len]}..."
    return f"{s[:max_len]}..." if len(s) > max_len else s


def emit_traversal_event(event_type: str, **data) -> None:
    """Emit a structured traversal progress event via the logging system.

    The API layer detects the ``traversal_event`` extra field and forwards it
    to WebSocket clients as a ``traversal_progress`` event so the UI can render
    live visualisations.
    """
    logger.info(
        "traversal:%s",
        event_type,
        extra={"traversal_event": {"type": event_type, **data}},
    )


# ------------------------------------------------------------------
# Backward-compatible aliases (underscore-prefixed names used in the
# original monolithic module).  These allow existing call-sites and
# any external code that imported the private names to keep working.
# ------------------------------------------------------------------
_strip_reasoning_traces = strip_reasoning_traces
_short_id = short_id
_emit_traversal_event = emit_traversal_event
