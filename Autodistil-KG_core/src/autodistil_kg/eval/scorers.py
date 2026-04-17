"""
Scorers: compute metrics comparing predictions against reference answers.

Uses DeepEval (Confident AI) as the standardised evaluation framework for
academically defensible LLM-as-judge scoring.  Available metrics:

  - AnswerRelevancyMetric — statement-level relevance scoring
  - GEval — G-Eval protocol (Liu et al., 2023) for correctness & grounding
  - FaithfulnessMetric — claim-level faithfulness against reference
  - HallucinationMetric — context contradiction detection

Works with any LLM provider supported by the project's LLM factory via a
DeepEvalBaseLLM adapter (OpenAI, vLLM, Ollama, Claude, Gemini).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Limits concurrent judge LLM calls to avoid burst-saturating rate-limited APIs.
# Free-tier providers (e.g. openrouter free models) allow only 16-20 req/min;
# running 8+ scorer threads simultaneously exhausts the quota instantly.
_JUDGE_SEMAPHORE = threading.Semaphore(2)


# ---------------------------------------------------------------------------
#  Abstract base
# ---------------------------------------------------------------------------

class Scorer(ABC):
    """Abstract base for a metric scorer."""

    name: str = "base"

    @abstractmethod
    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, Optional[float]]:
        """Score a single prediction against a reference answer.

        Returns a dict mapping metric keys to scores.  A score of ``None``
        indicates the metric could not be computed (e.g. judge LLM failure).
        """


# ---------------------------------------------------------------------------
#  Project LLM factory helpers (used by the DeepEval adapter)
# ---------------------------------------------------------------------------

def _create_judge_client(judge_config: Dict[str, Any]):
    """Create an LLMClient from judge_config using the project's LLM factory."""
    from ..llm.config import LLMConfig
    from ..llm.factory import create_llm_client

    provider = judge_config.get("provider", "openai")
    api_key = judge_config.get("api_key") or None
    base_url = judge_config.get("base_url") or None
    model = judge_config.get("model") or None

    # vLLM/Ollama/openai_compatible don't require an API key
    if provider in ("vllm", "ollama", "openai_compatible") and not api_key:
        api_key = "not-needed"

    config = LLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    return create_llm_client(config)


def _strip_thinking_traces(text: str) -> str:
    """Remove LLM reasoning/thinking traces so we can parse the actual content."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _parse_json_from_llm(text: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM output, tolerating markdown fences and thinking traces."""
    text = _strip_thinking_traces(text)

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Salvage a truncated "verdicts" array — DeepEval faithfulness responses can
    # be cut off mid-stream when the claims list is long, leaving the outer JSON
    # object unclosed.  Extract whatever complete verdict objects are present so
    # we can score with partial evidence rather than failing and retrying.
    verdict_objects = re.findall(
        r'\{\s*"verdict"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL
    )
    if verdict_objects:
        try:
            verdicts = [json.loads(v) for v in verdict_objects]
            logger.debug(
                "Salvaged %d verdict(s) from truncated JSON response", len(verdicts)
            )
            return {"verdicts": verdicts}
        except json.JSONDecodeError:
            pass

    kv_matches = re.findall(r'"?(\w+)"?\s*[:=]\s*([\d.]+)', text)
    if kv_matches:
        result = {}
        for key, val in kv_matches:
            try:
                result[key] = float(val)
            except ValueError:
                continue
        if result:
            return result

    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
#  DeepEval LLM adapter — wraps the project's LLMClient for DeepEval metrics
# ---------------------------------------------------------------------------

def _make_deepeval_adapter(judge_config: Dict[str, Any]):
    """Create a DeepEvalBaseLLM adapter wrapping the project's LLM factory.

    Uses a factory function so the DeepEval import is deferred until needed.
    """
    from deepeval.models import DeepEvalBaseLLM

    class _Adapter(DeepEvalBaseLLM):
        """Bridges the project's LLM factory to DeepEval metrics."""

        def __init__(self, config: Dict[str, Any]) -> None:
            self._client = _create_judge_client(config)
            self._model_name = config.get("model") or config.get("provider", "unknown")
            self._max_tokens = int(config.get("max_tokens") or 4096)
            super().__init__()

        def load_model(self):
            return self._client

        def generate(self, prompt: str, schema=None):
            from ..llm.interface import LLMMessage
            messages = [LLMMessage(role="user", content=prompt)]
            response = self._client.generate(
                messages, temperature=0.0, max_tokens=self._max_tokens,
            )
            response = _strip_thinking_traces(response)
            if schema is not None:
                try:
                    parsed = _parse_json_from_llm(response)
                    return schema(**parsed)
                except Exception:
                    logger.debug("JSON confinement parse failed, retrying with nudge")
                    messages = [
                        LLMMessage(
                            role="system",
                            content="You MUST respond with ONLY a valid JSON object. "
                                    "No explanation, no thinking, no markdown — just the JSON.",
                        ),
                        LLMMessage(role="user", content=prompt),
                    ]
                    response = self._client.generate(
                        messages, temperature=0.0, max_tokens=self._max_tokens,
                    )
                    response = _strip_thinking_traces(response)
                    parsed = _parse_json_from_llm(response)
                    return schema(**parsed)
            return response

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return self._model_name

    return _Adapter(judge_config)


# ---------------------------------------------------------------------------
#  DeepEval metric wrapper — adapts a DeepEval metric to our Scorer interface
# ---------------------------------------------------------------------------

class DeepEvalScorer(Scorer):
    """Wraps a single DeepEval metric as a project Scorer.

    Handles mapping our (prediction, reference, question) interface to
    DeepEval's LLMTestCase with the appropriate field mapping.

    Each call to ``score()`` creates a **fresh metric instance** from
    the stored factory callable.  This is essential for thread-safety
    because DeepEval metrics store mutable state (``self.score``,
    ``self.reason``, etc.) that would be corrupted by concurrent access.

    Retries failed scoring attempts up to ``max_retries`` times with
    exponential back-off.  If all attempts fail the score is reported
    as ``None`` (serialised as JSON ``null``) and ``_failed`` is set
    to ``True`` so downstream consumers can distinguish genuine zeros
    from evaluation failures.
    """

    MAX_RETRIES = 5
    MAX_RETRIES_PARSE = 1   # non-rate-limit parse errors are unlikely to self-heal
    RETRY_BACKOFF_BASE = 2  # seconds — used as fallback when no reset header found

    def __init__(
        self,
        metric_name: str,
        deepeval_metric: Any,
        score_key: str,
        reference_field: str,
        invert_score: bool = False,
        metric_factory: Optional[Any] = None,
    ) -> None:
        self.name = metric_name
        self._metric = deepeval_metric  # used only when no factory
        self._metric_factory = metric_factory
        self._score_key = score_key
        self._reference_field = reference_field  # 'expected_output', 'context', 'retrieval_context'
        self._invert_score = invert_score

    def _new_metric(self) -> Any:
        """Return a fresh metric instance for thread-safe scoring."""
        if self._metric_factory is not None:
            return self._metric_factory()
        return self._metric

    @staticmethod
    def _backoff_seconds(error: Exception) -> float:
        """Return how long to wait for a failed judge call.

        For HTTP 429 responses the provider includes an ``X-RateLimit-Reset``
        header (millisecond epoch) that tells us exactly when the quota window
        resets.  We parse it from the stringified exception so we sleep for
        precisely the right amount of time instead of blindly retrying every
        1-2 seconds into a still-exhausted quota.

        Falls back to 65 s (just over a per-minute window) when a 429 is
        detected but no reset timestamp can be parsed, and to a short
        exponential back-off for non-rate-limit errors.
        """
        err_str = str(error)
        # Parse millisecond reset timestamp from OpenRouter / OpenAI 429 bodies.
        match = re.search(r"['\"]?X-RateLimit-Reset['\"]?\s*:\s*['\"]?(\d{10,13})['\"]?", err_str)
        if match:
            reset_ms = int(match.group(1))
            # Normalise: timestamps <= 1e12 are already in seconds, else ms
            reset_sec = reset_ms / 1000.0 if reset_ms > 1e12 else float(reset_ms)
            wait = reset_sec - time.time() + 2.0  # +2 s buffer
            if 0 < wait < 300:  # sanity-check: between 0 and 5 minutes
                return wait
        if "429" in err_str or "rate limit" in err_str.lower():
            return 65.0  # safe default: just over a per-minute window
        return 0.0  # non-rate-limit error — caller uses exponential back-off

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, Optional[float]]:
        from deepeval.test_case import LLMTestCase

        kwargs: Dict[str, Any] = {
            "input": question,
            "actual_output": prediction,
        }
        if self._reference_field == "expected_output":
            kwargs["expected_output"] = reference
        elif self._reference_field == "context":
            kwargs["context"] = [reference]
        elif self._reference_field == "retrieval_context":
            kwargs["retrieval_context"] = [reference]
            kwargs["expected_output"] = reference

        test_case = LLMTestCase(**kwargs)
        if test_case.retrieval_context is None and self._reference_field == "retrieval_context":
            logger.error(
                "BUG: retrieval_context is None despite setting it. "
                "reference_field=%s, reference_len=%d, kwargs_keys=%s",
                self._reference_field, len(reference) if reference else -1,
                list(kwargs.keys()),
            )
        last_error: Optional[Exception] = None
        metric = self._new_metric()

        for attempt in range(1 + self.MAX_RETRIES):
            try:
                with _JUDGE_SEMAPHORE:
                    metric.measure(test_case)
                raw = float(metric.score)
                if self._invert_score:
                    raw = 1.0 - raw
                return {self._score_key: round(min(max(raw, 0.0), 1.0), 4)}
            except Exception as e:
                last_error = e
                rate_wait = self._backoff_seconds(e)
                is_rate_limit = rate_wait > 0
                max_attempts = 1 + self.MAX_RETRIES if is_rate_limit else 1 + self.MAX_RETRIES_PARSE
                if attempt < max_attempts - 1:
                    wait = rate_wait if is_rate_limit else min(self.RETRY_BACKOFF_BASE, 2.0)
                    logger.warning(
                        "DeepEval %s scorer failed (attempt %d/%d) for '%.60s…': %s — retrying in %.0fs",
                        self.name, attempt + 1, max_attempts,
                        question, e, wait,
                    )
                    time.sleep(wait)
                    metric = self._new_metric()  # fresh instance for retry
                else:
                    break  # give up — logged below

        logger.error(
            "DeepEval %s scorer failed after %d attempts for '%.60s…': %s",
            self.name, 1 + self.MAX_RETRIES, question, last_error,
        )
        return {self._score_key: None, f"{self._score_key}_failed": True}


# ---------------------------------------------------------------------------
#  G-Eval criteria and evaluation steps for custom dimensions
# ---------------------------------------------------------------------------

# Correctness sub-dimensions — each becomes a separate GEval instance
CORRECTNESS_GEVAL_DEFS = [
    (
        "judge_accuracy",
        "Factual accuracy of the prediction compared to the reference answer. "
        "All major claims should match the reference with no factual errors. "
        "Specific entity names (drugs, genes, proteins, diseases) that appear in "
        "the prediction but NOT in the reference should reduce the accuracy score.",
        [
            "Identify all major factual claims in the prediction (entities, relationships, mechanisms, quantities).",
            "Cross-reference each claim against the reference answer.",
            "Mark each claim as CORRECT (matches reference), INCORRECT (contradicts reference), or UNVERIFIABLE (not in reference).",
            "Pay special attention to specific entity names — if the prediction names entities absent from the reference as key claims, penalise accuracy.",
            "Calculate the proportion of correct claims relative to total claims.",
        ],
    ),
    (
        "judge_completeness",
        "Completeness of the prediction's coverage of the reference answer. "
        "All major themes, entities, and relationships from the reference should "
        "be addressed in the prediction.",
        [
            "Identify all key themes, entities, and relationships in the reference answer.",
            "Check which of these key points appear in the prediction.",
            "Determine whether major themes are missing entirely or only partially covered.",
            "Assess whether secondary points are also addressed or only the main points.",
            "Calculate coverage as the proportion of reference key points addressed.",
        ],
    ),
    (
        "judge_relevance",
        "Relevance of the prediction to the question asked. The prediction "
        "should directly and fully address the question without drifting into "
        "unrelated territory or rejecting the premise without justification.",
        [
            "Read the question and identify what specific information is being asked for.",
            "Assess whether the prediction directly answers the question asked.",
            "Check whether the prediction stays focused on the question or drifts into unrelated territory.",
            "If the prediction rejects the premise, check whether the reference also rejects it.",
            "Determine what proportion of the prediction is directly relevant to the question.",
        ],
    ),
]

# Grounding sub-dimensions
GROUNDING_GEVAL_DEFS = [
    (
        "entity_grounding",
        "Entity grounding: whether the specific entities named in the prediction "
        "(drugs, genes, proteins, diseases, anatomical structures) are found in "
        "the reference. All named entities should appear in the reference.",
        [
            "List all named entities in the prediction (drugs, genes, proteins, diseases, anatomical structures).",
            "List all named entities in the reference.",
            "Check which prediction entities appear in the reference.",
            "Calculate the proportion of prediction entities that are grounded in the reference.",
        ],
    ),
    (
        "claim_grounding",
        "Claim grounding: whether the factual claims and relationships asserted "
        "in the prediction are traceable to the reference. All claims should "
        "trace back to information in the reference.",
        [
            "List all factual claims and relationships asserted in the prediction.",
            "For each claim, determine whether it traces back to information in the reference.",
            "Distinguish between claims that are directly stated in the reference and those that are inferred or fabricated.",
            "Calculate the proportion of grounded claims relative to total claims.",
        ],
    ),
    (
        "scope_grounding",
        "Scope grounding: whether the prediction stays within the topical scope "
        "of the reference or introduces entirely new topics. The prediction "
        "should stay fully within the reference's scope.",
        [
            "Identify the topical scope of the reference (what subjects and domains it covers).",
            "Determine whether the prediction stays within that scope.",
            "Identify any topics in the prediction that are not covered in the reference.",
            "Assess the proportion of prediction content that falls within the reference's scope.",
        ],
    ),
]


# ---------------------------------------------------------------------------
#  Reference-based n-gram scorers (no judge LLM required)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, split on whitespace/punctuation for n-gram scoring."""
    import re as _re
    return _re.findall(r"\w+", text.lower())


def _ngrams(tokens: List[str], n: int) -> Dict[Any, int]:
    from collections import Counter
    return Counter(tuple(tokens[i: i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def _rouge_n_f1(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    pred_ng = _ngrams(pred_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)
    overlap = sum(min(pred_ng[g], ref_ng[g]) for g in pred_ng)
    precision = overlap / max(sum(pred_ng.values()), 1)
    recall = overlap / max(sum(ref_ng.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_l_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    # LCS via DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    precision = lcs_len / m
    recall = lcs_len / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class RougeScorer(Scorer):
    """Reference-based ROUGE scorer (ROUGE-1, ROUGE-2, ROUGE-L).

    Computes F1 scores using word-level n-gram overlap between the
    prediction and the reference answer.  No judge LLM required.
    """

    name = "rouge"

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, Optional[float]]:
        pred_tok = _tokenize(prediction)
        ref_tok = _tokenize(reference)
        return {
            "rouge1": round(_rouge_n_f1(pred_tok, ref_tok, 1), 4),
            "rouge2": round(_rouge_n_f1(pred_tok, ref_tok, 2), 4),
            "rougeL": round(_rouge_l_f1(pred_tok, ref_tok), 4),
        }


def _sentence_bleu(pred_tokens: List[str], ref_tokens: List[str], max_n: int = 4) -> float:
    """Sentence-level BLEU with modified n-gram precision and brevity penalty."""
    import math
    from collections import Counter

    if not pred_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / len(pred_tokens)
    )

    log_avg = 0.0
    for n in range(1, max_n + 1):
        pred_ng = Counter(tuple(pred_tokens[i: i + n]) for i in range(max(len(pred_tokens) - n + 1, 0)))
        ref_ng = Counter(tuple(ref_tokens[i: i + n]) for i in range(max(len(ref_tokens) - n + 1, 0)))
        clipped = sum(min(cnt, ref_ng[gram]) for gram, cnt in pred_ng.items())
        total = max(len(pred_tokens) - n + 1, 0)
        if total == 0 or clipped == 0:
            return 0.0
        log_avg += math.log(clipped / total)

    return bp * math.exp(log_avg / max_n)


class BleuScorer(Scorer):
    """Reference-based BLEU scorer (BLEU-1 through BLEU-4).

    Computes sentence-level BLEU with a brevity penalty.
    No judge LLM required.
    """

    name = "bleu"

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, Optional[float]]:
        pred_tok = _tokenize(prediction)
        ref_tok = _tokenize(reference)
        bleu1 = _sentence_bleu(pred_tok, ref_tok, max_n=1)
        bleu2 = _sentence_bleu(pred_tok, ref_tok, max_n=2)
        bleu4 = _sentence_bleu(pred_tok, ref_tok, max_n=4)
        return {
            "bleu1": round(bleu1, 4),
            "bleu2": round(bleu2, 4),
            "bleu4": round(bleu4, 4),
        }


# ---------------------------------------------------------------------------
#  Builder
# ---------------------------------------------------------------------------

def build_scorers(
    metrics: List[str],
    judge_config: Optional[Dict[str, Any]] = None,
) -> List[Scorer]:
    """Instantiate scorers from a list of metric names.

    Reference-based metrics (no judge LLM required):
      - "rouge": ROUGE-1 / ROUGE-2 / ROUGE-L F1 (word n-gram overlap)
      - "bleu": BLEU-1 / BLEU-2 / BLEU-4 (n-gram precision + brevity penalty)

    LLM-judge metrics (require judge_config):
      - "answer_relevancy": DeepEval AnswerRelevancyMetric
      - "correctness": 3x GEval (accuracy / completeness / relevance)
      - "faithfulness": DeepEval FaithfulnessMetric
      - "hallucination": skipped (unreliable; use faithfulness)
      - "grounding": 3x GEval (entity / claim / scope grounding)

    Aliases:
      - "llm_judge" -> "correctness"
    """
    NON_LLM_METRICS: set[str] = {"rouge", "bleu"}
    ALIASES: Dict[str, str] = {
        "llm_judge": "correctness",
    }

    # Build judge LLM only when at least one LLM-based metric is requested
    needs_judge = any(ALIASES.get(m, m) not in NON_LLM_METRICS for m in metrics)
    if needs_judge and not judge_config:
        logger.warning(
            "No judge_config provided — LLM-based scorers will be skipped. "
            "Configure a judge provider (vLLM, OpenAI, etc.) in the evaluator settings."
        )
    judge_llm = _make_deepeval_adapter(judge_config) if (judge_config and needs_judge) else None

    scorers: List[Scorer] = []
    seen: set[str] = set()

    for name in metrics:
        resolved = ALIASES.get(name, name)
        if resolved in seen:
            continue
        seen.add(resolved)

        if resolved == "rouge":
            scorers.append(RougeScorer())

        elif resolved == "bleu":
            scorers.append(BleuScorer())

        elif resolved == "answer_relevancy":
            if not judge_llm:
                logger.warning("Skipping answer_relevancy: no judge_config provided")
                continue
            from deepeval.metrics import AnswerRelevancyMetric

            def _make_ar(llm=judge_llm):
                return AnswerRelevancyMetric(
                    model=llm, threshold=0, async_mode=False, verbose_mode=False,
                )
            scorers.append(DeepEvalScorer(
                "answer_relevancy", _make_ar(), "answer_relevancy",
                reference_field="expected_output",
                metric_factory=_make_ar,
            ))

        elif resolved == "correctness":
            if not judge_llm:
                logger.warning("Skipping correctness: no judge_config provided")
                continue
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCaseParams

            for score_key, criteria, steps in CORRECTNESS_GEVAL_DEFS:
                def _make_geval(llm=judge_llm, sk=score_key, cr=criteria, st=steps):
                    return GEval(
                        name=sk, criteria=cr, evaluation_steps=st,
                        evaluation_params=[
                            LLMTestCaseParams.ACTUAL_OUTPUT,
                            LLMTestCaseParams.EXPECTED_OUTPUT,
                        ],
                        model=llm, threshold=0, async_mode=False, verbose_mode=False,
                    )
                scorers.append(DeepEvalScorer(
                    score_key, _make_geval(), score_key,
                    reference_field="expected_output",
                    metric_factory=_make_geval,
                ))

        elif resolved == "faithfulness":
            if not judge_llm:
                logger.warning("Skipping faithfulness: no judge_config provided")
                continue
            from deepeval.metrics import FaithfulnessMetric

            def _make_faith(llm=judge_llm):
                return FaithfulnessMetric(
                    model=llm, threshold=0, async_mode=False, verbose_mode=False,
                )
            scorers.append(DeepEvalScorer(
                "faithfulness", _make_faith(), "faithfulness",
                reference_field="retrieval_context",
                metric_factory=_make_faith,
            ))

        elif resolved == "hallucination":
            # Skipped: DeepEval's HallucinationMetric produces unreliable
            # binary (0/1) scores that contradict the faithfulness metric.
            # Faithfulness already measures claim grounding more reliably.
            logger.info("Skipping hallucination metric (unreliable; use faithfulness instead)")

        elif resolved == "grounding":
            if not judge_llm:
                logger.warning("Skipping grounding: no judge_config provided")
                continue
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCaseParams

            for score_key, criteria, steps in GROUNDING_GEVAL_DEFS:
                def _make_grounding(llm=judge_llm, sk=score_key, cr=criteria, st=steps):
                    return GEval(
                        name=sk, criteria=cr, evaluation_steps=st,
                        evaluation_params=[
                            LLMTestCaseParams.ACTUAL_OUTPUT,
                            LLMTestCaseParams.EXPECTED_OUTPUT,
                        ],
                        model=llm, threshold=0, async_mode=False, verbose_mode=False,
                    )
                scorers.append(DeepEvalScorer(
                    score_key, _make_grounding(), score_key,
                    reference_field="expected_output",
                    metric_factory=_make_grounding,
                ))

        else:
            logger.warning("Unknown metric %r; skipping", name)

    # Computed averages
    if "correctness" in seen:
        scorers.append(_CorrectnessAverager())
    if "grounding" in seen:
        scorers.append(_GroundingAverager())

    return scorers


# ---------------------------------------------------------------------------
#  Computed-average post-processors
# ---------------------------------------------------------------------------

class _CorrectnessAverager(Scorer):
    """Post-processor that computes judge_avg from correctness sub-scores."""

    name = "correctness_avg"

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, Optional[float]]:
        return {}


class _GroundingAverager(Scorer):
    """Post-processor that computes grounding_avg from grounding sub-scores."""

    name = "grounding_avg"

    def score(
        self, prediction: str, reference: str, question: str
    ) -> Dict[str, Optional[float]]:
        return {}
