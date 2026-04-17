"""
Log handlers that intercept structured pipeline events and forward them.

Two handlers exist for the two execution modes:

- :class:`RedisLogHandler` -- publishes events to a Redis pub/sub channel
  so the background pipeline worker can stream them to WebSocket clients.
- ``QueueLogHandler`` lives in ``routers/ws.py`` and pushes events into an
  in-process queue for the direct (no-Redis) execution path.

Both handlers detect structured events (``traversal_event``, ``eval_event``,
``finetuner_event``) attached as logging extras by the core package and
forward them with the appropriate channel prefix.
"""
import json
import logging
from typing import Set


class RedisLogHandler(logging.Handler):
    """Publishes log messages and structured events to a Redis pub/sub channel.

    Structured events (attached to log records via ``extra={"traversal_event": {...}}``)
    are forwarded as-is.  Regular log messages are truncated to 500 chars and
    filtered by :attr:`frontend_level`.
    """

    #: Logger name prefixes to forward. All other loggers are silently dropped.
    ALLOWED_LOGGERS: Set[str] = {
        "autodistil_kg", "autodistil_kg_graphrag", "unsloth", "trl",
        "transformers", "datasets", "unsloth.trainer",
    }

    def __init__(self, redis_client, channel: str, level: int = logging.INFO) -> None:  # type: ignore[no-untyped-def]
        super().__init__(level)
        self.redis_client = redis_client
        self.channel = channel
        # Frontend receives INFO+ logs by default; DEBUG only when log_level is DEBUG
        self.frontend_level = logging.INFO

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not any(record.name.startswith(p) for p in self.ALLOWED_LOGGERS):
                return
            # Structured events always flow through regardless of level
            traversal_event = getattr(record, "traversal_event", None)
            if traversal_event and isinstance(traversal_event, dict):
                self.redis_client.publish(self.channel, json.dumps({"event": "traversal_progress", **traversal_event}))
                return
            eval_event = getattr(record, "eval_event", None)
            if eval_event and isinstance(eval_event, dict):
                self.redis_client.publish(self.channel, json.dumps({"event": "eval_progress", **eval_event}))
                return
            finetuner_event = getattr(record, "finetuner_event", None)
            if finetuner_event and isinstance(finetuner_event, dict):
                self.redis_client.publish(self.channel, json.dumps({"event": "finetuner_progress", **finetuner_event}))
                return
            # Regular logs: filter by frontend_level
            if record.levelno < self.frontend_level:
                return
            msg = self.format(record)
            if len(msg) > 500:
                msg = msg[:497] + "..."
            self.redis_client.publish(self.channel, json.dumps({
                "event": "log",
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
            }))
        except Exception:
            pass
