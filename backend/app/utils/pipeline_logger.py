"""
Structured pipeline logging.

Writes one JSON object per line to backend/logs/pipeline.jsonl, one line
per pipeline stage (embedding, vector search, keyword search, RRF fusion,
reranking) for every request, correlated by a per-request ID. Kept
separate from the general application logger (utils/logger.py) so this
file stays clean, structured, and easy to grep/parse -- it's meant for
"what happened during this query", not general app diagnostics.

Usage:
    from ..utils.pipeline_logger import new_request_id, log_stage

    new_request_id()  # once, at the start of a request
    log_stage("embedding", duration_ms=12.3, model="voyage-4-lite")
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "pipeline.jsonl"

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_pipeline_logger = logging.getLogger("pipeline")


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # record.msg is already a dict, built by log_stage()
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            **record.msg,
        }
        return json.dumps(payload, default=str)


def setup_pipeline_logging() -> None:
    """Attach a rotating JSON-lines file handler to the pipeline logger.

    Call once at app startup. Idempotent (safe to call more than once).
    """
    if _pipeline_logger.handlers:
        return  # already configured

    LOG_DIR.mkdir(exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(_JsonLineFormatter())
    _pipeline_logger.addHandler(handler)
    _pipeline_logger.setLevel(logging.INFO)
    _pipeline_logger.propagate = False  # don't also spam the console/app log


def new_request_id() -> str:
    """Start a new correlated request; returns the id for reference."""
    rid = uuid.uuid4().hex[:8]
    _request_id.set(rid)
    return rid


def log_stage(stage: str, **fields: Any) -> None:
    """Log one pipeline stage as a JSON line, tagged with the current request id"""
    _pipeline_logger.info({
        "request_id": _request_id.get(),
        "stage": stage,
        **fields,
    })


class timed_stage:
    """Context manager: times a block and logs it as a pipeline stage on exit.

    Usage:
        with timed_stage("vector_search", code_type="cpt") as t:
            results = await search_cpt_codes_vector(...)
            t["count"] = len(results)
    """

    def __init__(self, stage: str, **fields: Any):
        self.stage = stage
        self.fields = fields

    def __enter__(self):
        self._start = time.perf_counter()
        return self.fields

    def __exit__(self, exc_type, exc, tb):
        duration_ms = (time.perf_counter() - self._start) * 1000
        if exc_type is not None:
            log_stage(self.stage, duration_ms=round(duration_ms, 1), error=str(exc))
        else:
            log_stage(self.stage, duration_ms=round(duration_ms, 1), **self.fields)
        return False
