"""
FastAPI application factory for the Autodistil-KG Pipeline API.

Provides REST endpoints for pipeline management, inference, and artifact
retrieval, plus a WebSocket endpoint for real-time progress streaming.

Execution modes:

- **Redis queue** (production): Pipeline jobs are enqueued and processed by
  a background worker thread.  Events flow via Redis pub/sub.
- **In-process** (development): Pipeline runs directly inside the WebSocket
  handler when Redis is not available.
"""
# CRITICAL: compat must be imported first -- it sets env vars and patches
# torch.compile before any other imports can trigger torch loading.
import autodistilkg_api.compat  # noqa: F401  -- side-effect import

import asyncio
import logging
import threading

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .redis_client import get_redis_url
from .routers import health, stages, pipelines, inference, schemas, ws
from .state import load_persisted_runs
from .worker import pipeline_worker_loop

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

_worker_stop = threading.Event()


async def _lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Application lifespan: restore state and start background worker."""
    load_persisted_runs()
    try:
        await asyncio.to_thread(lambda: redis.from_url(get_redis_url(), decode_responses=True).ping())
        ws.redis_available = True
        t = threading.Thread(target=pipeline_worker_loop, args=(_worker_stop,), daemon=True)
        t.start()
        logger.info("Redis pipeline queue enabled")
    except Exception as e:
        logger.info("Redis not used for pipeline queue: %s. WebSocket runs in-process.", e)
    yield
    _worker_stop.set()


app = FastAPI(
    title="KG Pipeline API",
    description="Create and run Autodistil-KG pipelines via REST and WebSocket",
    version="0.2.0",
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(stages.router, tags=["stages"])
app.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])
app.include_router(schemas.router, tags=["schemas"])
app.include_router(ws.router, tags=["websocket"])
