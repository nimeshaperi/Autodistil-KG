# Autodistil-KG API
# Build context must be the monorepo root so we can access all three local packages.
# docker build -f Autodistil-KG_api/Dockerfile -t autodistil-kg-api .

FROM python:3.13-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install graphrag library (API depends on it as a path dep) ─────────────
COPY Autodistil-KG_graphrag/pyproject.toml /app/Autodistil-KG_graphrag/
COPY Autodistil-KG_graphrag/src /app/Autodistil-KG_graphrag/src
RUN cd /app/Autodistil-KG_graphrag && pip install --no-cache-dir .

# ── Install core library (layer cached separately) ────────────────────────
COPY Autodistil-KG_core/pyproject.toml Autodistil-KG_core/README.md /app/Autodistil-KG_core/
COPY Autodistil-KG_core/src /app/Autodistil-KG_core/src

# Install core WITHOUT finetune extras (no GPU / unsloth needed in Docker).
# Pass --build-arg INSTALL_FINETUNE=1 to include fine-tuning support.
ARG INSTALL_FINETUNE=0
RUN cd /app/Autodistil-KG_core && \
    if [ "$INSTALL_FINETUNE" = "1" ]; then \
        pip install --no-cache-dir ".[finetune]"; \
    else \
        pip install --no-cache-dir .; \
    fi

# ── Install API framework dependencies ────────────────────────────────────
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "redis>=5.0.0"

# ── Install API package (--no-deps: local path deps already installed) ─────
COPY Autodistil-KG_api/pyproject.toml /app/Autodistil-KG_api/
COPY Autodistil-KG_api/src /app/Autodistil-KG_api/src
RUN cd /app/Autodistil-KG_api && pip install --no-cache-dir --no-deps -e .

# ── Workspace directory for pipeline outputs ──────────────────────────────
RUN mkdir -p /app/workspace/output
ENV KG_PIPELINE_WORKSPACE=/app/workspace

# Disable torch.compile (not needed without GPU, avoids startup errors)
ENV UNSLOTH_COMPILE_DISABLE=1
ENV TORCH_COMPILE_DISABLE=1

EXPOSE 8000

CMD ["uvicorn", "autodistilkg_api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
