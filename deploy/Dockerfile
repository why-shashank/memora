# memora api image — build from the repo root: docker compose up (dev-plan §11)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, project second: code edits reuse the cached torch/deps layer.
# --extra local bakes in the default zero-key embedding path (sentence-transformers).
COPY pyproject.toml uv.lock ./
# cache mount: wheels cache lives outside the image (≈1GB) and survives across builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra local --no-install-project

COPY README.md alembic.ini ./
COPY memora/ memora/
COPY deploy/alembic/ deploy/alembic/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra local

EXPOSE 8000
# Migrations run on boot (dev-plan §11), then serve.
CMD ["sh", "-c", "alembic upgrade head && uvicorn memora.api.app:app --host 0.0.0.0 --port 8000"]
