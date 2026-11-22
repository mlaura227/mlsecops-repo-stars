# Bootstrap image only: API and Compose services follow in later milestones.
FROM ghcr.io/astral-sh/uv:0.10.0 AS uv-bin
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1 \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=uv-bin /uv /uvx /usr/local/bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-editable

CMD ["python", "-c", "import mlsecops; print('Bootstrap ready: API will follow in later milestone.')"]
