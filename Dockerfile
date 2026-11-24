# One locked Python 3.13 image, shared by MLflow, data preparation and FastAPI.
FROM ghcr.io/astral-sh/uv:0.10.0 AS uv-bin
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_DEV=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY --from=uv-bin /uv /uvx /usr/local/bin/
WORKDIR /app

# No dependency on a README or checkout-specific files: install locked deps only.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY src/ ./src/
COPY dashboard/ ./dashboard/

# MLflow writes SQLite and model artifacts into a persistent named Docker volume.
RUN groupadd --system mlsecops \
    && useradd --system --gid mlsecops --home-dir /app mlsecops \
    && mkdir -p /mlflow/artifacts \
    && chown -R mlsecops:mlsecops /app /mlflow
USER mlsecops

EXPOSE 5000 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
