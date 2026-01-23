FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

# DVC is used in Cloud Run to materialize reference data from the shared remote (GCS)
RUN python -m pip install --no-cache-dir "dvc[gcs]"

COPY src src/
# Include DVC metadata (small files) so the service can `dvc pull` in Cloud Run
COPY dvc.yaml dvc.yaml
COPY dvc.lock dvc.lock
COPY .dvc .dvc
COPY data/*.dvc data/
COPY data/raw/*.dvc data/raw/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.mlops_course_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
