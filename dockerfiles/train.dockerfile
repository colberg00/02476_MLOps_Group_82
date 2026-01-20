# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./
# If your project build needs README present, keep this
COPY README.md ./

# Copy source code
COPY src ./src
# If your training uses configs, keep this (safe even if unused)
COPY configs ./configs
# copy the license
COPY LICENSE* ./

# Install dependencies exactly from the lockfile
RUN uv sync --frozen

# Default: run training script
CMD ["uv", "run", "python", "src/mlops_course_project/train.py"]
