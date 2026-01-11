# Base image
FROM python:3.12-slim

# System deps (common for python packages that compile things)
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Workdir inside container
WORKDIR /app

# Install python deps first (better caching)
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
RUN pip install -r requirements.txt --no-cache-dir

# Copy the actual code + data
COPY src/ src/
COPY data/ data/

# Install your package (so imports work)
RUN pip install . --no-deps --no-cache-dir

# TODO: change <project-name> to your package folder name under src/
ENTRYPOINT ["python", "-u", "src/mlops/train.py"]
