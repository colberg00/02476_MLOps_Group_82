# MLOps Project Checklist

> Tick boxes as we complete items. Keep this file updated.

## Week 1
- [X] Create a git repository (M5)
- [X] Ensure all team members have write access (M5)
- [X] Create a dedicated environment and track packages (M2)
- [X] Create initial file structure (cookiecutter/template) (M6)
- [X] Implement `data.py` download + preprocessing (M6)
- [X] Implement model in `model.py` + training in `train.py` and run it (M6)
- [X] Keep `requirements*.txt` or `pyproject.toml` up-to-date (M2+M6)
- [X] Follow coding practices (pep8) (M7)
- [X] Add type hints + document essential parts (M7)
- [ ] Setup version control for data / part of data (M8) --- Jacob
- [½] Add CLI commands where it makes sense (M9) --- Kelvin
- [X] Construct one or more dockerfiles (M10) --- Kelvin
- [X] Build dockerfiles locally and verify they work (M10) --- Kelvin
- [X] Write one or more config files for experiments (M11) --- Oskar
- [X] Use Hydra for configs + hyperparameters (M11) --- Oskar
- [ ] Profiling (M12) --- Oskar
- [ ] Logging important events (M14) --- Christian
- [ ] Use Weights & Biases for metrics/artifacts (M14)
- [ ] Consider hyperparameter sweep (M14) --- Oskar
- [ ] Use PyTorch Lightning (if applicable) (M15)

## Week 2
- [X] Unit tests for data code (M16) --- Kelvin
- [ ] Unit tests for model/training (M16) --- Kelvin
- [½] Code coverage (M16) --- Kelvin
- [ ] CI on GitHub (M17) --- Christian
- [ ] CI caching + multi-os/python/torch testing (M17) --- Christian
- [ ] Add linting step in CI (M17) --- Christian
- [ ] Pre-commit hooks (M18) --- Christian
- [ ] Workflow triggers when data changes (M19)
- [ ] Workflow triggers when model registry changes (M19)
- [ ] Create GCP bucket for data + connect to DVC (M21) --- Jacob
- [ ] Auto-build docker images workflow (M21) --- Jacob
- [ ] Run training in GCP (Engine or Vertex AI) (M21) --- Jacob
- [ ] FastAPI inference app (M22) --- Christian
- [ ] Deploy model in GCP (Functions or Run) (M23) --- Jacob
- [ ] API tests + CI for API tests (M24) --- Christian
- [ ] Load testing (M24) --- Christian + Oskar
- [ ] Specialized deployment API (ONNX/BentoML) (M25)
- [ ] Frontend for API (M26) --- Nice to have

## Week 3
- [ ] Robustness to data drift (M27) --- Oskar
- [ ] Collect input-output data from deployed app (M27) --- Jacob
- [ ] Deploy drift detection API (M27) --- Oskar
- [ ] Add system metrics to API (M28) --- Oskar
- [ ] Cloud monitoring for metrics (M28)
- [ ] Alerts in GCP if app misbehaves (M28)
- [ ] (If applicable) Optimize data loading (distributed) (M29)
- [ ] (If applicable) Optimize training pipeline (distributed) (M30)
- [ ] Quantization/compilation/pruning for inference speed (M31)

## Extra
- [ ] Documentation for application (M32)
- [ ] Publish docs on GitHub Pages (M32)
- [ ] Revisit project description (did it meet goals?)
- [ ] Architectural diagram of MLOps pipeline
- [ ] Ensure everyone understands all parts of the project
- [X] Upload all code to GitHub
