# MLOps Project Checklist

> Tick boxes as we complete items. Keep this file updated.

## Week 1
- [X] Create a git repository (M5)
- [X] Ensure all team members have write access (M5)
- [X] Create a dedicated environment and track packages (M2)
- [X] Create initial file structure (cookiecutter/template) (M6)
- [X] Implement `data.py` download + preprocessing (M6)
- [ ] Implement model in `model.py` + training in `train.py` and run it (M6)
- [ ] Keep `requirements*.txt` or `pyproject.toml` up-to-date (M2+M6)
- [ ] Follow coding practices (pep8) (M7)
- [ ] Add type hints + document essential parts (M7)
- [ ] Setup version control for data / part of data (M8)
- [ ] Add CLI commands where it makes sense (M9)
- [ ] Construct one or more dockerfiles (M10)
- [ ] Build dockerfiles locally and verify they work (M10)
- [ ] Write one or more config files for experiments (M11)
- [ ] Use Hydra for configs + hyperparameters (M11)
- [ ] Profiling (M12)
- [ ] Logging important events (M14)
- [ ] Use Weights & Biases for metrics/artifacts (M14)
- [ ] Consider hyperparameter sweep (M14)
- [ ] Use PyTorch Lightning (if applicable) (M15)

## Week 2
- [ ] Unit tests for data code (M16)
- [ ] Unit tests for model/training (M16)
- [ ] Code coverage (M16)
- [ ] CI on GitHub (M17)
- [ ] CI caching + multi-os/python/torch testing (M17)
- [ ] Add linting step in CI (M17)
- [ ] Pre-commit hooks (M18)
- [ ] Workflow triggers when data changes (M19)
- [ ] Workflow triggers when model registry changes (M19)
- [ ] Create GCP bucket for data + connect to DVC (M21)
- [ ] Auto-build docker images workflow (M21)
- [ ] Run training in GCP (Engine or Vertex AI) (M21)
- [ ] FastAPI inference app (M22)
- [ ] Deploy model in GCP (Functions or Run) (M23)
- [ ] API tests + CI for API tests (M24)
- [ ] Load testing (M24)
- [ ] Specialized deployment API (ONNX/BentoML) (M25)
- [ ] Frontend for API (M26)

## Week 3
- [ ] Robustness to data drift (M27)
- [ ] Collect input-output data from deployed app (M27)
- [ ] Deploy drift detection API (M27)
- [ ] Add system metrics to API (M28)
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
- [ ] Upload all code to GitHub
