from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from mlops_course_project.model import create_baseline_model

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

LABEL_MAP = {"nbc": 0, "fox": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def _load_split(path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)

    if "slug" not in df.columns or "outlet" not in df.columns:
        raise ValueError(f"{path} must have columns ['slug', 'outlet'], but has {list(df.columns)}")

    # Basic hygiene
    df["slug"] = df["slug"].astype(str).str.strip()
    df["outlet"] = df["outlet"].astype(str).str.strip().str.lower()

    df = df[df["slug"] != ""].copy()
    df = df[df["outlet"].isin(LABEL_MAP.keys())].copy()

    X = df["slug"].tolist()
    y = df["outlet"].map(LABEL_MAP).astype(int).tolist()

    if len(X) == 0:
        raise ValueError(f"{path} produced 0 rows after filtering. Check preprocessing.")
    return X, y


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[INV_LABEL_MAP[0], INV_LABEL_MAP[1]],
            zero_division=0,
            output_dict=True,
        ),
    }


def train(cfg: DictConfig) -> None:
    """
    Train TF-IDF + Logistic Regression baseline on URL slug text.

    Expects:
      data/processed/train.csv, val.csv, test.csv
    Each must have columns: slug,outlet
    """
    repo_root = Path(get_original_cwd())     # repo root where you launched
    run_dir = Path.cwd()                     # hydra run dir (because chdir: true)

    processed_dir = repo_root / cfg.processed_dir

    # If cfg.model_out / cfg.metrics_out are absolute or run-dir relative, just Path() them
    model_out = Path(cfg.model_out)
    metrics_out = Path(cfg.metrics_out)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing split: {p}. Run preprocessing first.")

    X_train, y_train = _load_split(train_path)
    X_val, y_val = _load_split(val_path)
    X_test, y_test = _load_split(test_path)

    pipeline = create_baseline_model(
        seed=cfg.seed,
        max_features=cfg.max_features,
        ngram_min=cfg.ngram_min,
        ngram_max=cfg.ngram_max,
        min_df=cfg.min_df,
        C=cfg.C,
        max_iter=cfg.max_iter,
    )

    print("Training baseline (TF-IDF + LogisticRegression)...")
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val).tolist()
    test_pred = pipeline.predict(X_test).tolist()

    val_metrics = _metrics(y_val, val_pred)
    test_metrics = _metrics(y_test, test_pred)

    results: dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "label_map": LABEL_MAP,
        "val": val_metrics,
        "test": test_metrics,
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    dump(pipeline, model_out)

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model to:   {model_out}")
    print(f"Saved metrics to: {metrics_out}")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()