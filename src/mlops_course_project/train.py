from __future__ import annotations
from mlops_course_project.model import Model
from mlops_course_project.data import MyDataset



import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    max_features: int = 50_000
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    C: float = 1.0
    max_iter: int = 2000


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


def train(
    processed_dir: Path = Path("data/processed"),
    model_out: Path = Path("models/baseline.joblib"),
    metrics_out: Path = Path("reports/baseline_metrics.json"),
    seed: int = 42,
    max_features: int = 50_000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    C: float = 1.0,
    max_iter: int = 2000,
) -> None:
    """
    Train TF-IDF + Logistic Regression baseline on URL slug text.

    Expects:
      data/processed/train.csv, val.csv, test.csv
    Each must have columns: slug,outlet
    """
    cfg = TrainConfig(
        seed=seed,
        max_features=max_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=min_df,
        C=C,
        max_iter=max_iter,
    )

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing split: {p}. Run preprocessing first.")

    X_train, y_train = _load_split(train_path)
    X_val, y_val = _load_split(val_path)
    X_test, y_test = _load_split(test_path)

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(cfg.ngram_min, cfg.ngram_max),
                    min_df=cfg.min_df,
                    max_features=cfg.max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    random_state=cfg.seed,
                    C=cfg.C,
                    max_iter=cfg.max_iter,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )

    print("Training baseline (TF-IDF + LogisticRegression)...")
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
    pipeline.fit(X_train, y_train)

    # Evaluate
    val_pred = pipeline.predict(X_val).tolist()
    test_pred = pipeline.predict(X_test).tolist()

    val_metrics = _metrics(y_val, val_pred)
    test_metrics = _metrics(y_test, test_pred)

    results: dict[str, Any] = {
        "config": cfg.__dict__,
        "label_map": LABEL_MAP,
        "val": val_metrics,
        "test": test_metrics,
    }

    # Save outputs
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    dump(pipeline, model_out)

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Saved model to:   {model_out}")
    print(f"Saved metrics to: {metrics_out}")
    print()
    print("Validation metrics:")
    print(
        f"  acc={val_metrics['accuracy']:.4f}  f1={val_metrics['f1']:.4f}  "
        f"prec={val_metrics['precision']:.4f}  rec={val_metrics['recall']:.4f}"
    )
    print("Test metrics:")
    print(
        f"  acc={test_metrics['accuracy']:.4f}  f1={test_metrics['f1']:.4f}  "
        f"prec={test_metrics['precision']:.4f}  rec={test_metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    typer.run(train)