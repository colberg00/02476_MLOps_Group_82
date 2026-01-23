from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from loguru import logger

from mlops_course_project import setup_logging
from mlops_course_project.model import create_baseline_model

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


setup_logging("train")

LABEL_MAP = {"nbc": 0, "fox": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def _load_split(path: Path) -> tuple[list[str], list[int]]:
    """Load and validate a data split from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (texts, labels).

    Raises:
        ValueError: If file is missing required columns or has no valid rows.
    """
    logger.debug(f"Loading split from {path}")
    df = pd.read_csv(path)
    logger.debug(f"Loaded {len(df)} rows from {path}")

    if "slug" not in df.columns or "outlet" not in df.columns:
        logger.error(f"{path} must have columns ['slug', 'outlet'], but has {list(df.columns)}")
        raise ValueError(f"{path} must have columns ['slug', 'outlet'], but has {list(df.columns)}")

    # Basic hygiene
    df["slug"] = df["slug"].astype(str).str.strip()
    df["outlet"] = df["outlet"].astype(str).str.lower()

    initial_count = len(df)
    df = df[df["slug"] != ""].copy()
    df = df[df["outlet"].isin(LABEL_MAP.keys())].copy()
    logger.debug(f"After filtering: {len(df)} rows (removed {initial_count - len(df)})")

    X = df["slug"].tolist()
    y = df["outlet"].map(LABEL_MAP).astype(int).tolist()

    if len(X) == 0:
        logger.error(f"{path} produced 0 rows after filtering. Check preprocessing.")
        raise ValueError(f"{path} produced 0 rows after filtering. Check preprocessing.")

    logger.info(f"Split {path.stem}: {len(X)} samples")
    return X, y


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, precision, recall, F1, confusion matrix, and classification report.
    """
    metrics = {
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
    logger.debug(f"Metrics - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    return metrics


def train(cfg: DictConfig) -> None:
    """
    Train TF-IDF + Logistic Regression baseline on URL slug text.

    Expects:
      data/processed/train.csv, val.csv, test.csv
    Each must have columns: slug,outlet
    """
    logger.info("Starting training pipeline")
    logger.debug(f"Config: {OmegaConf.to_yaml(cfg)}")

    repo_root = Path(get_original_cwd())
    run_dir = Path.cwd()
    logger.debug(f"Repo root: {repo_root}, Run dir: {run_dir}")

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.get("wandb_project", "news-classifier"),
        entity=cfg.get("wandb_entity", None),
        config=wandb_config,
        mode=cfg.get("wandb_mode", "online"),
    )
    logger.info(f"Initialized wandb run: {wandb.run.name}")

    processed_dir = repo_root / cfg.processed_dir

    # If cfg.model_out / cfg.metrics_out are absolute or run-dir relative, just Path() them
    model_out = Path(cfg.model_out)
    metrics_out = Path(cfg.metrics_out)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    logger.info(f"Looking for data in: {processed_dir}")
    for p in (train_path, val_path, test_path):
        if not p.exists():
            logger.error(f"Missing split: {p}. Run preprocessing first.")
            raise FileNotFoundError(f"Missing split: {p}. Run preprocessing first.")

    X_train, y_train = _load_split(train_path)
    X_val, y_val = _load_split(val_path)
    X_test, y_test = _load_split(test_path)

    logger.info(f"Dataset sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    wandb.log(
        {
            "dataset/train_size": len(X_train),
            "dataset/val_size": len(X_val),
            "dataset/test_size": len(X_test),
        }
    )

    logger.info("Creating baseline model (TF-IDF + LogisticRegression)")
    logger.debug(
        f"Model parameters - max_features: {cfg.max_features}, ngram_range: ({cfg.ngram_min}, {cfg.ngram_max}), min_df: {cfg.min_df}, C: {cfg.C}, max_iter: {cfg.max_iter}"
    )
    pipeline = create_baseline_model(
        seed=cfg.seed,
        max_features=cfg.max_features,
        ngram_min=cfg.ngram_min,
        ngram_max=cfg.ngram_max,
        min_df=cfg.min_df,
        C=cfg.C,
        max_iter=cfg.max_iter,
    )

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed")

    logger.info("Making predictions on validation set")
    val_pred = pipeline.predict(X_val).tolist()
    logger.info("Making predictions on test set")
    test_pred = pipeline.predict(X_test).tolist()

    logger.info("Computing validation metrics")
    val_metrics = _metrics(y_val, val_pred)
    logger.info(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

    wandb.log(
        {
            "val/accuracy": val_metrics["accuracy"],
            "val/precision": val_metrics["precision"],
            "val/recall": val_metrics["recall"],
            "val/f1": val_metrics["f1"],
        }
    )

    logger.info("Computing test metrics")
    test_metrics = _metrics(y_test, test_pred)
    logger.info(f"Test - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

    wandb.log(
        {
            "test/accuracy": test_metrics["accuracy"],
            "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"],
            "test/f1": test_metrics["f1"],
        }
    )

    wandb.log(
        {
            "confusion_matrix/val": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_val,
                preds=val_pred,
                class_names=[INV_LABEL_MAP[0], INV_LABEL_MAP[1]],
            ),
            "confusion_matrix/test": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test,
                preds=test_pred,
                class_names=[INV_LABEL_MAP[0], INV_LABEL_MAP[1]],
            ),
        }
    )

    results: dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "label_map": LABEL_MAP,
        "val": val_metrics,
        "test": test_metrics,
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    dump(pipeline, model_out)
    logger.info(f"Saved model to: {model_out}")

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_out}")

    artifact = wandb.Artifact(
        name="news-classifier-model",
        type="model",
        description="TF-IDF + LogisticRegression news outlet classifier",
        metadata={
            "accuracy": test_metrics["accuracy"],
            "f1": test_metrics["f1"],
            "config": wandb_config,
        },
    )
    artifact.add_file(str(model_out))
    wandb.log_artifact(artifact)
    logger.info(f"Logged model artifact to wandb: {artifact.name}")

    wandb.finish()
    logger.info("Training pipeline completed successfully")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    try:
        train(cfg)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
