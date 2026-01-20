from pathlib import Path

import pandas as pd
import pytest

from mlops_course_project.train import _load_split, _metrics


def _write_dummy_split_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "slug": [
                "blinken meets qatars prime minister",
                "senators grill ai companies",
                "nyc firefighter dies after blaze",
                "immigration debate heats up",
            ],
            "outlet": ["fox", "nbc", "fox", "nbc"],
        }
    )
    df.to_csv(path, index=False)


def test_load_split_success(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    _write_dummy_split_csv(csv_path)

    X, y = _load_split(csv_path)

    assert len(X) == 4
    assert len(y) == 4
    assert all(isinstance(x, str) for x in X)
    assert set(y).issubset({0, 1})


def test_load_split_missing_columns(tmp_path: Path):
    df = pd.DataFrame({"wrong": ["x", "y"]})
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        _load_split(csv_path)


def test_metrics_computation():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    metrics = _metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
