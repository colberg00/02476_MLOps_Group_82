from __future__ import annotations

import cProfile
from importlib.resources import path
import pstats
from pathlib import Path

import pandas as pd
import numpy as np
from evidently.legacy.metric_preset import DataQualityPreset, TargetDriftPreset, TextEvals
from evidently.legacy.report import Report
try:
    # Evidently versions differ in where ColumnMapping lives
    from evidently import ColumnMapping  # type: ignore
except Exception:  # pragma: no cover
    try:
        from evidently.legacy.pipeline.column_mapping import ColumnMapping  # type: ignore
    except Exception:
        from evidently.pipeline.column_mapping import ColumnMapping  # type: ignore

# Evidently TestSuite + Tests (version-safe imports)
try:
    from evidently.test_suite import TestSuite  # type: ignore
    from evidently.tests import (  # type: ignore
        TestShareOfRowsWithMissingValues,
        TestColumnDrift,
        TestColumnsType,
        TestColumnNumberOfMissingValues,
        TestColumnRegExp,
    )
except Exception:  # pragma: no cover
    from evidently.legacy.test_suite import TestSuite  # type: ignore
    from evidently.legacy.tests import (  # type: ignore
        TestShareOfRowsWithMissingValues,
        TestColumnDrift,
        TestColumnsType,
        TestColumnNumberOfMissingValues,
        TestColumnRegExp,
    )

from loguru import logger

from mlops_course_project import setup_logging


setup_logging("data_drift")


LABEL_MAP = {"nbc": 0, "fox": 1}


def load_current_last_n(path: str | Path, n: int) -> pd.DataFrame:
    """Load the current (prediction) CSV and keep only the last n rows."""
    df = pd.read_csv(path)
    return df.tail(n).reset_index(drop=True)


def _profile_block(fn, out: str | Path = "reports/data_drift.prof", sort_by: str = "cumtime", top: int = 30) -> None:
    """Profile a callable and write stats to a .prof file (snakeviz-compatible) + print summary."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prof = cProfile.Profile()
    prof.enable()
    fn()
    prof.disable()

    prof.dump_stats(str(out_path))
    logger.info(f"Wrote profiling stats to: {out_path}")
    pstats.Stats(prof).strip_dirs().sort_stats(sort_by).print_stats(top)


def run_drift_report(
    reference_csv: str | Path = "data/processed/train.csv",
    current_csv: str | Path = "prediction_database.csv",
    out_html: str | Path = "reports/data_drift_report.html",
) -> Path:
    """Generate and save an Evidently drift report."""
    reference_csv = Path(reference_csv)
    current_csv = Path(current_csv)
    out_html = Path(out_html)

    logger.info("Starting drift report generation")
    logger.debug(f"Reference CSV: {reference_csv}")
    logger.debug(f"Current CSV: {current_csv}")

    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")
    if not current_csv.exists():
        raise FileNotFoundError(f"Current CSV not found: {current_csv}")

    # Reference = training data
    ref = pd.read_csv(reference_csv)
    logger.info(f"Loaded reference rows: {len(ref)}")

    # Validate expected columns
    if "slug" not in ref.columns or "outlet" not in ref.columns:
        raise ValueError(f"Reference CSV must contain columns ['slug','outlet'], got {list(ref.columns)}")

    ref["target"] = ref["outlet"].astype(str).str.lower().map(LABEL_MAP)
    ref = ref.rename(columns={"slug": "content"})[["content", "target"]]

    # Current = logged predictions
    cur = load_current_last_n(current_csv, n=500)
    logger.info(f"Loaded current rows: {len(cur)}")

    if "slug" not in cur.columns or "prediction_int" not in cur.columns:
        raise ValueError(
            f"Current CSV must contain columns ['slug','prediction_int'], got {list(cur.columns)}"
        )

    cur = cur.rename(columns={"slug": "content", "prediction_int": "target"})[["content", "target"]]

    # Basic hygiene:
    # IMPORTANT: Do NOT drop missing values here — we want DataQualityPreset() + tests to surface them.
    # Evidently missing-value checks also treat empty strings as missing by default.

    for df_name, df in ("reference", ref), ("current", cur):
        # Normalize text but keep missing values as missing
        df["content"] = df["content"].astype("string").str.strip().str.lower()
        df.loc[df["content"].isna() | (df["content"] == ""), "content"] = pd.NA

        # Ensure target is numeric (nullable int keeps NA)
        df["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")

        # Evidently handles pandas missing values consistently; keep explicit NA values

        # If there are missing content rows, show a couple for sanity
        miss = df[df["content"].isna()]
        if len(miss) > 0:
            logger.warning(f"{df_name} has {len(miss)} rows with missing content. Example rows:\n{miss.head(3)}")

    # Tell Evidently which columns are text and which column is the target
    column_mapping = ColumnMapping(
        target="target",
        text_features=["content"],
        categorical_features=["target"],
        numerical_features=[],
    )

    # Tune drift test to your use case (binary target) so it doesn't fail on small, acceptable shifts
    drift_test = TestColumnDrift(column_name="target", stattest="psi", stattest_threshold=0.3)

    # --- Step: programmatic test output (course requirement) ---
    # A TestSuite with 5 checks on the data.
    missing_suite = TestSuite(
        tests=[
            # 1) Any missing values in current data?
            TestShareOfRowsWithMissingValues(),
            # 2) Target drift between reference and current
            drift_test,
            # 3) Column types are as expected
            TestColumnsType(),
            # 4) No missing values in the text column
            TestColumnNumberOfMissingValues(column_name="content"),
            # 5) Content matches our actual training/current format (lowercase text with spaces)
            TestColumnRegExp(column_name="content", reg_exp=r"^[a-z0-9_\- ]+$"),
        ]
    )


    missing_suite.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    # Print a compact PASS/FAIL summary to the console
    suite_dict = missing_suite.as_dict()
    for t in suite_dict.get("tests", []):
        logger.info(f"Test: {t.get('name', 'unknown')} -> {t.get('status', 'unknown')}")

    # 1) Data quality report (kept separate so text feature generation doesn't coerce NaNs)
    quality_report = Report(metrics=[DataQualityPreset()])
    logger.info("Running Evidently Data Quality report.run(...)")
    quality_report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    quality_path = out_html.parent / "data_quality_report.html"
    quality_report.save_html(str(quality_path))
    logger.info(f"Wrote data quality report to: {quality_path}")

    # 2) Text + target drift report
    drift_report = Report(metrics=[TextEvals(column_name="content"), TargetDriftPreset(columns=["target"])])
    logger.info("Running Evidently Drift report.run(...)")
    drift_report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    drift_path = out_html
    drift_report.save_html(str(drift_path))
    logger.info(f"Wrote drift report to: {drift_path}")

    return drift_path


def main() -> None:
    # Minimal toggle to match your train.py style
    profiling_enabled = True  # set False if you don't want profiling

    def _run() -> None:
        run_drift_report()

    if profiling_enabled:
        _profile_block(_run, out=Path("reports") / "data_drift.prof")
    else:
        _run()


if __name__ == "__main__":
    main()