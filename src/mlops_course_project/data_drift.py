from __future__ import annotations

import cProfile
import pstats
from pathlib import Path
import os
import json

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

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def standardize_current_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize current data into ['content','target'] for Evidently.

    Supports:
    - Legacy CSV rows: slug + prediction_int (+ extra columns time/url/proba_*)
    - JSON event logs: (slug,prediction_int) OR already-standardized (content,target)
    - Optional string prediction column 'prediction' with labels like 'nbc'/'fox'
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["content", "target"])

    out = df.copy()

    # Content
    if "content" not in out.columns and "slug" in out.columns:
        out = out.rename(columns={"slug": "content"})

    # Target
    if "target" not in out.columns:
        if "prediction_int" in out.columns:
            out = out.rename(columns={"prediction_int": "target"})
        elif "prediction" in out.columns:
            pred = out["prediction"].astype("string").str.lower()
            out["target"] = pred.map(LABEL_MAP)

    if "content" not in out.columns or "target" not in out.columns:
        raise ValueError(
            f"Current data must have content+target or slug+prediction_int. Got columns: {list(df.columns)}"
        )

    return out[["content", "target"]].reset_index(drop=True)


# --- Current data logging (cloud-safe pattern) ---
# The prediction API writes one JSON file per request.
# Locally these are stored under `prediction_events/` by default.
# In the cloud, they live in a GCS bucket/prefix.
PREDICTION_EVENTS_DIR = Path(os.getenv("PREDICTION_EVENTS_DIR", "prediction_events"))
PREDICTION_GCS_BUCKET = os.getenv("PREDICTION_GCS_BUCKET", "").strip()
PREDICTION_GCS_PREFIX = os.getenv("PREDICTION_GCS_PREFIX", "predictions/").strip() or "predictions/"


def load_current_last_n(path: str | Path, n: int) -> pd.DataFrame:
    """Load the current (prediction) CSV and keep only the last n rows."""
    df = pd.read_csv(path)
    return standardize_current_df(df.tail(n).reset_index(drop=True))


def _parse_event_row(d: dict) -> dict:
    """Normalize one event dict into the columns we expect."""
    # Accept either the new API keys (slug/prediction_int) or already-standardized (content/target)
    if "content" in d and "target" in d:
        return {"content": d.get("content"), "target": d.get("target")}
    return {"content": d.get("slug"), "target": d.get("prediction_int")}


def load_current_from_local_events(n: int = 500) -> pd.DataFrame:
    """Load the latest N JSON events from the local events directory."""
    if not PREDICTION_EVENTS_DIR.exists():
        raise FileNotFoundError(f"Events directory not found: {PREDICTION_EVENTS_DIR}")

    files = sorted(PREDICTION_EVENTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No event files found in: {PREDICTION_EVENTS_DIR}")

    # Filenames start with ISO timestamp, so lexicographic sort approximates time order
    latest = files[-n:]
    rows: list[dict] = []
    for fp in latest:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            rows.append(_parse_event_row(d))
        except Exception as e:
            logger.warning(f"Skipping unreadable event file {fp}: {e}")

    df = pd.DataFrame(rows)
    return standardize_current_df(df.reset_index(drop=True))


def load_current_from_gcs_events(n: int = 500) -> pd.DataFrame:
    """Load the latest N JSON events from a GCS bucket/prefix."""
    if not PREDICTION_GCS_BUCKET:
        raise ValueError("PREDICTION_GCS_BUCKET is not set")

    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "google-cloud-storage is required to read events from GCS. Install it (e.g., `uv add google-cloud-storage`)."
        ) from e

    client = storage.Client()
    bucket = client.bucket(PREDICTION_GCS_BUCKET)

    prefix = PREDICTION_GCS_PREFIX
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No blobs found in gs://{PREDICTION_GCS_BUCKET}/{prefix}")

    # Blob names start with ISO timestamp, so sorting by name approximates time order
    blobs_sorted = sorted(blobs, key=lambda b: b.name)
    latest = blobs_sorted[-n:]

    rows: list[dict] = []
    for b in latest:
        try:
            d = json.loads(b.download_as_text())
            rows.append(_parse_event_row(d))
        except Exception as e:
            logger.warning(f"Skipping unreadable blob {b.name}: {e}")

    df = pd.DataFrame(rows)
    return standardize_current_df(df.reset_index(drop=True))


def load_current_data(n: int = 500, legacy_csv: str | Path = "prediction_database.csv") -> pd.DataFrame:
    """Load current data from (1) GCS events, (2) local events, or (3) legacy CSV."""
    if PREDICTION_GCS_BUCKET:
        logger.info("Loading current data from GCS event logs")
        return load_current_from_gcs_events(n=n)

    if PREDICTION_EVENTS_DIR.exists() and any(PREDICTION_EVENTS_DIR.glob("*.json")):
        logger.info("Loading current data from local event logs")
        return load_current_from_local_events(n=n)

    # Fallback for older runs
    logger.info("Loading current data from legacy CSV")
    return load_current_last_n(legacy_csv, n=n)


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
    # Current data may come from event logs (local folder or GCS). Only require the CSV if we fall back to it.
    has_local_events = PREDICTION_EVENTS_DIR.exists() and any(PREDICTION_EVENTS_DIR.glob("*.json"))
    if not PREDICTION_GCS_BUCKET and not has_local_events and not current_csv.exists():
        raise FileNotFoundError(
            f"No current data found. Expected either GCS events (set PREDICTION_GCS_BUCKET), local events dir "
            f"({PREDICTION_EVENTS_DIR}), or legacy CSV at: {current_csv}"
        )

    # Reference = training data
    ref = pd.read_csv(reference_csv)
    logger.info(f"Loaded reference rows: {len(ref)}")

    # Validate expected columns
    if "slug" not in ref.columns or "outlet" not in ref.columns:
        raise ValueError(f"Reference CSV must contain columns ['slug','outlet'], got {list(ref.columns)}")

    ref["target"] = ref["outlet"].astype(str).str.lower().map(LABEL_MAP)
    ref = ref.rename(columns={"slug": "content"})[["content", "target"]]

    # Current = logged predictions (cloud-safe JSON events, with legacy CSV fallback)
    cur = load_current_data(n=500, legacy_csv=current_csv)
    logger.info(f"Loaded current rows: {len(cur)}")

    # Standardize schema (handles legacy CSV columns like slug/prediction_int)
    cur = standardize_current_df(cur)

    # Basic hygiene:
    # IMPORTANT: Do NOT drop missing values here — we want DataQualityPreset() + tests to surface them.
    # Evidently missing-value checks also treat empty strings as missing by default.

    for df_name, df in ("reference", ref), ("current", cur):
        # Normalize text but keep missing values as missing.
        # Use plain `object` dtype instead of pandas `string` dtype because Evidently's Column Types
        # test can crash with: "data type 'string' not understood".
        content = df["content"].astype("string").str.strip().str.lower()
        content = content.where(~(content.isna() | (content == "")), pd.NA)

        # Convert to object dtype for maximum compatibility and represent missing as np.nan
        df["content"] = content.astype(object)
        df.loc[pd.isna(df["content"]), "content"] = np.nan

        # Ensure target is numeric (nullable int keeps NA)
        target_num = pd.to_numeric(df["target"], errors="coerce")
        # Stable dtype for Evidently:
        # - If there are missing values -> keep pandas nullable Int64
        # - Otherwise -> use plain int64 (avoids Evidently type-inference issues)
        if target_num.isna().any():
            df["target"] = target_num.astype("Int64")
        else:
            df["target"] = target_num.astype("int64")

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
        name = t.get("name", "unknown")
        status = t.get("status", "unknown")
        logger.info(f"Test: {name} -> {status}")

        if status == "ERROR":
            # Evidently stores details under different keys depending on version
            detail = t.get("description") or t.get("message") or t.get("error") or t.get("details")
            logger.error(f"Test ERROR details for '{name}': {detail}")

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
