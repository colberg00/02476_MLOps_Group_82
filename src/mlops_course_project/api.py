"""FastAPI application for news outlet classification."""

import os
import time
import json
import uuid
from pathlib import Path
import csv
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from joblib import load
from pydantic import BaseModel

from mlops_course_project.data import _url_to_slug_text


LABEL_MAP = {0: "nbc", 1: "fox"}
DEFAULT_MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parent / "models" / "baseline.joblib"),
    )
)

PREDICTION_DB_PATH = Path(os.getenv("PREDICTION_DB_PATH", "prediction_database.csv"))

# --- Optional cloud logging (GCS) ---
# If PREDICTION_GCS_BUCKET is set, predictions are written as one JSON object per request to GCS.
# This avoids concurrent-write issues that happen when multiple Cloud Run instances append to one CSV.
PREDICTION_GCS_BUCKET = os.getenv("PREDICTION_GCS_BUCKET", "").strip()
PREDICTION_GCS_PREFIX = os.getenv("PREDICTION_GCS_PREFIX", "predictions/").strip() or "predictions/"

# Derived runtime mode (helps teammates verify what is enabled)
LOGGING_BACKEND = "gcs" if PREDICTION_GCS_BUCKET else "local_csv"

# Always write one JSON event per request (cloud-safe pattern). Locally, events are written to this directory.
PREDICTION_EVENTS_DIR = Path(os.getenv("PREDICTION_EVENTS_DIR", "prediction_events"))


# --- Monitoring (drift + data quality reports) ---
MONITORING_REPORT_PATH = Path(os.getenv("MONITORING_REPORT_PATH", "reports/data_drift_report.html"))
MONITORING_QUALITY_REPORT_PATH = Path(os.getenv("MONITORING_QUALITY_REPORT_PATH", "reports/data_quality_report.html"))
MONITORING_REFERENCE_CSV = Path(os.getenv("MONITORING_REFERENCE_CSV", "data/processed/train.csv"))
MONITORING_CURRENT_CSV_FALLBACK = Path(os.getenv("MONITORING_CURRENT_CSV_FALLBACK", "prediction_database.csv"))
MONITORING_REFRESH_SECONDS = int(os.getenv("MONITORING_REFRESH_SECONDS", "300"))  # reuse cached report for 5 min


def _append_prediction_row(row: dict) -> None:
    """Append one prediction row to CSV (create file + header if missing)."""
    PREDICTION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "time",
        "slug",
        "url",
        "prediction_int",
        "prediction",
        "proba_nbc",
        "proba_fox",
    ]

    file_exists = PREDICTION_DB_PATH.exists()
    with PREDICTION_DB_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # Ensure only known fields are written
        writer.writerow({k: row.get(k) for k in fieldnames})


def _write_prediction_event(row: dict) -> None:
    """Write one prediction event per request.

    Cloud-safe pattern:
    - If PREDICTION_GCS_BUCKET is set -> upload JSON event to GCS.
    - Otherwise -> write JSON event to local disk.

    For backwards compatibility with existing local pipelines, we also mirror to the local CSV
    when running without GCS.
    """
    # Keep the human-readable ISO timestamp in the event payload,
    # but use a filesystem-safe timestamp for the filename (Windows cannot handle ':' in filenames).
    ts_payload = row.get("time") or datetime.now(timezone.utc).isoformat()
    ts_file = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # e.g. 20260123T111303913374Z
    event_name = f"{ts_file}_{uuid.uuid4().hex}.json"

    if PREDICTION_GCS_BUCKET:
        # Import lazily so local dev works without the dependency unless cloud logging is enabled.
        try:
            from google.cloud import storage  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "google-cloud-storage is required for GCS logging. Install it (e.g., `uv add google-cloud-storage`)."
            ) from e

        client = storage.Client()
        bucket = client.bucket(PREDICTION_GCS_BUCKET)

        # Ensure prefix ends with '/'
        prefix = PREDICTION_GCS_PREFIX
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        object_name = f"{prefix}{event_name}"
        blob = bucket.blob(object_name)
        payload = json.dumps(row, ensure_ascii=False)
        blob.upload_from_string(payload, content_type="application/json")
        return

    # Local mode: write events to disk
    PREDICTION_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    event_path = PREDICTION_EVENTS_DIR / event_name

    # Ensure the payload has the ISO timestamp (human-readable + timezone-aware)
    payload = dict(row)
    payload["time"] = ts_payload
    event_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Backwards compatible local mirror (your existing drift script can keep reading the CSV)
    _append_prediction_row(payload)


app = FastAPI(
    title="News Outlet Classifier",
    description="Classify news headlines as either NBC or Fox News",
    version="0.0.1",
)

_model = None


def get_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Load and cache the model."""
    global _model
    if _model is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")
        _model = load(model_path)
    return _model


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""

    slug: Optional[str] = None
    url: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint."""

    slug: str
    prediction: str
    proba_nbc: Optional[float] = None
    proba_fox: Optional[float] = None


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {"message": "News Outlet Classifier API", "version": "0.0.1"}


def _should_regenerate_report(path: Path, refresh_seconds: int) -> bool:
    """Return True if the report file is missing or older than refresh_seconds."""
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age > refresh_seconds


@app.get("/health")
def health():
    """Health check endpoint."""
    # Keep the health endpoint deliberately minimal so tests and simple checks only depend on status.
    return {"status": "healthy"}


@app.get("/monitoring/report", response_class=HTMLResponse)
def monitoring_report(force: bool = False):
    """Generate (if needed) and return the drift report HTML.

    - Uses cached report if it is fresh enough (MONITORING_REFRESH_SECONDS)
    - Set `force=true` to regenerate immediately
    """
    try:
        # Import lazily so normal prediction requests don't pay Evidently import time.
        from mlops_course_project.data_drift import run_drift_report

        MONITORING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if force or _should_regenerate_report(MONITORING_REPORT_PATH, MONITORING_REFRESH_SECONDS):
            run_drift_report(
                reference_csv=MONITORING_REFERENCE_CSV,
                current_csv=MONITORING_CURRENT_CSV_FALLBACK,
                out_html=MONITORING_REPORT_PATH,
            )

        html = MONITORING_REPORT_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html, status_code=200)
    except FileNotFoundError as e:
        # Typically reference CSV missing in the container or no current data yet
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate monitoring report: {e}")


@app.get("/monitoring/quality", response_class=HTMLResponse)
def monitoring_quality(force: bool = False):
    """Generate (if needed) and return the data quality report HTML.

    - Uses cached report if it is fresh enough (MONITORING_REFRESH_SECONDS)
    - Set `force=true` to regenerate immediately
    """
    try:
        # Import lazily so normal prediction requests don't pay Evidently import time.
        from mlops_course_project.data_drift import run_drift_report

        MONITORING_QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # `run_drift_report` generates BOTH drift + quality reports.
        if force or _should_regenerate_report(MONITORING_QUALITY_REPORT_PATH, MONITORING_REFRESH_SECONDS):
            run_drift_report(
                reference_csv=MONITORING_REFERENCE_CSV,
                current_csv=MONITORING_CURRENT_CSV_FALLBACK,
                out_html=MONITORING_REPORT_PATH,
            )

        html = MONITORING_QUALITY_REPORT_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html, status_code=200)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate data quality report: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    """Predict news outlet (fox vs nbc) from URL slug text.

    Args:
        request: Request containing either a slug or URL.

    Returns:
        Prediction response with outlet classification and probabilities.
    """
    if not request.slug and not request.url:
        raise HTTPException(status_code=400, detail="Provide either 'slug' or 'url'.")

    slug = request.slug
    if request.url:
        slug = _url_to_slug_text(request.url)
        if not slug:
            raise HTTPException(
                status_code=400,
                detail="Could not extract a valid slug from URL (it may be a section page or too short).",
            )

    try:
        model = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    pred = int(model.predict([slug])[0])
    outlet = LABEL_MAP.get(pred, str(pred))

    response = PredictionResponse(slug=slug, prediction=outlet)

    proba_nbc = None
    proba_fox = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([slug])[0]
        proba_nbc = float(proba[0])
        proba_fox = float(proba[1])
        response.proba_nbc = proba_nbc
        response.proba_fox = proba_fox

    # Log request + prediction after response is returned
    row = {
        "time": datetime.now(timezone.utc).isoformat(),
        "slug": slug,
        "url": request.url,
        "prediction_int": pred,
        "prediction": outlet,
        "proba_nbc": proba_nbc,
        "proba_fox": proba_fox,
    }
    background_tasks.add_task(_write_prediction_event, row)

    return response
