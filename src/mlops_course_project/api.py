"""FastAPI application for news outlet classification."""

import os
from pathlib import Path
import csv
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
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


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


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
    background_tasks.add_task(_append_prediction_row, row)

    return response
