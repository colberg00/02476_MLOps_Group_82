"""FastAPI application for news outlet classification."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel

from mlops_course_project.data import _url_to_slug_text


LABEL_MAP = {0: "nbc", 1: "fox"}
DEFAULT_MODEL_PATH = Path("models/baseline.joblib")

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
def predict(request: PredictionRequest) -> PredictionResponse:
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

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([slug])[0]
        response.proba_nbc = float(proba[0])
        response.proba_fox = float(proba[1])

    return response
