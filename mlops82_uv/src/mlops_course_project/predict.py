from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import typer
from joblib import load


LABEL_MAP = {0: "nbc", 1: "fox"}


def _url_to_slug_text(url: str) -> str:
    p = urlparse(url)
    path = p.path.lower().strip("/")
    if not path:
        return ""

    # last segment only (slug)
    slug = path.split("/")[-1]

    # remove print suffix + NBC article id suffix
    slug = re.sub(r"\.print$", "", slug)
    slug = re.sub(r"-rcna\d+$", "", slug)

    # tokenize
    s = slug.replace("-", " ").replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # drop very short slugs (likely section landing pages)
    if len(s.split()) < 3:
        return ""

    # remove outlet tokens if they appear
    s = re.sub(r"\b(fox|foxnews|nbc|nbcnews)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def predict(
    slug: str = typer.Option("", help="Processed slug text (headline proxy)."),
    url: str = typer.Option("", help="Raw article URL. If provided, slug will be extracted from this."),
    model_path: Path = typer.Option(Path("models/baseline.joblib"), help="Path to a trained baseline model."),
    show_proba: bool = typer.Option(True, help="Print predicted probabilities if available."),
) -> None:
    """
    Predict news outlet (fox vs nbc) from URL slug text using the saved baseline model.
    Provide either --slug or --url.
    """
    if not slug and not url:
        raise typer.BadParameter("Provide either --slug or --url.")

    if url:
        slug_extracted = url_to_slug_text(url)
        if not slug_extracted:
            raise typer.BadParameter(
                "Could not extract a valid slug from URL (it may be a section page or too short)."
            )
        slug = slug_extracted

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first (train.py).")

    model = load(model_path)

    # Predict
    pred = int(model.predict([slug])[0])
    outlet = LABEL_MAP.get(pred, str(pred))

    typer.echo(f"slug: {slug}")
    typer.echo(f"prediction: {outlet}")

    # Probabilities (if the classifier supports predict_proba)
    if show_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba([slug])[0]
        # proba index order follows class labels (0 then 1) for sklearn classifiers
        typer.echo(f"proba_nbc: {float(proba[0]):.4f}")
        typer.echo(f"proba_fox: {float(proba[1]):.4f}")


if __name__ == "__main__":
    typer.run(predict)
