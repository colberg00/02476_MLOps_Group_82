from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import typer
from joblib import load
import cProfile
import io
import os
import pstats

LABEL_MAP = {0: "nbc", 1: "fox"}

# Profiling is ON by default. Set this to False to disable in-script profiling.
PROFILING_ENABLED = True
# Optional: also disable profiling by setting the env var `PREDICT_NO_PROFILE=1`
PROFILING_DISABLE_ENVVAR = "PREDICT_NO_PROFILE"

PROFILE_OUTPUT = Path("predict.prof")
PROFILE_TEXT_OUTPUT = Path("predict_profile.txt")
PROFILE_SORT_BY = "cumtime"
PROFILE_TOP = 40



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
        slug_extracted = _url_to_slug_text(url)
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
    # Profiling is enabled by default; disable by setting PROFILING_ENABLED=False
    # or by exporting PREDICT_NO_PROFILE=1.
    disable_via_env = os.getenv(PROFILING_DISABLE_ENVVAR, "").strip().lower() in {"1", "true", "yes"}
    profiling_on = PROFILING_ENABLED and not disable_via_env

    if not profiling_on:
        typer.run(predict)
    else:
        prof = cProfile.Profile()
        try:
            prof.enable()
            try:
                typer.run(predict)
            except SystemExit:
                # Typer/Click often raises SystemExit after handling CLI
                pass
        finally:
            prof.disable()

            prof.dump_stats(str(PROFILE_OUTPUT))

            s = io.StringIO()
            stats = pstats.Stats(prof, stream=s)
            stats.strip_dirs().sort_stats(PROFILE_SORT_BY).print_stats(PROFILE_TOP)
            
            PROFILE_TEXT_OUTPUT.write_text(s.getvalue(), encoding="utf-8")
            
            typer.echo(f"Saved cProfile stats to: {PROFILE_OUTPUT}")
            typer.echo(f"Saved cProfile summary to: {PROFILE_TEXT_OUTPUT}")

