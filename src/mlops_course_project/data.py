from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import typer
from torch.utils.data import Dataset
from datasets import load_dataset
from loguru import logger

from mlops_course_project import setup_logging


setup_logging("data")


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15  # fraction of total


FOX_DOMAINS = {"foxnews.com"}
NBC_DOMAINS = {"nbcnews.com"}


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    host = re.sub(r"^www\.", "", host)
    return host


def _outlet_from_url(url: str) -> Optional[str]:
    host = _domain(url)
    for d in FOX_DOMAINS:
        if host == d or host.endswith("." + d):
            return "fox"
    for d in NBC_DOMAINS:
        if host == d or host.endswith("." + d):
            return "nbc"
    return None


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


def _find_raw_file(data_path: Path) -> Path:
    """Find raw data file in directory.

    Args:
        data_path: Path to directory or file.

    Returns:
        Path to the raw data file.

    Raises:
        FileNotFoundError: If no data file found.
    """
    if data_path.is_file():
        logger.debug(f"Found raw file at {data_path}")
        return data_path

    logger.debug(f"Searching for raw files in {data_path}")
    candidates = []
    for ext in (".csv", ".tsv", ".txt"):
        candidates.extend(sorted(data_path.rglob(f"*{ext}")))

    if not candidates:
        logger.error(f"No .csv/.tsv/.txt files found under: {data_path.resolve()}")
        raise FileNotFoundError(f"No .csv/.tsv/.txt files found under: {data_path.resolve()}")

    logger.debug(f"Found {len(candidates)} candidate file(s)")
    # prefer csv if available
    for c in candidates:
        if c.suffix.lower() == ".csv":
            logger.debug(f"Selecting CSV file: {c}")
            return c
    logger.debug(f"Using file: {candidates[0]}")
    return candidates[0]


def _read_urls(file_path: Path) -> pd.DataFrame:
    """Read URLs from file.

    Args:
        file_path: Path to the file containing URLs.

    Returns:
        DataFrame with URL column.
    """
    logger.info(f"Reading URLs from {file_path}")
    suf = file_path.suffix.lower()
    if suf in (".csv", ".tsv"):
        sep = "\t" if suf == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
        lower_cols = {c.lower(): c for c in df.columns}
        if "url" in lower_cols:
            df = df.rename(columns={lower_cols["url"]: "url"})
        elif df.shape[1] == 1:
            df.columns = ["url"]
        else:
            df = pd.read_csv(file_path, sep=sep, header=None, names=["url"])
    else:
        urls = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.lower() == "url":
                continue
            urls.append(line)
        df = pd.DataFrame({"url": urls})

    logger.debug(f"Loaded {len(df)} URLs from file")
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.startswith("http", na=False)].copy()
    logger.info(f"Found {len(df)} valid URLs after filtering")
    return df.reset_index(drop=True)


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        raise NotImplementedError("Length is defined on processed splits, not raw URLs.")

    def __getitem__(self, index: int):
        raise NotImplementedError("Use processed splits for training (CSV -> Dataset).")

    def preprocess(self, output_folder: Path, cfg: SplitConfig = SplitConfig()) -> None:
        preferred = self.data_path / RAW_FILENAME
        """Preprocess dataset: filter by outlet, extract slugs, and create splits.
        
        Args:
            output_folder: Directory to save processed splits.
            cfg: Configuration for train/val/test split ratios.
            
        Raises:
            ValueError: If no data remains after filtering.
        """
        logger.info("Starting preprocessing")
        raw_file = preferred if preferred.exists() else _find_raw_file(self.data_path)
        df = _read_urls(raw_file)
        logger.info(f"Raw data: {len(df)} rows")

        df["outlet"] = df["url"].apply(_outlet_from_url)
        initial_count = len(df)
        df = df.dropna(subset=["outlet"]).copy()
        logger.info(f"After outlet filtering: {len(df)} rows (removed {initial_count - len(df)})")

        df["slug"] = df["url"].apply(_url_to_slug_text)
        initial_count = len(df)
        df = df[df["slug"] != ""].copy()
        logger.info(f"After slug extraction: {len(df)} rows (removed {initial_count - len(df)})")

        initial_count = len(df)
        df = df.drop_duplicates(subset=["slug", "outlet"]).reset_index(drop=True)
        logger.info(f"After deduplication: {len(df)} rows (removed {initial_count - len(df)})")

        if df.empty:
            logger.error("No rows left after filtering. Check domain filters and slug extraction rules.")
            raise ValueError("No rows left after filtering. Check domain filters and slug extraction rules.")

        # stratified split without sklearn
        df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

        def stratified_split(df_in: pd.DataFrame, frac: float, seed: int):
            a_parts, b_parts = [], []
            for outlet, g in df_in.groupby("outlet"):
                g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
                cut = int(round(len(g) * frac))
                b_parts.append(g.iloc[:cut])
                a_parts.append(g.iloc[cut:])
            a = pd.concat(a_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            b = pd.concat(b_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            return a, b

        train_val, test = stratified_split(df, cfg.test_size, cfg.seed)
        val_frac_of_train_val = cfg.val_size / (1.0 - cfg.test_size)
        train, val = stratified_split(train_val, val_frac_of_train_val, cfg.seed)

        logger.debug(f"Split sizes - train: {len(train)}, val: {len(val)}, test: {len(test)}")

        output_folder.mkdir(parents=True, exist_ok=True)
        train[["slug", "outlet"]].to_csv(output_folder / "train.csv", index=False)
        logger.info(f"Wrote train split ({len(train)} rows) to {output_folder / 'train.csv'}")
        val[["slug", "outlet"]].to_csv(output_folder / "val.csv", index=False)
        logger.info(f"Wrote val split ({len(val)} rows) to {output_folder / 'val.csv'}")
        test[["slug", "outlet"]].to_csv(output_folder / "test.csv", index=False)
        logger.info(f"Wrote test split ({len(test)} rows) to {output_folder / 'test.csv'}")

        # quick summary
        summary = pd.DataFrame(
            {
                "split": ["train", "val", "test"],
                "rows": [len(train), len(val), len(test)],
                "fox": [
                    int((train["outlet"] == "fox").sum()),
                    int((val["outlet"] == "fox").sum()),
                    int((test["outlet"] == "fox").sum()),
                ],
                "nbc": [
                    int((train["outlet"] == "nbc").sum()),
                    int((val["outlet"] == "nbc").sum()),
                    int((test["outlet"] == "nbc").sum()),
                ],
            }
        )
        summary.to_csv(output_folder / "split_summary.csv", index=False)
        logger.info(f"Wrote summary to {output_folder / 'split_summary.csv'}")
        logger.info("Preprocessing complete")


def download_dataset(
    data_path: Path = Path("data/cis519_news_urls"),
) -> None:
    """Download dataset from Hugging Face and save to local directory.

    Args:
        data_path: Directory to save the downloaded dataset.
    """
    logger.info("Starting dataset download from Hugging Face (Jia555/cis519_news_urls)")
    try:
        dataset = load_dataset("Jia555/cis519_news_urls")
        logger.info("Successfully downloaded dataset")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    # Create data directory if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created data directory: {data_path}")

    # Save each split to CSV
    for split_name, split_data in dataset.items():
        logger.debug(f"Converting {split_name} split to DataFrame")
        df = split_data.to_pandas()
        output_file = data_path / f"{split_name}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {split_name} split ({len(df)} rows) to {output_file}")


def preprocess(
    data_path: Path = Path("data/cis519_news_urls"),
    output_folder: Path = Path("data/processed"),
) -> None:
    logger.info(f"Preprocessing dataset from {data_path} to {output_folder}")
    try:
        dataset = MyDataset(data_path)
        dataset.preprocess(output_folder)
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise


app = typer.Typer(help="Dataset utilities")


@app.command()
def download(
    data_path: Path = Path("data/cis519_news_urls"),
) -> None:
    """Download dataset from Hugging Face and save to local directory."""
    download_dataset(data_path)


@app.command()
def run_preprocess(
    data_path: Path = Path("data/cis519_news_urls"),
    output_folder: Path = Path("data/processed"),
) -> None:
    """Preprocess raw URL data into train/val/test splits."""
    preprocess(data_path=data_path, output_folder=output_folder)


RAW_FILENAME = "raw.csv"
PROCESSED_FILES = ("train.csv", "val.csv", "test.csv", "split_summary.csv")


def _raw_ready(data_path: Path) -> bool:
    """Return True if we have a usable raw file on disk."""
    raw_file = data_path / RAW_FILENAME
    return raw_file.exists() and raw_file.is_file() and raw_file.stat().st_size > 0


def _processed_ready(output_folder: Path) -> bool:
    """Return True if preprocessing outputs exist."""
    return all((output_folder / f).exists() for f in PROCESSED_FILES)


def ensure_downloaded(data_path: Path = Path("data/cis519_news_urls")) -> None:
    """
    Download only if raw.csv is missing.
    Writes a single combined raw.csv so preprocess is deterministic.
    """
    if _raw_ready(data_path):
        logger.info(f"Raw dataset already present: {(data_path / RAW_FILENAME).resolve()}")
        return

    logger.info("Raw dataset missing -> downloading from Hugging Face...")
    try:
        dataset = load_dataset("Jia555/cis519_news_urls")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    data_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created data directory: {data_path}")

    # Combine all splits into one raw file (so preprocess doesn't pick a random split)
    dfs = []
    for split_name, split_data in dataset.items():
        df = split_data.to_pandas()
        dfs.append(df)
        logger.debug(f"Loaded split '{split_name}' with {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(dfs)} splits into {len(combined)} total rows")

    # Optional: if HF dataset doesn't have 'url' named exactly, try to normalize
    lower_cols = {c.lower(): c for c in combined.columns}
    if "url" in lower_cols and lower_cols["url"] != "url":
        combined = combined.rename(columns={lower_cols["url"]: "url"})
        logger.debug("Normalized 'url' column name")

    out_file = data_path / RAW_FILENAME
    combined.to_csv(out_file, index=False)
    logger.info(f"Saved combined raw dataset to: {out_file.resolve()} ({len(combined)} rows)")


def run_pipeline(
    data_path: Path = Path("data/cis519_news_urls"),
    output_folder: Path = Path("data/processed"),
    force_download: bool = False,
    force_preprocess: bool = False,
) -> None:
    logger.info(f"Starting pipeline: force_download={force_download}, force_preprocess={force_preprocess}")

    if force_download:
        logger.info("Force download enabled - deleting existing raw.csv")
        # Force re-download by deleting raw.csv if present
        raw = data_path / RAW_FILENAME
        if raw.exists():
            raw.unlink()
            logger.debug(f"Deleted: {raw}")
        ensure_downloaded(data_path)
    else:
        ensure_downloaded(data_path)

    if force_preprocess or not _processed_ready(output_folder):
        if _processed_ready(output_folder) and force_preprocess:
            logger.warning("Forcing preprocess: overwriting existing processed files")
        logger.info("Starting preprocessing step")
        preprocess(data_path=data_path, output_folder=output_folder)
    else:
        logger.info(f"Processed data already present in: {output_folder.resolve()}")
        logger.debug(f"Found: {', '.join(PROCESSED_FILES)}")

    logger.info("Pipeline completed successfully")


def main(
    data_path: Path = Path("data/cis519_news_urls"),
    output_folder: Path = Path("data/processed"),
    force_download: bool = False,
    force_preprocess: bool = False,
) -> None:
    run_pipeline(
        data_path=data_path,
        output_folder=output_folder,
        force_download=force_download,
        force_preprocess=force_preprocess,
    )


if __name__ == "__main__":
    app()
