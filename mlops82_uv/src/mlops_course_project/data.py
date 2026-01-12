from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import typer
from torch.utils.data import Dataset


NBC_DOMAINS = {"nbcnews.com"}   # label 0
FOX_DOMAINS = {"foxnews.com"}   # label 1


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15  # fraction of total


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    host = re.sub(r"^www\.", "", host)
    return host


def _label_from_url(url: str) -> Optional[int]:
    host = _domain(url)
    for d in NBC_DOMAINS:
        if host == d or host.endswith("." + d):
            return 0
    for d in FOX_DOMAINS:
        if host == d or host.endswith("." + d):
            return 1
    return None


def _url_to_text(url: str) -> str:
    """Turn a URL into a simple token string for baseline models."""
    p = urlparse(url)
    host = re.sub(r"^www\.", "", p.netloc.lower())
    path = p.path.lower()

    # remove a couple of common noise suffixes
    path = re.sub(r"\.print$", "", path)

    s = f"{host} {path}"
    s = re.sub(r"[/\-_]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_raw_file(data_path: Path) -> Path:
    """Accept either a directory or a file; return the file to read."""
    if data_path.is_file():
        return data_path

    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path.resolve()}")

    candidates = []
    for ext in (".csv", ".tsv", ".txt"):
        candidates.extend(sorted(data_path.rglob(f"*{ext}")))

    if not candidates:
        raise FileNotFoundError(
            f"No .csv/.tsv/.txt files found under: {data_path.resolve()}"
        )

    # Heuristic: prefer CSV first if present
    for c in candidates:
        if c.suffix.lower() == ".csv":
            return c
    return candidates[0]


def _read_urls(file_path: Path) -> pd.DataFrame:
    """
    Read a one-column URL file.
    Supports:
      - CSV with header "url"
      - CSV without header (single column)
      - TXT with one URL per line
    """
    suf = file_path.suffix.lower()

    if suf in (".csv", ".tsv"):
        try:
            sep = "\t" if suf == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep)

            # normalize column name
            lower_cols = {c.lower(): c for c in df.columns}
            if "url" in lower_cols:
                df = df.rename(columns={lower_cols["url"]: "url"})
            elif df.shape[1] == 1:
                df.columns = ["url"]
            else:
                # try no-header mode
                df = pd.read_csv(file_path, sep=sep, header=None, names=["url"])
        except Exception:
            df = pd.read_csv(file_path, header=None, names=["url"])
    else:
        # line-based
        urls = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower() == "url":
                continue
            urls.append(line)
        df = pd.DataFrame({"url": urls})

    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.startswith("http", na=False)].copy()
    return df.reset_index(drop=True)


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.df: pd.DataFrame | None = None  # loaded/processed view (optional)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call preprocess() first or load processed files.")
        return len(self.df)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call preprocess() first or load processed files.")
        row = self.df.iloc[index]
        # Return (text, label) as common for NLP baselines
        return row["text"], int(row["label"])

    def preprocess(
        self,
        output_folder: Path,
        cfg: SplitConfig = SplitConfig(),
    ) -> None:
        """Preprocess the raw data and save it to the output folder."""
        raw_file = _find_raw_file(self.data_path)
        raw = _read_urls(raw_file)

        raw["label"] = raw["url"].apply(_label_from_url)
        raw = raw.dropna(subset=["label"]).copy()
        raw["label"] = raw["label"].astype(int)

        raw["text"] = raw["url"].apply(_url_to_text)
        raw = raw[raw["text"].str.len() > 5].copy()
        raw = raw.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

        if raw.empty:
            raise ValueError(
                "After filtering to nbcnews.com and foxnews.com, no rows remained. "
                "Check that your raw file contains those domains."
            )

        # Split: stratified train/val/test without bringing in sklearn
        # (If you prefer sklearn, I can switch to train_test_split.)
        df = raw.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

        # Stratified split implemented via per-class indexing
        def stratified_split(df_in: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
            parts_a = []
            parts_b = []
            for label, g in df_in.groupby("label"):
                g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
                cut = int(round(len(g) * frac))
                parts_b.append(g.iloc[:cut])   # "B" part size ~ frac
                parts_a.append(g.iloc[cut:])   # remainder
            a = pd.concat(parts_a).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            b = pd.concat(parts_b).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            return a, b

        train_val, test = stratified_split(df, cfg.test_size, cfg.seed)
        # val_size is fraction of total; convert to fraction of train_val
        val_frac_of_train_val = cfg.val_size / (1.0 - cfg.test_size)
        train, val = stratified_split(train_val, val_frac_of_train_val, cfg.seed)

        output_folder.mkdir(parents=True, exist_ok=True)
        train[["text", "label"]].to_csv(output_folder / "train.csv", index=False)
        val[["text", "label"]].to_csv(output_folder / "val.csv", index=False)
        test[["text", "label"]].to_csv(output_folder / "test.csv", index=False)

        summary = pd.DataFrame(
            {
                "split": ["train", "val", "test"],
                "rows": [len(train), len(val), len(test)],
                "nbc": [int((train["label"] == 0).sum()), int((val["label"] == 0).sum()), int((test["label"] == 0).sum())],
                "fox": [int((train["label"] == 1).sum()), int((val["label"] == 1).sum()), int((test["label"] == 1).sum())],
            }
        )
        summary.to_csv(output_folder / "split_summary.csv", index=False)

        # Optionally keep a view in memory (useful for debugging)
        self.df = train.reset_index(drop=True)

        print(f"Raw file: {raw_file}")
        print(f"Wrote: {output_folder / 'train.csv'}")
        print(f"Wrote: {output_folder / 'val.csv'}")
        print(f"Wrote: {output_folder / 'test.csv'}")
        print(f"Wrote: {output_folder / 'split_summary.csv'}")
        print(summary.to_string(index=False))


def preprocess(data_path: Path = Path("data/cis519_news_urls"), output_folder: Path = Path("data/processed")) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
