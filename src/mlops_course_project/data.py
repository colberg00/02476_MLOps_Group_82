from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import typer
from torch.utils.data import Dataset
from typing import Callable
from datasets import load_dataset


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
    if data_path.is_file():
        return data_path

    candidates = []
    for ext in (".csv", ".tsv", ".txt"):
        candidates.extend(sorted(data_path.rglob(f"*{ext}")))

    if not candidates:
        raise FileNotFoundError(f"No .csv/.tsv/.txt files found under: {data_path.resolve()}")

    # prefer csv if available
    for c in candidates:
        if c.suffix.lower() == ".csv":
            return c
    return candidates[0]


def _read_urls(file_path: Path) -> pd.DataFrame:
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

    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.startswith("http", na=False)].copy()
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
        raw_file = _find_raw_file(self.data_path)
        df = _read_urls(raw_file)

        df["outlet"] = df["url"].apply(_outlet_from_url)
        df = df.dropna(subset=["outlet"]).copy()

        df["slug"] = df["url"].apply(_url_to_slug_text)
        df = df[df["slug"] != ""].copy()

        df = df.drop_duplicates(subset=["slug", "outlet"]).reset_index(drop=True)

        if df.empty:
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

        output_folder.mkdir(parents=True, exist_ok=True)
        train[["slug", "outlet"]].to_csv(output_folder / "train.csv", index=False)
        val[["slug", "outlet"]].to_csv(output_folder / "val.csv", index=False)
        test[["slug", "outlet"]].to_csv(output_folder / "test.csv", index=False)

        # quick summary
        summary = pd.DataFrame(
            {
                "split": ["train", "val", "test"],
                "rows": [len(train), len(val), len(test)],
                "fox": [int((train["outlet"] == "fox").sum()), int((val["outlet"] == "fox").sum()), int((test["outlet"] == "fox").sum())],
                "nbc": [int((train["outlet"] == "nbc").sum()), int((val["outlet"] == "nbc").sum()), int((test["outlet"] == "nbc").sum())],
            }
        )
        summary.to_csv(output_folder / "split_summary.csv", index=False)

        print(f"Raw file: {raw_file}")
        print(f"Wrote: {output_folder / 'train.csv'}")
        print(f"Wrote: {output_folder / 'val.csv'}")
        print(f"Wrote: {output_folder / 'test.csv'}")
        print(summary.to_string(index=False))


def download_dataset(
    data_path: Path = Path("data/cis519_news_urls"),
) -> None:
    """Download dataset from Hugging Face and save to local directory."""
    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset('Jia555/cis519_news_urls')
    
    # Create data directory if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Save each split to CSV
    for split_name, split_data in dataset.items():
        df = split_data.to_pandas()
        df.to_csv(data_path / f"{split_name}.csv", index=False)
        print(f"Saved {split_name} to {data_path / f'{split_name}.csv'}")


def preprocess(
    data_path: Path = Path("data/cis519_news_urls"),
    output_folder: Path = Path("data/processed"),
) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_dataset()
    else:
        typer.run(preprocess)
