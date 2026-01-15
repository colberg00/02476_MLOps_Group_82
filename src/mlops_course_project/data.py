from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
import typer
from torch.utils.data import Dataset

from typing import Union

import pandas as pd


# Hugging Face direct download URL for the CSV
HF_CSV_URL = (
    "https://huggingface.co/datasets/Jia555/cis519_news_urls/resolve/main/url_only_data_extra.csv"
)

# Output filename inside output_folder
OUTPUT_FILENAME = "processed.csv"

def _download_csv_if_missing(dest: Union[str, Path]) -> None:
    """
    Download the dataset CSV from Hugging Face if it does not exist at `dest`.

    Parameters
    ----------
    dest:
        File path where the raw CSV should be stored (e.g. data/raw/url_only_data_extra.csv).
        Can be a str or pathlib.Path.
    """
    dest = Path(dest)  # force Path, even if caller passes a string

    # Guardrail: this function expects a FILE path, not a directory
    if dest.exists() and dest.is_dir():
        raise ValueError(
            f"_download_csv_if_missing expected a file path, but got a directory: {dest}. "
            "Pass the full CSV path (e.g. data/raw/<file>.csv)."
        )

    # If file already exists and is non-empty, do nothing
    if dest.exists() and dest.stat().st_size > 0:
        return

    # Ensure parent folder exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Some servers are happier if we set a user-agent header
    req = Request(HF_CSV_URL, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(req) as resp, dest.open("wb") as f:
        chunk_size = 1024 * 1024  # 1 MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def _extract_news_source(url: str) -> Optional[str]:
    """
    Extract the news source from the URL domain.

    Returns
    -------
    'foxnews' or 'nbcnews' if detected, otherwise None.
    """
    if not isinstance(url, str):
        return None

    u = url.strip()
    if not u:
        return None

    # Ensure urlparse works if scheme is missing
    if "://" not in u:
        u = "https://" + u

    domain = (urlparse(u).netloc or "").lower()

    if "foxnews.com" in domain:
        return "foxnews"
    if "nbcnews.com" in domain:
        return "nbcnews"
    return None


def _url_to_keywords(url: str) -> str:
    """
    Convert a URL into whitespace-separated "keywords".

    Strategy:
    - Take the URL path (e.g., /world/iran/iran-trump-nuclear-...-rcna214328)
    - Lowercase
    - Replace separators with spaces (/, -, _, ., ?, =, & etc.)
    - Keep only alphanumeric tokens
    """
    if not isinstance(url, str):
        return ""

    u = url.strip()
    if not u:
        return ""

    if "://" not in u:
        u = "https://" + u

    parsed = urlparse(u)

    # Include domain to help distinguish common patterns if you want;
    # here we mainly rely on path, but domain tokens don't hurt.
    text = f"{parsed.netloc} {parsed.path}".lower()

    # Replace common separators with spaces
    for sep in ["/", "-", "_", ".", "?", "=", "&", "%", ":", "#", "+"]:
        text = text.replace(sep, " ")

    # Split and keep alphanumeric tokens only
    tokens = [t for t in text.split() if t.isalnum()]

    # Drop common junk tokens
    junk = {"www", "com", "http", "https"}
    tokens = [t for t in tokens if t not in junk]

    return " ".join(tokens)




class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Union[str, Path]) -> None:
        # Accept str or Path, normalize immediately
        p = Path(data_path)

        # If a directory was passed (e.g. "data/raw"), resolve it to a CSV inside it
        if p.exists() and p.is_dir():
            preferred = p / "url_only_data_extra.csv"
            if preferred.exists():
                p = preferred
            else:
                csvs = sorted(p.glob("*.csv"))
                if len(csvs) == 1:
                    p = csvs[0]
                elif len(csvs) == 0:
                    raise ValueError(
                        f"MyDataset got a directory ({p}) but found no .csv files inside it. "
                        f"Pass the full CSV path (e.g. {p}/<file>.csv)."
                    )
                else:
                    raise ValueError(
                        f"MyDataset got a directory ({p}) with multiple .csv files. "
                        f"Pass the exact CSV file path. Found: {[c.name for c in csvs[:10]]}"
                    )

        # Now p should be the raw CSV file path
        self.data_path = p

        # Ensure raw file exists (download if needed)
        _download_csv_if_missing(self.data_path)

        # Load the raw CSV once (cheap for ~40k rows)
        df = pd.read_csv(self.data_path)

        # Basic validation
        if "url" not in df.columns:
            raise ValueError(
                f"Expected a column named 'url' in {self.data_path}. Found columns: {list(df.columns)}"
            )

        # Clean and normalize
        df["url"] = df["url"].astype(str).str.strip()

        # Defensive: if any row accidentally contains multiple URLs separated by whitespace,
        # split and explode to separate rows.
        df["url"] = df["url"].str.split(r"\s+")
        df = df.explode("url", ignore_index=True)
        df["url"] = df["url"].astype(str).str.strip()
        df = df[df["url"].ne("")].reset_index(drop=True)

        # Create label and features
        df["news_source"] = df["url"].apply(_extract_news_source)
        df = df.dropna(subset=["news_source"]).reset_index(drop=True)
        df["keywords"] = df["url"].apply(_url_to_keywords)

        self.df = df

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int):
        """
        Return a given sample from the dataset.

        Returns a dict so it is flexible for later training code.
        """
        row = self.df.iloc[index]
        return {
            "url": row["url"],
            "news_source": row["news_source"],
            "keywords": row["keywords"],
        }

    def process(self, output_folder: Union[str, Path]) -> None:
        """Process the raw data and save it to the output folder."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        out_path = output_folder / OUTPUT_FILENAME

        # Save the full processed table (url + label + keywords)
        self.df.to_csv(out_path, index=False)


if __name__ == "__main__":
    typer.run(process)