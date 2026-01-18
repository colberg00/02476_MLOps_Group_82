from pathlib import Path

import pandas as pd

from mlops_course_project.data import MyDataset, PROCESSED_FILES, RAW_FILENAME


def _write_dummy_raw_csv(data_path: Path, n_per_class: int = 25) -> None:
    """
    Create a raw.csv compatible with data.py preprocessing, large enough
    that train/val/test splits are non-empty.
    """
    data_path.mkdir(parents=True, exist_ok=True)

    fox_urls = [
        f"https://www.foxnews.com/politics/blinken-meets-with-qatars-prime-minister-{i}"
        for i in range(n_per_class)
    ]
    nbc_urls = [
        f"https://www.nbcnews.com/politics/immigration/abolish-ice-democratic-messaging-{i}"
        for i in range(n_per_class)
    ]

    df = pd.DataFrame({"url": fox_urls + nbc_urls})
    df.to_csv(data_path / RAW_FILENAME, index=False)


def test_preprocess_creates_expected_files(tmp_path: Path) -> None:
    data_path = tmp_path / "raw_data"
    output_path = tmp_path / "processed"

    _write_dummy_raw_csv(data_path, n_per_class=10)

    dataset = MyDataset(data_path)
    dataset.preprocess(output_path)

    for fname in PROCESSED_FILES:
        assert (output_path / fname).exists(), f"Missing output file: {fname}"


def test_processed_csv_has_required_columns(tmp_path: Path) -> None:
    """
    Validate that train/val/test CSVs have 'slug' and 'outlet' columns
    and are non-empty for a sufficiently large dummy dataset.
    """
    data_path = tmp_path / "raw_data"
    output_path = tmp_path / "processed"

    _write_dummy_raw_csv(data_path, n_per_class=25)  # 50 rows total

    dataset = MyDataset(data_path)
    dataset.preprocess(output_path)

    for split in ("train.csv", "val.csv", "test.csv"):
        df = pd.read_csv(output_path / split)

        assert "slug" in df.columns
        assert "outlet" in df.columns

        # With enough rows, all splits should be non-empty
        assert len(df) > 0, f"{split} unexpectedly empty"

        # Slugs should be non-empty strings
        assert df["slug"].map(lambda x: isinstance(x, str) and len(x.strip()) > 0).all()

        # Outlets must be fox or nbc
        assert set(df["outlet"].unique()).issubset({"fox", "nbc"})


def test_slug_extraction_removed_domains(tmp_path: Path) -> None:
    data_path = tmp_path / "raw_data"
    output_path = tmp_path / "processed"

    _write_dummy_raw_csv(data_path, n_per_class=10)

    dataset = MyDataset(data_path)
    dataset.preprocess(output_path)

    df = pd.read_csv(output_path / "train.csv")

    forbidden = {"fox", "foxnews", "nbc", "nbcnews"}
    for slug in df["slug"]:
        words = set(str(slug).split())
        assert words.isdisjoint(forbidden)
