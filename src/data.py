"""Dataset loading and cleaning."""

import re
from pathlib import Path

import pandas as pd

from config import KAGGLE_DATASET

HTML_TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Replace HTML tags with spaces and collapse repeated whitespace.

    Tags are replaced with a space (not removed) so that words separated
    by <br /> do not get glued together.
    """
    text = HTML_TAG.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def load_raw_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """Load the raw reviews, from a local CSV or by downloading with kagglehub."""
    if csv_path is None:
        import kagglehub

        folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        csv_files = sorted(folder.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found in {folder}. "
                "Delete the folder and run again to force a fresh download."
            )
        csv_path = str(csv_files[0])

    return pd.read_csv(csv_path)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, clean the text and encode labels (positive=1, negative=0)."""
    df = df.drop_duplicates(subset="review").reset_index(drop=True)
    df["review"] = df["review"].map(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df
