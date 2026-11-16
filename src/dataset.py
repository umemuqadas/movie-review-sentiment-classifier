"""Dataset utilities for the IMDB movie-review corpus.

Source: the official Stanford IMDB corpus (Maas et al., 2011) mirrored on
Hugging Face as ``stanfordnlp/imdb``. It keeps the original, disjoint
train/test split, so every reported metric is computed on data that was never
used to fit the vectorizer or the classifier (no preprocessing data leakage).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

# Primary source (the original ai.stanford.edu host is frequently unreachable).
HF_BASE = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text"
HF_FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
# Fallback: the original tarball from Stanford.
STANFORD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb.tar.gz"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_DIR = DATA_DIR / "imdb_parquet"
ARCHIVE_PATH = DATA_DIR / "aclImdb.tar.gz"
IMDB_DIR = DATA_DIR / "aclImdb"

RANDOM_STATE = 42
LABELS = {0: "negative", 1: "positive"}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        total_mb = max(total // (1024 * 1024), 1)
        done = 0
        with open(dest, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r  {done // (1024 * 1024):>4}/{total_mb} MB "
                        f"({done * 100 // total:3d}%)",
                        end="",
                        flush=True,
                    )
    print()


def _parquet_ready() -> bool:
    return all((PARQUET_DIR / name).is_file() for name in HF_FILES.values())


def download_dataset(force: bool = False) -> Path:
    """Download the IMDB train/test parquet files into ``data/`` if missing."""
    if _parquet_ready() and not force:
        print(f"Dataset already present at {PARQUET_DIR}")
        return PARQUET_DIR

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for split, filename in HF_FILES.items():
        dest = PARQUET_DIR / filename
        if dest.is_file() and not force:
            continue
        print(f"Downloading IMDB {split} split -> {dest.name}")
        _download(f"{HF_BASE}/{filename}", dest)

    if not _parquet_ready():  # last resort: original Stanford tarball
        print(f"Falling back to {STANFORD_URL}")
        if not ARCHIVE_PATH.exists():
            _download(STANFORD_URL, ARCHIVE_PATH)
        import tarfile

        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            try:
                tar.extractall(DATA_DIR, filter="data")
            except TypeError:  # pragma: no cover - Python < 3.12
                tar.extractall(DATA_DIR)
        return IMDB_DIR
    return PARQUET_DIR


def _normalise(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Standardise any source to columns ``text``, ``label``, ``split``."""
    if "text" not in frame.columns:  # aclImdb fallback layout
        raise ValueError(f"Unexpected columns for {split}: {list(frame.columns)}")
    out = pd.DataFrame(
        {
            "text": frame["text"].astype(str),
            "label": frame["label"].astype(int),
        }
    )
    out["split"] = split
    return out


def _load_parquet_split(split: str) -> pd.DataFrame:
    frame = pd.read_parquet(PARQUET_DIR / HF_FILES[split])
    return _normalise(frame, split)


def _load_aclimdb_split(split: str) -> pd.DataFrame:
    frames = []
    for polarity, label in (("pos", 1), ("neg", 0)):
        folder = IMDB_DIR / split / polarity
        paths = sorted(folder.glob("*.txt"), key=lambda p: int(p.stem))
        texts = [p.read_text(encoding="utf-8", errors="ignore") for p in paths]
        frames.append(pd.DataFrame({"text": texts, "label": label}))
    return _normalise(pd.concat(frames, ignore_index=True), split)


def load_imdb(shuffle: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, test_df)`` with columns ``text``, ``label``, ``split``."""
    if _parquet_ready():
        train = _load_parquet_split("train")
        test = _load_parquet_split("test")
    else:
        if not (IMDB_DIR / "train" / "pos").is_dir():
            download_dataset()
        train = _load_aclimdb_split("train")
        test = _load_aclimdb_split("test")

    if shuffle:
        train = train.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        test = test.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return train, test
