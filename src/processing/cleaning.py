from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "sample_movies.csv"
LATEST_EXTRACTED_DATA_PATH = ROOT_DIR / "data" / "raw" / "tmdb_animation_movies_latest.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "movies_clean.csv"

NUMERIC_COLUMNS = [
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
]


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    if path:
        source = Path(path)
    else:
        source = LATEST_EXTRACTED_DATA_PATH if LATEST_EXTRACTED_DATA_PATH.exists() else RAW_DATA_PATH
    return pd.read_csv(source)


def normalize_genres(value: object) -> str:
    if pd.isna(value):
        return ""
    genres = [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]
    unique_genres = sorted(dict.fromkeys(genres))
    return ", ".join(unique_genres)


def clean_movie_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [column.strip().lower() for column in cleaned.columns]

    cleaned["title"] = cleaned["title"].fillna("Unknown title").astype(str).str.strip()
    if "overview" not in cleaned.columns:
        cleaned["overview"] = None
    if "director" not in cleaned.columns:
        cleaned["director"] = None
    if "keywords" not in cleaned.columns:
        cleaned["keywords"] = None
    cleaned["release_date"] = pd.to_datetime(cleaned["release_date"], errors="coerce")
    cleaned["genres"] = cleaned["genres"].apply(normalize_genres)
    cleaned["original_language"] = (
        cleaned["original_language"].fillna("unknown").astype(str).str.lower().str[:2]
    )
    cleaned["production_country"] = (
        cleaned["production_country"].fillna("unknown").astype(str).str.upper().str[:2]
    )

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    dedupe_columns = [column for column in ["tmdb_id", "imdb_id", "title", "release_date"] if column in cleaned.columns]
    cleaned = cleaned.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)

    cleaned["budget"] = cleaned["budget"].fillna(cleaned["budget"].median())
    cleaned["revenue"] = cleaned["revenue"].fillna(0)
    cleaned["runtime"] = cleaned["runtime"].fillna(cleaned["runtime"].median())
    cleaned["vote_average"] = cleaned["vote_average"].fillna(cleaned["vote_average"].median())
    cleaned["vote_count"] = cleaned["vote_count"].fillna(0)
    cleaned["popularity"] = cleaned["popularity"].fillna(cleaned["popularity"].median())

    cleaned["roi"] = cleaned["revenue"] / cleaned["budget"].replace(0, pd.NA)
    cleaned["roi"] = cleaned["roi"].fillna(0)
    cleaned["is_animation"] = cleaned["genres"].str.contains("Animation", case=False, na=False).astype(int)

    return cleaned


def merge_sources(dataframes: Iterable[pd.DataFrame]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for dataframe in dataframes:
        current = clean_movie_data(dataframe)
        if merged is None:
            merged = current
            continue
        merged = merged.combine_first(current)

    if merged is None:
        raise ValueError("At least one dataframe is required for merging.")

    return merged.reset_index(drop=True)


def save_clean_data(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    target = Path(path) if path else PROCESSED_DATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def build_clean_dataset(source_path: Optional[Path] = None, output_path: Optional[Path] = None) -> pd.DataFrame:
    df = load_raw_data(source_path)
    cleaned = clean_movie_data(df)
    save_clean_data(cleaned, output_path)
    return cleaned


if __name__ == "__main__":
    dataset = build_clean_dataset()
    print(f"Clean dataset saved with {len(dataset)} rows to {PROCESSED_DATA_PATH}")
