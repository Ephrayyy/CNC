from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

GENRE_FLAGS = [
    "Animation",
    "Adventure",
    "Family",
    "Drama",
    "Fantasy",
    "Comedy",
    "Action",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["release_year"] = featured["release_date"].dt.year.fillna(0).astype(int)
    featured["release_month"] = featured["release_date"].dt.month.fillna(0).astype(int)
    featured["release_quarter"] = featured["release_date"].dt.quarter.fillna(0).astype(int)
    featured["is_holiday_window"] = featured["release_month"].isin([6, 7, 11, 12]).astype(int)
    return featured


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["budget_musd"] = featured["budget"] / 1_000_000
    featured["revenue_musd"] = featured["revenue"] / 1_000_000
    featured["profit_musd"] = (featured["revenue"] - featured["budget"]) / 1_000_000
    featured["vote_count_log"] = np.log1p(featured["vote_count"])
    featured["popularity_log"] = np.log1p(featured["popularity"])
    featured["num_genres"] = featured["genres"].str.split(",").apply(lambda values: len([v for v in values if v.strip()]))

    for genre in GENRE_FLAGS:
        column_name = f"genre_{genre.lower()}"
        featured[column_name] = featured["genres"].str.contains(genre, case=False, na=False).astype(int)

    return featured


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    targeted = df.copy()
    targeted["success_score"] = (
        0.45 * targeted["roi"]
        + 0.25 * targeted["vote_average"]
        + 0.15 * targeted["popularity_log"]
        + 0.15 * targeted["vote_count_log"]
    )
    threshold = targeted["success_score"].median()
    targeted["success_label"] = (targeted["success_score"] >= threshold).astype(int)
    return targeted


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_temporal_features(df)
    featured = add_business_features(featured)
    featured = add_target(featured)
    return featured


def get_model_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_columns = [
        "budget_musd",
        "runtime",
        "vote_average",
        "vote_count_log",
        "popularity_log",
        "num_genres",
        "release_year",
        "release_month",
        "release_quarter",
        "is_holiday_window",
        "is_animation",
        "genre_animation",
        "genre_adventure",
        "genre_family",
        "genre_drama",
        "genre_fantasy",
        "genre_comedy",
        "genre_action",
        "original_language",
        "production_country",
    ]
    X = df[feature_columns]
    y = df["success_label"]
    return X, y, feature_columns
