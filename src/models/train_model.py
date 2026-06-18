from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.processing.cleaning import PROCESSED_DATA_PATH, build_clean_dataset
from src.processing.feature_engineering import build_feature_table, get_model_inputs

MODEL_PATH = ROOT_DIR / "data" / "processed" / "success_model.joblib"
METRICS_PATH = ROOT_DIR / "data" / "processed" / "model_metrics.json"


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [column for column in X.columns if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def train_candidate_models(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    predictions = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "report": classification_report(y_test, predictions, output_dict=True),
    }


def train_model() -> Dict[str, object]:
    clean_df = build_clean_dataset()
    featured_df = build_feature_table(clean_df)
    X, y, feature_columns = get_model_inputs(featured_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    candidates = train_candidate_models(X_train, y_train, preprocessor)

    best_name = ""
    best_pipeline: Pipeline | None = None
    best_metrics: Dict[str, object] = {"f1": -1}

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        if float(metrics["f1"]) > float(best_metrics["f1"]):
            best_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    if best_pipeline is None:
        raise RuntimeError("Model training failed: no candidate model was fitted.")

    artifact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    metrics_payload = {
        "selected_model": best_name,
        "rows": len(featured_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": best_metrics,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(metrics_payload, stream, indent=2)

    featured_df.to_csv(PROCESSED_DATA_PATH, index=False)
    return metrics_payload


if __name__ == "__main__":
    metrics = train_model()
    print(json.dumps(metrics, indent=2))
