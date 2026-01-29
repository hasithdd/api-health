import pickle
import polars as pl
import numpy as np
from pathlib import Path

from src.preparation import (
    select_and_cast,
    apply_features_inference,
)

# Load artifacts once
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")

with open(MODEL_DIR / "best_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(ARTIFACT_DIR / "feature_columns.pkl", "rb") as f:
    FEATURE_COLUMNS = pickle.load(f)

with open(ARTIFACT_DIR / "bytes_transferred_mean.pkl", "rb") as f:
    MEAN = pickle.load(f)

with open(ARTIFACT_DIR / "bytes_transferred_std.pkl", "rb") as f:
    STD = pickle.load(f)

LABEL_MAP = {0: "benign", 1: "suspicious", 2: "malicious"}

# Inference-safe casting
def cast_inference_df(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([
        pl.col("source_ip").cast(pl.Utf8),
        pl.col("protocol").cast(pl.Categorical),
        pl.col("log_type").cast(pl.Categorical),
        pl.col("bytes_transferred").cast(pl.Int64),
        pl.col("user_agent").cast(pl.Utf8),
        pl.col("request_path").cast(pl.Utf8),
    ])

# Prediction function
def predict(payload: dict):
    df = pl.DataFrame([payload])

    df = cast_inference_df(df)
    df = apply_features_inference(df, MEAN, STD)

    X = df.to_numpy()

    probs = MODEL.predict_proba(X)[0]
    pred = int(np.argmax(probs))

    return {
        "prediction": LABEL_MAP[pred],
        "confidence": float(probs[pred]),
    }
