import joblib
import pickle
import polars as pl
import numpy as np
from pathlib import Path

from src.preparation import apply_features_inference

# Load artifacts once
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")

def load_model():
    """Load model from joblib or pkl format, converting if needed."""
    joblib_path = MODEL_DIR / "best_model.joblib"
    pkl_path = MODEL_DIR / "best_model.pkl"
    
    # Prefer joblib if exists
    if joblib_path.exists():
        return joblib.load(joblib_path)
    
    # Fall back to pkl and convert
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        
        # Save as joblib for future loads (if writable)
        try:
            joblib.dump(model, joblib_path, compress=3)
        except (PermissionError, OSError):
            pass  # Read-only filesystem (e.g., Docker)
        
        return model
    
    raise FileNotFoundError(
        f"No model found. Expected {joblib_path} or {pkl_path}"
    )

# Load model once at startup
MODEL = load_model()

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
