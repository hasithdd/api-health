"""
train.py

Training and orchestration script for cybersecurity threat classification.

Responsibilities
----------------
- Load preprocessed datasets and metadata
- Compute class weights for imbalanced data
- Train multiple candidate models
- Evaluate models using security-focused metrics
- Select and persist the best-performing model

This module MUST NOT:
- Perform feature engineering
- Define model architectures
- Implement evaluation logic
"""

import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import polars as pl

from src.model import get_model
from src.evaluation import evaluate_classification, summarize_results

# Configuration

ARTIFACT_DIR = Path("artifacts")
MODEL_OUTPUT_DIR = Path("models")
MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# Utility: Load artifacts

def load_artifacts(artifact_dir: Path) -> Dict[str, Any]:
    """
    Load preprocessing artifacts produced by preparation.py.

    Expected Artifacts
    ------------------
    - train : pl.DataFrame
    - val   : pl.DataFrame
    - test  : pl.DataFrame
    - feature_columns : list[str]
    - label_mapping : dict
    """
    artifacts = {}
    for file in artifact_dir.glob("*.pkl"):
        with open(file, "rb") as f:
            artifacts[file.stem] = pickle.load(f)
    return artifacts


# Class Weight Computation (CRITICAL)

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights inversely proportional to class frequency.

    This emphasizes minority classes (suspicious, malicious) during training.

    Parameters
    ----------
    y : np.ndarray
        Encoded target labels from training set

    Returns
    -------
    class_weight : dict
        Mapping {class_label: weight}
    """
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()

    class_weight = {
        cls: total / (len(unique) * cnt)
        for cls, cnt in zip(unique, counts)
    }
    return class_weight


# Model Training

def train_model(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    model = get_model(model_name, model_config)
    class_weight = model_config.get("class_weight")
    sample_weight = None
    if class_weight is not None:
        sample_weight = np.array([class_weight[y] for y in y_train])

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


# Main Training Pipeline

def main():
    # Load artifacts
    artifacts = load_artifacts(ARTIFACT_DIR)

    train_df: pl.DataFrame = artifacts["train"]
    val_df: pl.DataFrame = artifacts["val"]
    test_df: pl.DataFrame = artifacts["test"]

    feature_columns = artifacts["feature_columns"]

    # Convert Polars → NumPy (model input)
    X_train = train_df.select(feature_columns).to_numpy()
    y_train = train_df.select("threat_label_encoded").to_numpy().ravel()

    X_val = val_df.select(feature_columns).to_numpy()
    y_val = val_df.select("threat_label_encoded").to_numpy().ravel()

    X_test = test_df.select(feature_columns).to_numpy()
    y_test = test_df.select("threat_label_encoded").to_numpy().ravel()

    # Handle class imbalance (TRAIN ONLY)
    class_weight = compute_class_weights(y_train)

    # Model configurations
    experiments = {
        "logistic_regression": {
            "penalty": "l2",
            "solver": "saga",
            "class_weight": class_weight,
            "max_iter": 2000,
            "n_jobs": -1,
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": None,
            "class_weight": class_weight,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        },
        "gradient_boosting": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "device": "gpu",  # change to 'cpu' if no GPU
            "random_state": RANDOM_STATE,
        },
    }

    results = {}

    # Training & Evaluation Loop
    for model_name, config in experiments.items():
        print(f"\nTraining model: {model_name}")

        model = train_model(
            model_name=model_name,
            model_config=config,
            X_train=X_train,
            y_train=y_train,
        )

        metrics = evaluate_classification(
            model=model,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
        )

        results[model_name] = {
            "model": model,
            "metrics": metrics,
        }

    # Model Selection (Security-first)
    summary = summarize_results(results)

    best_model_name = summary["best_model"]
    best_model = results[best_model_name]["model"]

    print(f"\nSelected best model: {best_model_name}")

    # Final Evaluation on Test Set
    final_metrics = evaluate_classification(
        model=best_model,
        X_val=X_test,
        y_val=y_test,
        model_name=f"{best_model_name}_test",
    )

    # Save final model
    with open(MODEL_OUTPUT_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(MODEL_OUTPUT_DIR / "final_metrics.pkl", "wb") as f:
        pickle.dump(final_metrics, f)


if __name__ == "__main__":
    main()
