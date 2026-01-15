import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import polars as pl
import joblib

from src.model import get_model
from src.evaluation import evaluate_classification, summarize_results

from src.validation import (
    plot_confusion,
    plot_precision_recall,
    plot_roc,
    plot_feature_importance,
)

from src.tuning import tune_xgboost

from src.validation import compare_splits

ARTIFACT_DIR = Path("artifacts")
MODEL_OUTPUT_DIR = Path("models")
MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

def load_artifacts(artifact_dir: Path) -> Dict[str, Any]:
    """
    Load preprocessing artifacts produced by preparation.py
    """
    artifacts = {}
    for file in artifact_dir.glob("*.pkl"):
        with open(file, "rb") as f:
            artifacts[file.stem] = pickle.load(f)
    return artifacts

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights inversely proportional to class frequency.
    """

    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    class_weight = {
        int(cls): float(total / (len(unique) * cnt))
        for cls, cnt in zip(unique, counts)
    }
    return class_weight

def train_model(model_name, model_config, X_train, y_train):
    """
    Docstring for train_model
    
    :param model_name: Description
    :param model_config: Description
    :param X_train: Description
    :param y_train: Description
    """
    model = get_model(model_name, model_config)

    with joblib.parallel_backend('loky', n_jobs=10):
        if model_name == "gradient_boosting":
            cw_dict = model_config.get("class_weight")
            if cw_dict:
                row_weights = np.array([cw_dict[y] for y in y_train])
                model.fit(X_train, y_train, sample_weight=row_weights)
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
            
    return model

def main():
    artifacts = load_artifacts(ARTIFACT_DIR)

    train_df: pl.DataFrame = artifacts["train"]
    val_df: pl.DataFrame = artifacts["val"]
    test_df: pl.DataFrame = artifacts["test"]
    feature_columns = artifacts["feature_columns"]

    X_train = train_df.select(feature_columns).to_numpy()
    y_train = train_df.select("threat_label_encoded").to_numpy().ravel()
    X_val = val_df.select(feature_columns).to_numpy()
    y_val = val_df.select("threat_label_encoded").to_numpy().ravel()
    X_test = test_df.select(feature_columns).to_numpy()
    y_test = test_df.select("threat_label_encoded").to_numpy().ravel()

    # Compute weights once
    class_weight = compute_class_weights(y_train)

    experiments = {
        "logistic_regression": {
            "class_weight": class_weight,
            "random_state": RANDOM_STATE,
        },
        "random_forest": {
            "n_estimators": 200, # Reduced slightly for memory stability
            "class_weight": class_weight,
            "random_state": RANDOM_STATE,
        },
        "gradient_boosting": {
            "n_estimators": 400,
            "class_weight": class_weight,
            "device": "gpu", # Ensure this matches the logic in model.py
            "random_state": RANDOM_STATE,
        },
    }

    results = {}
    for model_name, config in experiments.items():
        print(f"\nTraining model: {model_name}")
        model = train_model(model_name, config, X_train, y_train)
        metrics = evaluate_classification(model, X_val, y_val, model_name)
        results[model_name] = {"model": model, "metrics": metrics}

    summary = summarize_results(results)
    best_model_name = summary["best_model"]
    best_model = results[best_model_name]["model"]

    print(f"\nSelected best model: {best_model_name}")
    final_metrics = evaluate_classification(best_model, X_test, y_test, f"{best_model_name}_test")

    with open(MODEL_OUTPUT_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_OUTPUT_DIR / "final_metrics.pkl", "wb") as f:
        pickle.dump(final_metrics, f)

    plot_confusion(y_val, best_model.predict(X_val), "val")
    plot_precision_recall(best_model, X_val, y_val, "val")
    plot_roc(best_model, X_val, y_val, "val")

    # Test diagnostics
    plot_confusion(y_test, best_model.predict(X_test), "test")
    plot_precision_recall(best_model, X_test, y_test, "test")
    plot_roc(best_model, X_test, y_test, "test")

    plot_feature_importance(best_model, feature_columns)

    compare_splits(
    train_metrics=results[best_model_name]["metrics"],
    val_metrics=results[best_model_name]["metrics"],
    test_metrics=final_metrics,
    )

    print("\n🔍 Running XGBoost hyperparameter tuning...")
    best_xgb, best_params = tune_xgboost(
        X_train,
        y_train,
        base_config={
            "device": "gpu",
            "random_state": RANDOM_STATE,
        },
    )

    print("Best XGBoost params:", best_params)

    # Evaluate tuned model
    tuned_val_metrics = evaluate_classification(
        best_xgb, X_val, y_val, "xgboost_tuned"
    )

    tuned_test_metrics = evaluate_classification(
        best_xgb, X_test, y_test, "xgboost_tuned_test"
    )

    # Compare against previous best
    results["xgboost_tuned"] = {
        "model": best_xgb,
        "metrics": tuned_val_metrics,
    }

    print("Best XGBoost params:", best_params)

if __name__ == "__main__":
    main()