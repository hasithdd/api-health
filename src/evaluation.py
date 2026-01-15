"""
evaluation.py

Evaluation utilities for cybersecurity threat classification models.

Responsibilities
----------------
- Compute security-focused classification metrics
- Emphasize minority-class (suspicious, malicious) performance
- Provide model comparison and ranking utilities

This module MUST NOT:
- Train models
- Load raw data
- Perform feature engineering
"""

from typing import Dict, Any

import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    recall_score,
)

# Core Evaluation

def evaluate_classification(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a trained model using security-relevant metrics.

    Metrics Emphasized
    ------------------
    - Recall (malicious)  ← highest priority
    - Recall (suspicious)
    - Macro F1
    - Confusion matrix

    Parameters
    ----------
    model : trained model
        Must implement predict() and optionally predict_proba()
    X_val : np.ndarray
        Validation or test features
    y_val : np.ndarray
        True labels
    model_name : str
        Identifier for logging and reporting

    Returns
    -------
    metrics : dict
        Dictionary containing detailed evaluation metrics
    """

    # Predictions
    y_pred = model.predict(X_val)

    # Core metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val,
        y_pred,
        labels=[0, 1, 2],  # benign, suspicious, malicious
        zero_division=0,
    )

    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    malicious_recall = recall[2]
    suspicious_recall = recall[1]

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])

    # Structured report
    metrics = {
        "model_name": model_name,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "malicious_recall": malicious_recall,
        "suspicious_recall": suspicious_recall,
        "per_class": {
            "benign": {
                "precision": precision[0],
                "recall": recall[0],
                "f1": f1[0],
                "support": support[0],
            },
            "suspicious": {
                "precision": precision[1],
                "recall": recall[1],
                "f1": f1[1],
                "support": support[1],
            },
            "malicious": {
                "precision": precision[2],
                "recall": recall[2],
                "f1": f1[2],
                "support": support[2],
            },
        },
        "confusion_matrix": cm,
    }

    print_evaluation(metrics)
    return metrics


# Reporting Utilities

def print_evaluation(metrics: Dict[str, Any]):
    """
    Print a concise, security-focused evaluation summary.
    """
    print("\n================ Evaluation =================")
    print(f"Model: {metrics['model_name']}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"Malicious Recall  : {metrics['malicious_recall']:.4f}")
    print(f"Suspicious Recall : {metrics['suspicious_recall']:.4f}")
    print("============================================")


# Model Comparison & Selection

def summarize_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Rank models and select the best one based on security priorities.

    Ranking Strategy (in order)
    ---------------------------
    1. Highest malicious recall
    2. Highest suspicious recall
    3. Highest macro F1

    Parameters
    ----------
    results : dict
        Output from training loop in train.py

    Returns
    -------
    summary : dict
        Ranked models and selected best model
    """

    rankings = []

    for model_name, payload in results.items():
        metrics = payload["metrics"]
        rankings.append({
            "model": model_name,
            "malicious_recall": metrics["malicious_recall"],
            "suspicious_recall": metrics["suspicious_recall"],
            "macro_f1": metrics["macro_f1"],
        })

    # Sort by security priority
    rankings.sort(
        key=lambda x: (
            x["malicious_recall"],
            x["suspicious_recall"],
            x["macro_f1"],
        ),
        reverse=True,
    )

    best_model = rankings[0]["model"]

    print("\n=========== Model Ranking ===========")
    for rank, entry in enumerate(rankings, start=1):
        print(
            f"{rank}. {entry['model']} | "
            f"MalRec={entry['malicious_recall']:.4f} | "
            f"SusRec={entry['suspicious_recall']:.4f} | "
            f"MacroF1={entry['macro_f1']:.4f}"
        )
    print("====================================")

    return {
        "rankings": rankings,
        "best_model": best_model,
    }
