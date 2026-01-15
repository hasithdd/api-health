"""
validation.py

Model validation, diagnostics, and visualization utilities.

Responsibilities
----------------
- Detect overfitting
- Visualize class confusion
- Analyze probability behavior
- Inspect feature importance
- Generate artifacts for reporting

This module MUST NOT:
- Train models
- Modify model weights
- Perform feature engineering
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)

OUTPUT_DIR = Path("validation")
OUTPUT_DIR.mkdir(exist_ok=True)

# Utility

def _save_plot(name: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=150)
    plt.close()

# Confusion Matrix

def plot_confusion(y_true, y_pred, split: str):
    labels = ["benign", "suspicious", "malicious"]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({split})")

    _save_plot(f"confusion_matrix_{split}")
# Precision–Recall Curves (CRITICAL)

def plot_precision_recall(model, X, y, split: str):
    if not hasattr(model, "predict_proba"):
        return

    y_prob = model.predict_proba(X)

    plt.figure(figsize=(7, 5))

    for cls, name in zip([1, 2], ["suspicious", "malicious"]):
        precision, recall, _ = precision_recall_curve(y == cls, y_prob[:, cls])
        plt.plot(recall, precision, label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curves ({split})")
    plt.legend()

    _save_plot(f"precision_recall_{split}")

# ROC Curves (Optional but useful)

def plot_roc(model, X, y, split: str):
    if not hasattr(model, "predict_proba"):
        return

    y_prob = model.predict_proba(X)

    plt.figure(figsize=(7, 5))

    for cls, name in zip([1, 2], ["suspicious", "malicious"]):
        fpr, tpr, _ = roc_curve(y == cls, y_prob[:, cls])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({split})")
    plt.legend()

    _save_plot(f"roc_{split}")


# Feature Importance
def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]

    plt.figure(figsize=(8, 6))
    plt.barh(
        np.array(feature_names)[idx],
        importances[idx],
    )
    plt.title("Top 20 Feature Importances")

    _save_plot("feature_importance")

# Overfitting Detection
def compare_splits(
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
):
    """
    Print train vs val vs test degradation.
    """

    print("\n========== Generalization Check ==========")
    for metric in ["macro_f1", "malicious_recall", "suspicious_recall"]:
        print(
            f"{metric:20s} | "
            f"Train={train_metrics[metric]:.4f} | "
            f"Val={val_metrics[metric]:.4f} | "
            f"Test={test_metrics[metric]:.4f}"
        )
    print("==========================================")
