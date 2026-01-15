"""
tuning.py

Hyperparameter optimization for cybersecurity threat models.

Uses RANDOMIZED search with SECURITY-FIRST scoring.
"""

from typing import Dict, Any
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score

from src.model import build_gradient_boosting

# Custom Security Scorer

def malicious_recall_scorer(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)


SECURITY_SCORER = make_scorer(malicious_recall_scorer)

# Randomized Search for XGBoost

def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    base_config: Dict[str, Any],
    n_iter: int = 25,
):
    """
    Perform security-aware hyperparameter tuning for XGBoost.

    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels
    base_config : dict
        Base XGBoost configuration
    n_iter : int
        Number of random search iterations

    Returns
    -------
    best_model
    best_params
    """

    model = build_gradient_boosting(base_config)

    param_dist = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "gamma": [0, 1, 5],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=SECURITY_SCORER,
        cv=cv,
        verbose=2,
        n_jobs=1,  
        refit=True,
    )

    search.fit(X, y)

    return search.best_estimator_, search.best_params_
