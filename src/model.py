"""
model.py

Model definitions for cybersecurity threat classification.

This module provides untrained, reusable model architectures
optimized for large-scale, imbalanced security datasets.

Responsibilities
----------------
- Define model architectures
- Support class imbalance handling
- Enable GPU acceleration where applicable
- Remain fully data-agnostic

This module MUST NOT:
- Load data
- Train models
- Compute metrics
- Perform feature engineering
"""

from typing import Dict, Any


from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def build_logistic_regression(config: Dict[str, Any]):
    # Using SGDClassifier as a parallel-friendly Logistic Regression alternative
    return SGDClassifier(
        loss="log_loss",  # Equivalent to Logistic Regression
        penalty=config.get("penalty", "l2"),
        alpha=1/config.get("C", 1.0), # alpha is the inverse of C
        max_iter=config.get("max_iter", 1000),
        random_state=config.get("random_state", 42),
        class_weight=config.get("class_weight", None),
        n_jobs=13  # SGDClassifier supports n_jobs for internal BLAS ops
    )

def build_random_forest(config: Dict[str, Any]):
    return RandomForestClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", None),
        class_weight=config.get("class_weight", None),
        random_state=config.get("random_state", 42),
        n_jobs=13  # Specifically targeting 13 cores
    )

def build_gradient_boosting(config: Dict[str, Any]):
    device = config.get("device", "cpu")
    tree_method = "gpu_hist" if device == "gpu" else "hist"
    
    return xgb.XGBClassifier(
        n_estimators=config.get("n_estimators", 400),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=6,
        tree_method=tree_method,
        n_jobs=13,  # XGBoost will use 13 threads
        random_state=config.get("random_state", 42)
    )

# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.ensemble import RandomForestClassifier

# import xgboost as xgb


# def build_logistic_regression(config: Dict[str, Any]):
#     """
#     Build a Logistic Regression classifier suitable for highly imbalanced
#     multi-class security data.

#     This model is primarily used as:
#     - A baseline
#     - A feature-sanity checker
#     - A highly interpretable reference model

#     Notes on Performance
#     --------------------
#     - Efficient for large datasets when features are numeric
#     - Uses optimized solvers from scikit-learn (liblinear, saga)

#     Expected Config Parameters
#     --------------------------
#     - penalty : str (default: 'l2')
#     - C : float (default: 1.0)
#     - solver : str (default: 'saga')
#     - class_weight : dict or 'balanced'
#     - max_iter : int (default: 1000)
#     - n_jobs : int (default: -1)

#     Returns
#     -------
#     model : LogisticRegression
#         Untrained logistic regression model
#     """
#     return LogisticRegression(
#         C=config.get("C", 1.0),
#         solver=config.get("solver", "saga"),
#         class_weight=config.get("class_weight", None),
#         max_iter=config.get("max_iter", 1000),
#     )




# def build_random_forest(config: Dict[str, Any]):
#     """
#     Build a Random Forest classifier optimized for tabular security logs.

#     This model:
#     - Handles non-linear feature interactions well
#     - Is robust to noisy features
#     - Serves as a strong CPU-based baseline

#     Notes on Performance
#     --------------------
#     - Parallelized via n_jobs
#     - Memory-heavy for very large datasets
#     - Not GPU-accelerated (by design)

#     Expected Config Parameters
#     --------------------------
#     - n_estimators : int (default: 200)
#     - max_depth : int or None
#     - min_samples_split : int (default: 2)
#     - min_samples_leaf : int (default: 1)
#     - class_weight : dict or 'balanced'
#     - n_jobs : int (default: -1)
#     - random_state : int (default: 42)

#     Returns
#     -------
#     model : RandomForestClassifier
#         Untrained random forest model
#     """
#     return RandomForestClassifier(
#         n_estimators=config.get("n_estimators", 200),
#         max_depth=config.get("max_depth", None),
#         min_samples_split=config.get("min_samples_split", 2),
#         min_samples_leaf=config.get("min_samples_leaf", 1),
#         class_weight=config.get("class_weight", None),
#         n_jobs=config.get("n_jobs", -1),
#         random_state=config.get("random_state", 42)
#     )


# def build_gradient_boosting(config: Dict[str, Any]):
#     """
#     Build an XGBoost multi-class classifier optimized for
#     imbalanced cybersecurity datasets.

#     This is the PRIMARY production candidate.

#     Why XGBoost?
#     -------------
#     - Excellent performance on tabular data
#     - Native handling of imbalance via scale_pos_weight
#     - GPU acceleration (hist / gpu_hist)
#     - Handles large datasets efficiently

#     Expected Config Parameters
#     --------------------------
#     - n_estimators : int (default: 300)
#     - learning_rate : float (default: 0.1)
#     - max_depth : int (default: 6)
#     - subsample : float (default: 0.8)
#     - colsample_bytree : float (default: 0.8)
#     - scale_pos_weight : dict or None
#         (Computed in train.py)
#     - device : str ('cpu' or 'gpu')

#     Returns
#     -------
#     model : xgb.XGBClassifier
#         Untrained XGBoost classifier
#     """

#     device = config.get("device", "cpu")

#     tree_method = "gpu_hist" if device == "gpu" else "hist"

#     return xgb.XGBClassifier(
#         objective="multi:softprob",
#         num_class=3,
#         n_estimators=config.get("n_estimators", 300),
#         learning_rate=config.get("learning_rate", 0.1),
#         max_depth=config.get("max_depth", 6),
#         subsample=config.get("subsample", 0.8),
#         colsample_bytree=config.get("colsample_bytree", 0.8),
#         scale_pos_weight=config.get("scale_pos_weight", None),
#         tree_method=tree_method,
#         predictor="gpu_predictor" if device == "gpu" else "auto",
#         eval_metric="mlogloss",
#         n_jobs=config.get("n_jobs", -1),
#         random_state=config.get("random_state", 42),
#         enable_categorical=False
#     )


def get_model(model_name: str, config: Dict[str, Any]):
    """
    Factory function to retrieve a model by name.

    This function centralizes model creation and prevents
    conditional logic from spreading into training code.

    Supported Models
    ----------------
    - 'logistic_regression'
    - 'random_forest'
    - 'gradient_boosting'

    Parameters
    ----------
    model_name : str
        Identifier of the model
    config : dict
        Model-specific configuration

    Returns
    -------
    model
        Untrained model instance

    Raises
    ------
    ValueError
        If model_name is unsupported
    """
    if model_name == "logistic_regression":
        return build_logistic_regression(config)

    elif model_name == "random_forest":
        return build_random_forest(config)

    elif model_name == "gradient_boosting":
        return build_gradient_boosting(config)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
