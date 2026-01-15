from typing import Dict, Any
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def build_logistic_regression(config: Dict[str, Any]):
    """
    Logistic Regression using SGDClassifier with 'modified_huber' loss.
    """
    return SGDClassifier(
        loss="log_loss", 
        penalty=config.get("penalty", "l2"),
        alpha=0.0001, 
        max_iter=2000,
        random_state=config.get("random_state", 42),
        class_weight=config.get("class_weight", None),
        n_jobs=13 
    )

def build_random_forest(config: Dict[str, Any]):
    """
    Docstring for build_random_forest
    
    :param config: Description
    :type config: Dict[str, Any]
    """
    return RandomForestClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=None,
        class_weight=config.get("class_weight", None),
        random_state=config.get("random_state", 42),
        n_jobs=13 
    )

def build_gradient_boosting(config: Dict[str, Any]):
    """
    XGBoost 2.0+ GPU configuration.
    Uses tree_method='hist' with device='cuda'.
    """
    is_gpu = config.get("device") == "gpu"
    
    return xgb.XGBClassifier(
        n_estimators=config.get("n_estimators", 400),
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        device="cuda" if is_gpu else "cpu",
        n_jobs=13,
        random_state=config.get("random_state", 42)
    )

def get_model(model_name: str, config: Dict[str, Any]):
    """
    Docstring for get_model
    
    :param model_name: Description
    :type model_name: str
    :param config: Description
    :type config: Dict[str, Any]
    """
    if model_name == "logistic_regression":
        return build_logistic_regression(config)
    elif model_name == "random_forest":
        return build_random_forest(config)
    elif model_name == "gradient_boosting":
        return build_gradient_boosting(config)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")