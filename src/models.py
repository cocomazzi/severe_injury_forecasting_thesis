# models.py

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_model_configs(
    n_samples: int,
    use_linear: bool = True,
    use_tree: bool = True,
    random_state: int = 0,
    n_jobs: int = -1,
) -> Dict[str, Dict[str, Any]]:
    """
    Return a registry of baseline model configurations.

    Parameters
    ----------
    n_samples : int
        Number of training samples (len(X_train)). Used to set n_estimators
        for tree-based methods via:
            n_estimators = min(300, max(50, 5 * n_samples))
    use_linear : bool, default True
        Include linear models (Ridge, Lasso, ElasticNet, PLS).
    use_tree : bool, default True
        Include tree-based boosted models (XGBoost, LightGBM, CatBoost).
    random_state : int, default 0
        Random seed for reproducibility.
    n_jobs : int, default -1
        Number of threads for models that support it.

    Returns
    -------
    configs : dict
        {model_name: {"cls": estimator_class, "init": init_kwargs}}
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    configs: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------
    # Linear models
    # -------------------------------------------------
    if use_linear:
        configs["Ridge"] = {
            "cls": Ridge,
            "init": dict(
                alpha=1.0,
                random_state=random_state,
            ),
        }

        configs["Lasso"] = {
            "cls": Lasso,
            "init": dict(
                alpha=0.001,
                max_iter=10_000,
                random_state=random_state,
            ),
        }

        configs["ElasticNet"] = {
            "cls": ElasticNet,
            "init": dict(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=10_000,
                random_state=random_state,
            ),
        }

        configs["PLS"] = {
            "cls": PLSRegression,
            "init": dict(
                n_components=10,  # can be tuned later
            ),
        }

    # -------------------------------------------------
    # Tree-based / boosted models
    # -------------------------------------------------
    if use_tree:
        # Heuristic based on dataset size
        n_estimators = min(300, max(50, 5 * n_samples))

        configs["XGBoost"] = {
            "cls": XGBRegressor,
            "init": dict(
                n_estimators=n_estimators,
                learning_rate=0.05,        # could go to 0.1 if you reduce n_estimators
                max_depth=3,               # shallow trees
                subsample=1.0,             # small dataset → use all rows
                colsample_bytree=1.0,      # few AR features → use all
                reg_lambda=1.0,            # mild L2
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        }

        configs["LightGBM"] = {
            "cls": LGBMRegressor,
            "init": dict(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=3,               # bounded depth
                num_leaves=7,              # 2^3 - 1, small tree
                subsample=1.0,
                colsample_bytree=1.0,
                objective="regression",
                reg_lambda=0.0,            # can bump to 1.0 if needed
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=-1,
            ),
        }

        configs["CatBoost"] = {
            "cls": CatBoostRegressor,
            "init": dict(
                iterations=n_estimators,
                learning_rate=0.05,
                depth=3,                   # shallow
                l2_leaf_reg=3.0,           # mild regularization
                loss_function="RMSE",
                random_state=random_state,
                verbose=False,
            ),
        }

    return configs


def instantiate_models(
    model_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Instantiate all models in the registry.

    Parameters
    ----------
    model_configs : dict
        Output of get_model_configs().

    Returns
    -------
    models : dict
        {model_name: estimator_instance}
    """
    models: Dict[str, Any] = {}
    for name, cfg in model_configs.items():
        cls = cfg["cls"]
        init_kwargs = cfg.get("init", {})
        models[name] = cls(**init_kwargs)
    return models
