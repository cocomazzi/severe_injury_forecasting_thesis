# training.py

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd
from sklearn.base import RegressorMixin

from .splitting import temporal_panel_split
from .models import get_model_configs, instantiate_models


def fit_global_models(
    models: Dict[str, RegressorMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, RegressorMixin]:
    """
    Fit all models on the global training set.

    Parameters
    ----------
    models : dict
        {model_name: estimator_instance}
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    fitted_models : dict
        Same dict, but after calling .fit() on each estimator.
    """
    fitted_models: Dict[str, RegressorMixin] = {}

    for name, model in models.items():
        m = model
        m.fit(X_train, y_train)
        fitted_models[name] = m

    return fitted_models


def train_and_predict_global_models(
    models: Dict[str, RegressorMixin],
    splits: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Fit models on the TRAIN split and generate predictions for TRAIN and TEST.

    Parameters
    ----------
    models : dict
        {model_name: estimator_instance}
    splits : dict
        Output of temporal_panel_split, with keys:
          - "train": {"X": X_train, "y": y_train, "meta": meta_train}
          - "test":  {"X": X_test,  "y": y_test,  "meta": meta_test}

    Returns
    -------
    results : dict
        Nested dict:

        {
          model_name: {
            "train": {
               "y_true": pd.Series,
               "y_pred": pd.Series,
               "meta":  pd.DataFrame,
            },
            "test": {
               "y_true": pd.Series,
               "y_pred": pd.Series,
               "meta":  pd.DataFrame,
            },
          },
          ...
        }

        This structure is designed to make error analysis and
        per-state metrics easy later (using meta["State"], meta["Date"]).
    """
    # Unpack splits
    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    meta_train = splits["train"]["meta"]

    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]
    meta_test = splits["test"]["meta"]

    # 1) Fit models on train
    fitted_models = fit_global_models(models=models, X_train=X_train, y_train=y_train)

    # 2) Predict on train and test for each model
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for name, model in fitted_models.items():
        # Train predictions
        y_pred_train = pd.Series(
            model.predict(X_train),
            index=y_train.index,
            name=f"{name}_pred_train",
        )

        # Test predictions
        y_pred_test = pd.Series(
            model.predict(X_test),
            index=y_test.index,
            name=f"{name}_pred_test",
        )

        results[name] = {
            "train": {
                "y_true": y_train.reset_index(drop=True),
                "y_pred": y_pred_train.reset_index(drop=True),
                "meta": meta_train.reset_index(drop=True),
            },
            "test": {
                "y_true": y_test.reset_index(drop=True),
                "y_pred": y_pred_test.reset_index(drop=True),
                "meta": meta_test.reset_index(drop=True),
            },
        }

    return results


from .training import train_and_predict_global_models  # or local import to avoid cycles


def multi_step_global_models(
    horizons: Sequence[int],
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    date_col: str,
    train_end: str,
    test_start: str | None = None,
    test_size: int = 12,
    panel_col: str = "State",
    use_linear: bool = True,
    use_tree: bool = True,
    random_state: int = 0,
    n_jobs: int = -1,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    TRUE DIRECT MULTI-STEP STRATEGY (panel-safe).

    For each horizon h:
      - build y_h(t) = y(t+h) by shifting WITHIN each panel (State)
      - drop rows where y_h is missing (last h per State)
      - split by global calendar time using temporal_panel_split
      - train a separate model set per horizon
    """
    if panel_col not in meta.columns:
        raise KeyError(f"panel_col '{panel_col}' not found in meta DataFrame.")
    if date_col not in meta.columns:
        raise KeyError(f"date_col '{date_col}' not found in meta DataFrame.")
    if not (len(X) == len(y) == len(meta)):
        raise ValueError("X, y, and meta must have the same number of rows.")
    if not (X.index.equals(y.index) and X.index.equals(meta.index)):
        raise ValueError("Indices of X, y, and meta do not align.")

    # Ensure datetime and stable ordering (panel, then time)
    meta_sorted = meta.copy()
    meta_sorted[date_col] = pd.to_datetime(meta_sorted[date_col], errors="raise")

    order = meta_sorted.sort_values([panel_col, date_col]).index
    Xo = X.loc[order].reset_index(drop=True)
    yo = y.loc[order].reset_index(drop=True)
    mo = meta_sorted.loc[order].reset_index(drop=True)

    results_all: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for h in horizons:
        if h <= 0:
            raise ValueError(f"Horizon must be positive, got {h}.")

        # Panel-safe shift: y_h(t) = y(t+h) within each State
        y_h = yo.groupby(mo[panel_col], sort=False).shift(-h)

        valid_mask = y_h.notna()
        X_h = Xo.loc[valid_mask].reset_index(drop=True)
        y_h_valid = y_h.loc[valid_mask].reset_index(drop=True)
        meta_h = mo.loc[valid_mask].reset_index(drop=True)

        # Temporal split on origins (global calendar cut)
        splits_h = temporal_panel_split(
            X=X_h,
            y=y_h_valid,
            meta=meta_h,
            date_col=date_col,
            train_end=train_end,
            test_size=test_size,
            test_start=test_start,
        )

        X_train_h = splits_h["train"]["X"]
        if len(X_train_h) == 0:
            raise ValueError(
                f"No training samples available for horizon h={h}. "
                "Check train_end and data length."
            )

        model_configs_h = get_model_configs(
            n_samples=len(X_train_h),
            use_linear=use_linear,
            use_tree=use_tree,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        models_h = instantiate_models(model_configs_h)

        results_h = train_and_predict_global_models(
            models=models_h,
            splits=splits_h,
        )

        for base_name, res in results_h.items():
            results_all[f"{base_name}_h{h}"] = res

    return results_all
