from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from sklearn.base import RegressorMixin
from sklearn.base import clone

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ts_analysis_utils import prepare_series

@dataclass
class LocalETSResult:
    models: Dict[str, Any]          # state -> fitted ETS model
    forecasts: pd.DataFrame         # long df: one row per state-date-split
    metrics_by_state: pd.DataFrame  # rmse/mae per state
    metrics_overall: pd.Series      # global rmse/mae across all states


def run_local_ets_experiment(
    df: pd.DataFrame,
    state_col: str = "State",
    date_col: str = "Date",
    target_col: str = "Hospitalized",
    freq: str = "M",
    agg: str = "sum",
    n_test: int = 12,
    min_train_points: int = 24,
    ets_kwargs: Optional[dict] = None,
) -> LocalETSResult:
    """
    Fit one ETS model per state on the target time series.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with at least [state_col, date_col, target_col].
    state_col : str
        Column indicating the state identifier.
    date_col : str
        Column with the timestamp (will be parsed inside prepare_series).
    target_col : str
        Target variable, e.g. 'Hospitalized'.
    freq : str
        Frequency for the time series, e.g. 'M' for monthly, 'W' for weekly.
    agg : str
        Aggregation function passed to prepare_series ('sum', 'mean', ...).
    n_test : int
        Number of last periods to reserve as test set (same across all states).
    min_train_points : int
        Minimum number of observations required in the training set for a state
        to fit ETS; states with fewer points are skipped.
    ets_kwargs : dict, optional
        Extra keyword args for ExponentialSmoothing, e.g.
        dict(trend="add", seasonal=None, initialization_method="estimated").

    Returns
    -------
    LocalETSResult
        An object containing:
        - models: dict[state] -> fitted ETS model
        - forecasts: long df with columns:
              [state_col, "Date", "split", "y_true", "y_pred"]
        - metrics_by_state: rmse/mae per state
        - metrics_overall: aggregated rmse/mae across all states+dates
    """
    if ets_kwargs is None:
        # Reasonable default: additive trend, no seasonality
        ets_kwargs = dict(trend="add",
                          seasonal=None,
                          initialization_method="estimated")

    all_rows = []           # will accumulate prediction rows
    models: Dict[str, Any] = {}
    metrics_records = []

    for state, df_state in df.groupby(state_col):
        # 1. Build the univariate series for this state
        y = prepare_series(
            df_state,
            date_col=date_col,
            value_col=target_col,
            freq=Optional[str](freq),  # or just freq if your prepare_series expects that
            agg=agg,
        ).sort_index()

        if len(y) <= n_test + min_train_points:
            # Not enough data, skip this state
            continue

        # 2. Train/test split by last n_test points
        train = y.iloc[:-n_test]
        test = y.iloc[-n_test:]

        # 3. Fit ETS on train
        model = ExponentialSmoothing(train, **ets_kwargs).fit(optimized=True)
        models[state] = model

        # 4. In-sample fitted values (train predictions)
        fitted_train = model.fittedvalues.reindex(train.index)

        # 5. Out-of-sample forecast for test period
        forecast_test = model.forecast(len(test))
        forecast_test.index = test.index  # ensure index aligns

        # 6. Collect rows for train and test
        # Train rows
        all_rows.append(
            pd.DataFrame(
                {
                    state_col: state,
                    "Date": train.index,
                    "split": "train",
                    "y_true": train.values,
                    "y_pred": fitted_train.values,
                }
            )
        )
        # Test rows
        all_rows.append(
            pd.DataFrame(
                {
                    state_col: state,
                    "Date": test.index,
                    "split": "test",
                    "y_true": test.values,
                    "y_pred": forecast_test.values,
                }
            )
        )

        # 7. Per-state metrics on test
        rmse = np.sqrt(((forecast_test - test) ** 2).mean())
        mae = (forecast_test - test).abs().mean()

        metrics_records.append(
            {state_col: state, "rmse": rmse, "mae": mae}
        )

    if not all_rows:
        raise ValueError("No state had enough data to fit a local ETS model.")

    forecasts = pd.concat(all_rows, ignore_index=True)
    forecasts.set_index([state_col, "Date"], inplace=True)
    forecasts.sort_index(inplace=True)

    metrics_by_state = pd.DataFrame(metrics_records).set_index(state_col).sort_index()

    # Overall metrics across all states, test only
    mask_test = forecasts["split"] == "test"
    y_true_all = forecasts.loc[mask_test, "y_true"]
    y_pred_all = forecasts.loc[mask_test, "y_pred"]

    rmse_overall = np.sqrt(((y_pred_all - y_true_all) ** 2).mean())
    mae_overall = (y_pred_all - y_true_all).abs().mean()

    metrics_overall = pd.Series(
        {"rmse": rmse_overall, "mae": mae_overall}, name="overall"
    )

    return LocalETSResult(
        models=models,
        forecasts=forecasts,
        metrics_by_state=metrics_by_state,
        metrics_overall=metrics_overall,
    )



def fit_local_models_for_state(
    models: Dict[str, RegressorMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, RegressorMixin]:
    """
    Fit one fresh clone of each model for a single state's training set.
    """
    fitted: Dict[str, RegressorMixin] = {}
    for name, model in models.items():
        m = clone(model)  # IMPORTANT: avoid sharing state across states
        m.fit(X_train, y_train)
        fitted[name] = m
    return fitted


def train_and_predict_local_models(
    models: Dict[str, RegressorMixin],
    splits_by_state: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    state_col: str = "State",
) -> Tuple[
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[Tuple[str, str], RegressorMixin],
]:
    """
    Local analogue of train_and_predict_global_models.

    Fits one model per state (per model type), predicts for TRAIN and TEST,
    and returns:

    1) results: dict in the SAME SHAPE as the global function, but with
       y_true/y_pred/meta concatenated across states for each model.

    2) fitted_models: dict keyed by (model_name, state) -> fitted estimator,
       so per-state forecasts remain accessible outside this function.
    """
    # Accumulators per model and split
    acc: Dict[str, Dict[str, Dict[str, list]]] = {}
    for name in models.keys():
        acc[name] = {
            "train": {"y_true": [], "y_pred": [], "meta": []},
            "test": {"y_true": [], "y_pred": [], "meta": []},
        }

    fitted_models: Dict[Tuple[str, str], RegressorMixin] = {}

    # Loop over states
    for state, splits in splits_by_state.items():
        X_train = splits["train"]["X"]
        y_train = splits["train"]["y"]
        meta_train = splits["train"]["meta"]

        X_test = splits["test"]["X"]
        y_test = splits["test"]["y"]
        meta_test = splits["test"]["meta"]

        # Fit all model types for this state
        fitted_state_models = fit_local_models_for_state(
            models=models,
            X_train=X_train,
            y_train=y_train,
        )

        # Predict and store
        for name, model in fitted_state_models.items():
            fitted_models[(name, state)] = model

            y_pred_train = pd.Series(
                model.predict(X_train),
                index=y_train.index,
                name=f"{name}_pred_train",
            )

            y_pred_test = pd.Series(
                model.predict(X_test),
                index=y_test.index,
                name=f"{name}_pred_test",
            )

            # Ensure meta contains the state (it should, but guard anyway)
            if state_col not in meta_train.columns:
                meta_train = meta_train.copy()
                meta_train[state_col] = state
            if state_col not in meta_test.columns:
                meta_test = meta_test.copy()
                meta_test[state_col] = state

            acc[name]["train"]["y_true"].append(y_train.reset_index(drop=True))
            acc[name]["train"]["y_pred"].append(y_pred_train.reset_index(drop=True))
            acc[name]["train"]["meta"].append(meta_train.reset_index(drop=True))

            acc[name]["test"]["y_true"].append(y_test.reset_index(drop=True))
            acc[name]["test"]["y_pred"].append(y_pred_test.reset_index(drop=True))
            acc[name]["test"]["meta"].append(meta_test.reset_index(drop=True))

    # Concatenate across states to match global output format
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in models.keys():
        results[name] = {}
        for split in ["train", "test"]:
            y_true_all = pd.concat(acc[name][split]["y_true"], axis=0, ignore_index=True)
            y_pred_all = pd.concat(acc[name][split]["y_pred"], axis=0, ignore_index=True)
            meta_all = pd.concat(acc[name][split]["meta"], axis=0, ignore_index=True)

            results[name][split] = {
                "y_true": y_true_all,
                "y_pred": y_pred_all,
                "meta": meta_all,
            }

    return results, fitted_models

