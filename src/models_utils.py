from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from typing import Dict, Any, Iterable, Optional, List



def aggregate_time_series(
    df: pd.DataFrame,
    date_col: str = "EventDate",
    target_cols: list[str] = ["Hospitalized", "Amputation"],
    freq: str = "MS",     # "MS" monthly, "W-MON" weekly, "Q" quarterly, etc.
    agg: str = "sum",
) -> pd.DataFrame:
    """
    Aggregate incident-level data into a time series of arbitrary frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Original event-level dataframe.
    date_col : str
        Name of the timestamp column.
    target_cols : list of str
        Columns to aggregate (e.g. ["Hospitalized", "Amputation"]).
    freq : str
        Target resampling frequency. Examples:
            "MS"     → Month Start
            "M"      → Month End
            "W-MON"  → Weekly (Mondays)
            "W"      → Weekly (Sundays)
            "Q"      → Quarterly
            "A"      → Annual
    agg : str
        Aggregation function applied to the period ("sum", "mean", "max", ...).

    Returns
    -------
    ts_df : pd.DataFrame
        Aggregated time series indexed with the desired frequency.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # 1) Aggregate to daily first (handles multiple events per day)
    daily = (
        df.groupby(date_col)[target_cols]
          .agg(agg)
          .sort_index()
    )

    # 2) Resample to chosen frequency
    ts_df = daily.resample(freq).agg(agg)

    # 3) Ensure frequency is explicitly set
    ts_df = ts_df.asfreq(freq)

    return ts_df

# -----------------------
# Metrics
# -----------------------

def mae_rmse(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Compute MAE and RMSE between two aligned series.
    """
    mask = ~y_true.isna() & ~y_pred.isna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return {"MAE": mae, "RMSE": rmse}



def build_calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build calendar/seasonal features from a DatetimeIndex.
    """
    cal = pd.DataFrame(index=idx)
    cal["year"] = idx.year
    cal["month"] = idx.month
    cal["quarter"] = idx.quarter

    # cyclical encoding for month
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12)

    return cal


def build_feature_matrix(
    y: pd.Series,
    *,
    # AR features
    lags: Iterable[int] = (1, 2, 3, 6, 12),
    rolling_windows: Iterable[int] = (3, 6, 12),
    ewma_spans: Iterable[int] = (3, 6, 12),
    add_calendar: bool = True,
    # Exogenous features
    exog_df: Optional[pd.DataFrame] = None,
    exog_cols: Optional[Iterable[str]] = None,
    exog_lag_config: Optional[Dict[str, Iterable[int]]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a flexible feature matrix for ML models from a univariate target series.

    Features:
    - Calendar / seasonal features (optional)
    - Lagged values of y
    - Rolling means of y (using only past data)
    - EWMAs of y (using only past data)
    - Selected exogenous features (exog_cols) from exog_df
    - Optional lagged exogenous features as specified in exog_lag_config

    Parameters
    ----------
    y : pd.Series
        Target time series with DatetimeIndex and fixed monthly frequency.
    lags : iterable of int
        Lags of y to include (e.g. [1,2,3,6,12]).
    rolling_windows : iterable of int
        Window sizes for rolling means of y.
    ewma_spans : iterable of int
        Spans for EWMA of y.
    add_calendar : bool
        If True, add calendar / seasonal features.
    exog_df : pd.DataFrame, optional
        DataFrame with exogenous features, indexed by the same DatetimeIndex as y.
        This can contain things like NAICS counts, encoded states, event types, etc.
    exog_cols : iterable of str, optional
        Names of columns in exog_df to include as contemporaneous features.
        (We assume these have been constructed in a leakage-safe way.)
    exog_lag_config : dict, optional
        Mapping from exogenous column name -> iterable of lags to create.
        Example: {"naics_top5_share": [1, 12], "state_te": [1, 2, 3]}

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y_supervised : pd.Series
        Target aligned with X after dropping rows with NaNs.
    """
    y = y.sort_index()
    df = pd.DataFrame({"y": y})

    # -----------------------
    # AR lag features (y)
    # -----------------------
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # -----------------------
    # Rolling means (no leakage: use shifted y)
    # -----------------------
    y_shifted = df["y"].shift(1)
    for w in rolling_windows:
        df[f"rollmean_{w}"] = y_shifted.rolling(window=w, min_periods=w).mean()

    # -----------------------
    # EWMA features (no leakage: use shifted y)
    # -----------------------
    for span in ewma_spans:
        df[f"ewma_{span}"] = y_shifted.ewm(span=span, adjust=False).mean()

    # -----------------------
    # Calendar / seasonal features
    # -----------------------
    if add_calendar:
        cal = build_calendar_features(df.index)
        df = df.join(cal)

    # -----------------------
    # Exogenous features (user-controlled)
    # -----------------------
    if exog_df is not None:
        # Make sure exog index aligns with y
        exog_df = exog_df.sort_index()
        exog_df = exog_df.reindex(df.index)

        # 1) contemporaneous exog cols
        if exog_cols is not None:
            exog_cols = list(exog_cols)
            df = df.join(exog_df[exog_cols])

        # 2) lagged exog features
        if exog_lag_config is not None:
            for col, lag_list in exog_lag_config.items():
                if col not in exog_df.columns:
                    raise ValueError(f"Column '{col}' not in exog_df.")
                for lag in lag_list:
                    df[f"{col}_lag_{lag}"] = exog_df[col].shift(lag)

    # -----------------------
    # Drop rows with missing values after all transformations
    # -----------------------
    df_clean = df.dropna()

    y_supervised = df_clean["y"].copy()
    X = df_clean.drop(columns=["y"])

    return X, y_supervised

def fit_evaluate_linear_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit and evaluate linear models: Ridge, Lasso, ElasticNet, PLSRegression.
    Uses a config dictionary to eliminate redundancy.

    Returns
    -------
    results : dict
        Keys are model names; values contain predictions, metrics, and estimators.
    """

    # Optional safety checks
    assert len(X_train) == len(y_train), "X_train and y_train must be aligned."
    assert len(X_test) == len(y_test), "X_test and y_test must be aligned."

    # Convert y to numpy for sklearn
    y_train_arr = y_train.to_numpy()

    results: Dict[str, Dict[str, Any]] = {}

    # ----------------------------------------------------------
    # 1. CONFIGURATION DICT
    # ----------------------------------------------------------
    model_configs = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
    }

    # ----------------------------------------------------------
    # 2. LOOP THROUGH RIDGE / LASSO / ENET
    # ----------------------------------------------------------
    for name, estimator in model_configs.items():

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train_arr)

        # Predictions
        train_pred = pd.Series(pipe.predict(X_train), index=y_train.index, name=f"{name}_train_pred")
        test_pred  = pd.Series(pipe.predict(X_test),  index=y_test.index,  name=f"{name}_test_pred")

        # Store results
        results[name] = {
            "model_name": name,
            "estimator": pipe,
            "train_true": y_train,
            "train_pred": train_pred,
            "test_true": y_test,
            "test_pred": test_pred,
            "train_metrics": mae_rmse(y_train, train_pred),
            "test_metrics": mae_rmse(y_test, test_pred),
        }

    # ----------------------------------------------------------
    # 3. SPECIAL CASE: PLSRegression
    # ----------------------------------------------------------
    n_components = max(1, min(10, X_train.shape[1], X_train.shape[0] - 1))

    pls_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", PLSRegression(n_components=n_components)),
    ])
    pls_pipe.fit(X_train, y_train_arr)

    pls_train_pred = pd.Series(pls_pipe.predict(X_train).ravel(), index=y_train.index, name="PLS_train_pred")
    pls_test_pred  = pd.Series(pls_pipe.predict(X_test).ravel(),  index=y_test.index,  name="PLS_test_pred")

    results["PLS"] = {
        "model_name": f"PLS(n_components={n_components})",
        "estimator": pls_pipe,
        "train_true": y_train,
        "train_pred": pls_train_pred,
        "test_true": y_test,
        "test_pred": pls_test_pred,
        "train_metrics": mae_rmse(y_train, pls_train_pred),
        "test_metrics": mae_rmse(y_test, pls_test_pred),
    }

    return results


def fit_evaluate_boosting_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit and evaluate boosting-based tree models:
    - XGBoost
    - LightGBM
    - CatBoost

    Uses simple, stable defaults. Returns results dicts that match the structure
    used for classical and linear models.

    results[model_key] = {
        "model_name": ...,
        "estimator": fitted_model,
        "train_true": y_train,
        "train_pred": train_pred,
        "test_true": y_test,
        "test_pred": test_pred,
        "train_metrics": {...},
        "test_metrics": {...},
        "train_residuals": y_train - train_pred,
        "test_residuals": y_test - test_pred,
    }
    """

    # Basic alignment checks
    assert len(X_train) == len(y_train), "X_train and y_train must have the same length."
    assert len(X_test) == len(y_test), "X_test and y_test must have the same length."

    results: Dict[str, Dict[str, Any]] = {}

    y_train_arr = y_train.to_numpy()

    n_estimators = min(300, max(50, 5 * len(X_train)))
    # -----------------------
    # Config dict + unified loop
    # -----------------------
    model_configs = {
        "XGBoost": {
            "cls": XGBRegressor,
            "init": dict(
                n_estimators=n_estimators,
                learning_rate=0.05,        # could go to 0.1 if you reduce n_estimators
                max_depth=3,               # shallow trees
                subsample=1.0,             # small dataset → use all rows
                colsample_bytree=1.0,      # few AR features → use all
                reg_lambda=1.0,            # mild L2
                objective="reg:squarederror",
                random_state=0,
                n_jobs=-1,
            ),
        },
        "LightGBM": {
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
                random_state=0,
                n_jobs=-1,
                verbose=-1
            ),
        },
        "CatBoost": {
            "cls": CatBoostRegressor,
            "init": dict(
                iterations=n_estimators,
                learning_rate=0.05,
                depth=3,                   # shallow
                l2_leaf_reg=3.0,           # mild regularization
                loss_function="RMSE",
                random_state=0,
                verbose=False,
            ),
        },
    }

    for name, cfg in model_configs.items():
        cls = cfg["cls"]
        if cls is None:
            # Library not installed / available
            continue

        model = cls(**cfg["init"])
        model.fit(X_train, y_train_arr)

        train_pred = pd.Series(model.predict(X_train), index=y_train.index, name=f"{name}_train_pred")
        test_pred  = pd.Series(model.predict(X_test),  index=y_test.index,  name=f"{name}_test_pred")

        train_metrics = mae_rmse(y_train, train_pred)
        test_metrics  = mae_rmse(y_test, test_pred)

        results[name] = {
            "model_name": name,
            "estimator": model,
            "train_true": y_train,
            "train_pred": train_pred,
            "test_true": y_test,
            "test_pred": test_pred,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            # Extra goodies for error analysis:
            "train_residuals": y_train - train_pred,
            "test_residuals": y_test - test_pred,
        }

    return results

# Feature importance


def extract_feature_importance(
    estimator: Any,
    feature_names: List[str],
    model_name: str,
    use_abs: bool = True,
) -> pd.Series:
    """
    Robust feature importance extractor handling Pipelines and different model types.
    """
    # 1. Unpack Pipeline if necessary
    # Instead of looking for a step named "model", we look for the last step
    if hasattr(estimator, "steps"):
        base_model = estimator.steps[-1][1]
    else:
        base_model = estimator

    importances = None

    # 2. Linear models (coef_)
    if hasattr(base_model, "coef_"):
        # Handle multi-class (coef_ is shape [n_classes, n_features])
        if base_model.coef_.ndim > 1:
            # Common strategy: take the mean absolute importance across classes
            coefs = np.mean(np.abs(np.asarray(base_model.coef_)), axis=0)
        else:
            coefs = np.asarray(base_model.coef_).ravel()
            if use_abs:
                coefs = np.abs(coefs)
        importances = coefs

    # 3. Tree/boosting models (feature_importances_)
    elif hasattr(base_model, "feature_importances_"):
        importances = np.asarray(base_model.feature_importances_)

    # 4. CatBoost-style
    elif hasattr(base_model, "get_feature_importance"):
        # Note: CatBoost get_feature_importance can require pool data for SHAP
        # This works for the default 'PredictionValuesChange'
        importances = np.asarray(base_model.get_feature_importance())

    else:
        raise ValueError(
            f"Estimator for {model_name} ({type(base_model).__name__}) "
            "does not expose coef_ or feature_importances_."
        )

    # 5. Validation
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Shape Mismatch for {model_name}: Model has {len(importances)} features, "
            f"but {len(feature_names)} feature names were provided. "
            "Did you use a Preprocessor (OneHot/Poly) inside the pipeline?"
        )

    s = pd.Series(importances, index=feature_names, name=model_name)
    s = s.sort_values(ascending=False)
    return s

def get_feature_importances(
    results: dict,
    X_train: pd.DataFrame,
    use_abs: bool = True,
) -> dict[str, pd.Series]:
    
    importances_dict: dict[str, pd.Series] = {}

    for key, res in results.items():
        est = res["estimator"]
        model_name = res["model_name"]
        
        # --- LOGIC TO HANDLE TRANSFORMED FEATURE NAMES ---
        current_feature_names = list(X_train.columns)
        
        # If it is a pipeline, we must check if features were added/removed
        if hasattr(est, "steps"):
            # We treat everything BEFORE the final model as the "preprocessor"
            preprocessor = est[:-1] 
            
            # Try to get output names from preprocessor
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    # Note: This requires the preprocessor to be fitted
                    current_feature_names = list(preprocessor.get_feature_names_out())
                except Exception as e:
                    print(f"Could not extract feature names from pipeline for {model_name}: {e}")
        # --------------------------------------------------

        try:
            importance_series = extract_feature_importance(
                estimator=est,
                feature_names=current_feature_names,
                model_name=model_name,
                use_abs=use_abs,
            )
            importances_dict[key] = importance_series
            
        except ValueError as e:
            # print(f"Skipping {model_name}: {e}") # Optional: easy debugging
            continue

    return importances_dict
