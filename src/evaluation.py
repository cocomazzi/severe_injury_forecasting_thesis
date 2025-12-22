# evaluation.py

from __future__ import annotations

from typing import Any, Dict, Iterable, Literal
import numpy as np
import pandas as pd

MetricName = Literal["rmse", "mae"]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def aggregate_split_weekly_to_monthly_sum(
    split_dict: Dict[str, Any],
    date_col: str = "Date",
    group_col: str = "State",
    week_midpoint_shift_days: int = 3,  # for W-MON
) -> Dict[str, Any]:
    """
    Aggregates one split container (train or test) from weekly to monthly for SUM targets.

    Input split_dict keys: y_true, y_pred, meta
    Output structure matches your pipeline: y_true (Series), y_pred (Series), meta (DataFrame)
    with meta[date_col] labeled at month start (MS).
    """
    meta = split_dict["meta"].copy()
    dt = pd.to_datetime(meta[date_col], errors="raise")

    # Assign each week to a month using week midpoint heuristic
    dt_mid = dt - pd.to_timedelta(week_midpoint_shift_days, unit="D")
    month = dt_mid.dt.to_period("M").dt.to_timestamp("MS")  # month start labels

    df = pd.DataFrame(
        {
            group_col: meta[group_col].to_numpy(),
            "__month__": month.to_numpy(),
            "y_true": pd.Series(split_dict["y_true"]).to_numpy(dtype=float),
            "y_pred": pd.Series(split_dict["y_pred"]).to_numpy(dtype=float),
        }
    )

    agg = (
        df.groupby([group_col, "__month__"], as_index=False)
          .agg(y_true=("y_true", "sum"), y_pred=("y_pred", "sum"))
          .sort_values([group_col, "__month__"])
          .reset_index(drop=True)
    )

    out = {
        "y_true": pd.Series(agg["y_true"]),
        "y_pred": pd.Series(agg["y_pred"]),
        "meta": pd.DataFrame(
            {
                group_col: agg[group_col].to_numpy(),
                date_col: agg["__month__"].to_numpy(),
            }
        ),
    }
    return out


def aggregate_results_weekly_to_monthly_sum(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    date_col: str = "Date",
    group_col: str = "State",
    week_midpoint_shift_days: int = 3,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Applies weekly->monthly SUM aggregation to BOTH train and test splits
    for every model, returning a results dict with the same structure.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_name, model_res in results.items():
        out[model_name] = {}
        for split_name in ("train", "test"):
            out[model_name][split_name] = aggregate_split_weekly_to_monthly_sum(
                model_res[split_name],
                date_col=date_col,
                group_col=group_col,
                week_midpoint_shift_days=week_midpoint_shift_days,
            )

    return out


def add_panel_mean_to_global(
    metrics: Dict[str, Dict[str, pd.DataFrame]],
    group_col: str = "State",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Adds '<metric>_panel_mean' to global metrics by averaging
    the by_group metrics across groups (unweighted).

    This makes the global metric comparable across panels
    with different sizes.
    """
    out = {k: {s: df.copy() for s, df in v.items()} for k, v in metrics.items()}

    for split in ("train", "test"):
        by_g = out["by_group"][split].reset_index()

        if group_col not in by_g.columns:
            raise KeyError(
                f"group_col '{group_col}' not found in by_group[{split}] "
                f"(columns={list(by_g.columns)})"
            )

        if "model" not in by_g.columns:
            raise KeyError("'model' column missing from by_group metrics.")

        for col in ("rmse", "mae"):
            if col in by_g.columns:
                panel_mean = (
                    by_g.groupby("model", sort=False)[col]
                        .mean()
                        .rename(f"{col}_panel_mean")
                )
                out["global"][split] = out["global"][split].join(panel_mean, how="left")

    return out


def compute_panel_metrics(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    group_col: str = "State",
    metric_names: Iterable[MetricName] = ("rmse", "mae"),
    dropna: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Compute global and per-group metrics for each model and split.
    """
    metric_names = tuple(metric_names)
    valid_metrics = {"rmse", "mae"}
    if any(m not in valid_metrics for m in metric_names):
        raise ValueError(f"Unsupported metrics requested. Valid: {valid_metrics}")

    def _prep_arrays(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        yt = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
        if dropna:
            mask = np.isfinite(yt) & np.isfinite(yp)
            yt, yp = yt[mask], yp[mask]
        return yt, yp

    def _compute_metrics_array(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        yt, yp = _prep_arrays(y_true, y_pred)
        out: Dict[str, float] = {}
        if "rmse" in metric_names:
            out["rmse"] = _rmse(yt, yp)
        if "mae" in metric_names:
            out["mae"] = _mae(yt, yp)
        return out

    global_train_rows = []
    global_test_rows = []
    by_group_train_rows = []
    by_group_test_rows = []

    for model_name, model_res in results.items():
        for split, global_rows, by_group_rows in [
            ("train", global_train_rows, by_group_train_rows),
            ("test",  global_test_rows,  by_group_test_rows),
        ]:
            container = model_res[split]
            y_true = container["y_true"]
            y_pred = container["y_pred"]
            meta = container["meta"]

            if not (len(y_true) == len(y_pred) == len(meta)):
                raise ValueError(
                    f"Length mismatch for model='{model_name}', split='{split}': "
                    f"len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}, len(meta)={len(meta)}"
                )

            if group_col not in meta.columns:
                raise KeyError(f"group_col '{group_col}' not found in meta for model='{model_name}', split='{split}'.")

            # Global metrics
            met_vals = _compute_metrics_array(y_true, y_pred)
            global_rows.append({"model": model_name, **met_vals})

            # By-group metrics
            df_tmp = pd.DataFrame(
                {
                    "y_true": pd.to_numeric(y_true, errors="coerce"),
                    "y_pred": pd.to_numeric(y_pred, errors="coerce"),
                    group_col: meta[group_col].to_numpy(),
                }
            )

            if dropna:
                df_tmp = df_tmp.dropna(subset=["y_true", "y_pred"])

            for group, df_g in df_tmp.groupby(group_col, sort=False):
                met_vals_g = _compute_metrics_array(df_g["y_true"], df_g["y_pred"])
                by_group_rows.append({group_col: group, "model": model_name, **met_vals_g})

    global_train_df = pd.DataFrame(global_train_rows).set_index("model").sort_index()
    global_test_df = pd.DataFrame(global_test_rows).set_index("model").sort_index()

    by_group_train_df = pd.DataFrame(by_group_train_rows).set_index([group_col, "model"]).sort_index()
    by_group_test_df = pd.DataFrame(by_group_test_rows).set_index([group_col, "model"]).sort_index()

    return {
        "global": {"train": global_train_df, "test": global_test_df},
        "by_group": {"train": by_group_train_df, "test": by_group_test_df},
    }



def _infer_feature_group(feature_name: str) -> str:
    """
    Heuristic grouping of features into interpretable blocks:
      - "State effects"
      - "AR / time-series"
      - "NAICS mix"
      - "Calendar"
      - "Other"
    """
    # State dummies
    if feature_name.startswith("State_"):
        return "State effects"

    # NAICS composition (raw or lagged)
    if feature_name.startswith("share_NAICS2_") or "NAICS2" in feature_name:
        return "NAICS mix"

    # Calendar
    if feature_name in {"year", "month", "quarter", "weekofyear", "weekday"}:
        return "Calendar"

    # AR / time-series features (target lags, rolling, ewm, etc.)
    if ("lag" in feature_name) or ("rollmean" in feature_name) or ("ewm" in feature_name):
        return "AR / time-series"

    # Fallback
    return "Other"



def compute_feature_importances(
    models: Dict[str, RegressorMixin],
    X: pd.DataFrame,
    normalize: bool = True,
    use_abs: bool = True,
) -> pd.DataFrame:
    """
    Compute model-based feature importances for a set of fitted models.

    Supports:
      - Tree/boosting models with `feature_importances_`
        (XGBRegressor, LGBMRegressor, CatBoostRegressor, etc.)
      - Linear models with `coef_` (Ridge, Lasso, ElasticNet, PLSRegression),
        where importance is based on |coef_j| * sd(X_j) to account for
        feature scale.

    Parameters
    ----------
    models : dict
        {model_name: fitted_estimator}
        The models must already be fitted on X.
    X : pd.DataFrame
        Feature matrix used for training (or aligned with the model).
        Column order defines the feature order.
    normalize : bool, default True
        If True, normalizes importances for each model so that they sum to 1.
    use_abs : bool, default True
        If True, uses absolute value of linear coefficients before computing
        importance (i.e. |coef|). This is usually what you want for
        interpretability.

    Returns
    -------
    importances : pd.DataFrame
        DataFrame with index = feature names, columns = model names.
        Each column gives the (optionally normalized) importance of each feature
        for that model.
    """
    feature_names = list(X.columns)
    n_features = len(feature_names)
    X_values = X.to_numpy(dtype=float)

    # std per feature (used for linear models)
    std = X_values.std(axis=0, ddof=0)
    # Avoid NaNs if some features are constant
    std[std == 0.0] = 0.0

    imp_dict: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        importances: np.ndarray | None = None

        # ----------------------------------------------------------
        # 1) Tree / boosting models with feature_importances_
        # ----------------------------------------------------------
        if hasattr(model, "feature_importances_"):
            raw = np.asarray(model.feature_importances_, dtype=float)
            if raw.shape[0] != n_features:
                raise ValueError(
                    f"Model '{name}' has feature_importances_ of length {raw.shape[0]}, "
                    f"but X has {n_features} features."
                )
            importances = raw

        # ----------------------------------------------------------
        # 2) Linear / PLS models with coef_ (scale-aware)
        # ----------------------------------------------------------
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)

            # Handle shape (n_features,), (1, n_features), etc.
            if coef.ndim == 1:
                coef_vec = coef
            elif coef.ndim == 2:
                # If multi-target, average across targets
                if coef.shape[0] == n_features or coef.shape[1] == n_features:
                    coef_vec = coef.ravel()
                else:
                    coef_vec = np.mean(coef, axis=0)
            else:
                raise ValueError(
                    f"Unexpected coef_ shape {coef.shape} for model '{name}'."
                )

            if coef_vec.shape[0] != n_features:
                raise ValueError(
                    f"Model '{name}' has coef_ of length {coef_vec.shape[0]}, "
                    f"but X has {n_features} features."
                )

            if use_abs:
                coef_vec = np.abs(coef_vec)

            # SCALE-AWARE: multiply by feature std to get standardized effect
            raw = coef_vec * std
            importances = raw

        else:
            # Model does not expose a standard importance interface → skip
            continue

        # ----------------------------------------------------------
        # 3) Normalize if requested
        # ----------------------------------------------------------
        if normalize:
            total = np.sum(importances)
            if total > 0:
                importances = importances / total

        imp_dict[name] = importances

    if not imp_dict:
        raise ValueError(
            "No feature importances could be extracted from the provided models. "
            "Make sure they expose either 'feature_importances_' or 'coef_'."
        )

    imp_df = pd.DataFrame(imp_dict, index=feature_names)

    return imp_df

def aggregate_feature_importances_by_group(
    importances: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate feature importances by coarse feature groups.

    Groups (heuristic, based on feature name patterns):
      - "State effects"     (features starting with "State_")
      - "NAICS mix"         (features like share_NAICS2_... or containing "NAICS2")
      - "Calendar"          (year, month, quarter, weekofyear, weekday)
      - "AR / time-series"  (features containing 'lag', 'rollmean', or 'ewm')
      - "Other"             (everything else)

    Parameters
    ----------
    importances : pd.DataFrame
        Feature importances as returned by compute_feature_importances,
        with index = feature names, columns = model names.

    Returns
    -------
    grouped_importances : pd.DataFrame
        DataFrame with index = group names, columns = model names.
        Each entry is the sum of importances of all features in that group.
    """
    if importances.empty:
        raise ValueError("importances is empty.")

    groups = [ _infer_feature_group(f) for f in importances.index ]

    tmp = importances.copy()
    tmp["__group__"] = groups

    grouped = tmp.groupby("__group__", as_index=True).sum()
    # Make sure group name is the index
    grouped.index.name = "feature_group"

    return grouped
