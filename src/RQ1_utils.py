from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Optional

from src.models import get_model_configs, instantiate_models
from src.training import fit_global_models

from statsmodels.tsa.stattools import acf

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, make_scorer

def yearly_train_ends(meta: pd.DataFrame, date_col: str = "Date") -> list[pd.Timestamp]:
    dates = pd.to_datetime(meta[date_col], errors="raise")
    tmp = pd.DataFrame({"Date": dates})
    tmp["year"] = tmp["Date"].dt.year
    ends = tmp.groupby("year")["Date"].max().sort_values()
    return list(ends.values)

def rolling_origin_masks(
    meta: pd.DataFrame,
    date_col: str,
    train_ends: list[pd.Timestamp],
    test_size: int = 12,
) -> list[dict]:
    dates = pd.to_datetime(meta[date_col], errors="raise")
    unique_dates = pd.Index(np.sort(dates.unique()))

    splits = []
    for te in train_ends:
        te = pd.to_datetime(te)

        train_dates = unique_dates[unique_dates <= te]
        future_dates = unique_dates[unique_dates > te]
        if len(train_dates) == 0 or len(future_dates) == 0:
            continue

        test_dates = future_dates[:test_size]  # truncated if not enough

        train_mask = dates.isin(train_dates)
        test_mask  = dates.isin(test_dates)

        if (train_mask & test_mask).any():
            raise RuntimeError("Overlap detected between train and test.")

        splits.append({
            "train_end": te,
            "test_start": test_dates.min(),
            "test_end": test_dates.max(),
            "n_train_rows": int(train_mask.sum()),
            "n_test_rows": int(test_mask.sum()),
            "n_train_weeks": len(train_dates),
            "n_test_weeks": len(test_dates),
            "train_mask": train_mask,
            "test_mask": test_mask,
        })

    return splits

# ---------- helpers ----------

def validate_splits_increasing(splits: list[dict]) -> None:
    splits_sorted = sorted(splits, key=lambda s: pd.to_datetime(s["train_end"]))
    train_counts = [int(s["train_mask"].sum()) for s in splits_sorted]
    if any(train_counts[i] >= train_counts[i + 1] for i in range(len(train_counts) - 1)):
        raise ValueError(
            "Train rows are not strictly increasing across splits. "
            f"Train counts: {train_counts}"
        )


def seasonal_naive_direct(
    y: pd.Series,
    meta: pd.DataFrame,
    group_col: str,
    seasonal_period: int,
    h: int
) -> pd.Series:
    # predicts y_{t+h} using y_{t+h-m} => shift by (m - h)
    return y.groupby(meta[group_col], sort=False).shift(seasonal_period - h)


def mase_scale_per_state(
    y: pd.Series,
    meta: pd.DataFrame,
    group_col: str,
    train_mask: pd.Series,
    seasonal_period: int
) -> pd.Series:
    y_tr = y.where(train_mask)
    y_lag = y.groupby(meta[group_col], sort=False).shift(seasonal_period).where(train_mask)
    abs_err = (y_tr - y_lag).abs()
    scale_by_state = abs_err.groupby(meta[group_col], sort=False).mean()
    return meta[group_col].map(scale_by_state)


def rmse(y_true, y_pred) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def safe_mase(y_true, y_pred, scale) -> float:
    scale = np.asarray(scale, dtype=float)
    err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    ok = np.isfinite(scale) & (scale > 0) & np.isfinite(err)
    if ok.sum() == 0:
        return np.nan
    return float(np.mean(err[ok] / scale[ok]))


def compute_metrics_from_preds(preds_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gcols = ["split_id", "train_end", "horizon", "model", "set"]
    for keys, g in preds_df.groupby(gcols, sort=False):
        split_id, train_end, h, model, set_name = keys
        rows.append({
            "split_id": int(split_id),
            "train_end": pd.to_datetime(train_end),
            "horizon": int(h),
            "model": model,
            "set": set_name,
            "n_rows": int(len(g)),
            "MAE": float(mean_absolute_error(g["y_true"], g["y_pred"])),
            "RMSE": rmse(g["y_true"], g["y_pred"]),
            "MASE": safe_mase(g["y_true"], g["y_pred"], g["mase_scale"]),
        })
    return pd.DataFrame(rows)


def save_outputs_csv(
    metrics_df: pd.DataFrame,
    perm_df: Optional[pd.DataFrame],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = metrics_df.copy()
    metrics_df["train_end"] = pd.to_datetime(metrics_df["train_end"]).dt.strftime("%Y-%m-%d")
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    if perm_df is not None:
        perm_df = perm_df.copy()
        perm_df["train_end"] = pd.to_datetime(perm_df["train_end"]).dt.strftime("%Y-%m-%d")
        perm_df.to_csv(out_dir / "perm_importance.csv", index=False)


# ---------- main ----------

def run_global(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    splits: list[dict],
    horizons: list[int],
    *,
    group_col: str = "State",
    date_col: str = "Date",
    seasonal_period: int = 12,
    use_linear: bool = True,
    use_tree: bool = True,
    compute_perm_importance: bool = False,
    savedir: str | Path | None = None,
    run_name: str = "run_global",
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:

    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError("X, y, meta must have same length and be row-aligned.")
    if date_col not in meta.columns or group_col not in meta.columns:
        raise ValueError(f"meta must contain {date_col} and {group_col}")

    meta = meta.copy()
    meta[date_col] = pd.to_datetime(meta[date_col], errors="raise")

    # order splits + validate increasing train sizes
    splits = sorted(splits, key=lambda s: pd.to_datetime(s["train_end"]))
    validate_splits_increasing(splits)

    out_dir = (Path(savedir) / run_name) if savedir is not None else None

    neg_rmse_scorer = make_scorer(rmse, greater_is_better=False)
    preds_parts = []
    perm_parts = [] if compute_perm_importance else None

    for split_id, split in enumerate(splits, start=1):
        train_end = pd.to_datetime(split["train_end"])

        if (split["train_mask"] & split["test_mask"]).any():
            raise RuntimeError(f"Split {split_id}: overlap between train and test masks.")

        for h in horizons:
            # direct target
            y_h = y.groupby(meta[group_col], sort=False).shift(-h)
            valid = y_h.notna()

            train_mask = split["train_mask"] & valid
            test_mask  = split["test_mask"] & valid

            X_train, y_train = X.loc[train_mask], y_h.loc[train_mask]
            X_test,  y_test  = X.loc[test_mask],  y_h.loc[test_mask]

            # MASE scale from TRAIN slice (per-state)
            mase_scale = mase_scale_per_state(y, meta, group_col, train_mask, seasonal_period)

            # baseline predictions (test only)
            y_naive = seasonal_naive_direct(y, meta, group_col, seasonal_period, h)
            test_mask_naive = split["test_mask"] & valid & y_naive.notna()
            if test_mask_naive.any():
                base = meta.loc[test_mask_naive, [date_col, group_col]].copy()
                base.rename(columns={date_col: "Date", group_col: "State"}, inplace=True)
                base["split_id"] = split_id
                base["train_end"] = train_end
                base["horizon"] = h
                base["set"] = "test"
                base["model"] = "Seasonal Naive"
                base["y_true"] = y_h.loc[test_mask_naive].values
                base["y_pred"] = y_naive.loc[test_mask_naive].values
                base["mase_scale"] = mase_scale.loc[test_mask_naive].values
                preds_parts.append(base)

            # fit pooled/global models (your existing plumbing)
            model_configs = get_model_configs(
                n_samples=len(X_train),
                use_linear=use_linear,
                use_tree=use_tree,
            )
            models = instantiate_models(model_configs)
            fitted = fit_global_models(models=models, X_train=X_train, y_train=y_train)

            for model_name, model in fitted.items():
                # train fitted
                yhat_train = model.predict(X_train)
                tr = meta.loc[train_mask, [date_col, group_col]].copy()
                tr.rename(columns={date_col: "Date", group_col: "State"}, inplace=True)
                tr["split_id"] = split_id
                tr["train_end"] = train_end
                tr["horizon"] = h
                tr["set"] = "train"
                tr["model"] = model_name
                tr["y_true"] = y_train.values
                tr["y_pred"] = yhat_train
                tr["mase_scale"] = mase_scale.loc[train_mask].values
                preds_parts.append(tr)

                # test forecast
                yhat_test = model.predict(X_test)
                te = meta.loc[test_mask, [date_col, group_col]].copy()
                te.rename(columns={date_col: "Date", group_col: "State"}, inplace=True)
                te["split_id"] = split_id
                te["train_end"] = train_end
                te["horizon"] = h
                te["set"] = "test"
                te["model"] = model_name
                te["y_true"] = y_test.values
                te["y_pred"] = yhat_test
                te["mase_scale"] = mase_scale.loc[test_mask].values
                preds_parts.append(te)

                if compute_perm_importance:
                    pi = permutation_importance(
                        model, X_test, y_test,
                        scoring=neg_rmse_scorer,
                        n_repeats=5,
                        random_state=0,
                        n_jobs=-1,
                    )
                    imp = pd.DataFrame({
                        "feature": X_test.columns,
                        "importance": pi.importances_mean,
                        "importance_std": pi.importances_std,
                    })
                    imp["split_id"] = split_id
                    imp["train_end"] = train_end
                    imp["horizon"] = h
                    imp["model"] = model_name
                    perm_parts.append(imp)

    preds_df = pd.concat(preds_parts, ignore_index=True)
    metrics_df = compute_metrics_from_preds(preds_df)
    perm_df = pd.concat(perm_parts, ignore_index=True) if compute_perm_importance else None

    if out_dir is not None:
        save_outputs_csv(metrics_df, perm_df, out_dir)

    return preds_df, metrics_df, perm_df


def validate_predictions_basic(
    preds_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    train_end_col: str = "train_end",
    horizon_col: str = "horizon",
    model_col: str = "model",
    set_col: str = "set",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> None:
    """
    Minimal sanity checks for backtest predictions.

    Checks:
      1) y_true and y_pred contain no null values
      2) consistent number of predictions per (split,horizon,model,set)
      3) no temporal leakage: test dates > train_end

    Raises ValueError if a check fails.
    """

    # ---------- 1) No nulls ----------
    nulls = preds_df[[y_true_col, y_pred_col]].isna().sum()
    if (nulls > 0).any():
        raise ValueError(
            "Null values detected in predictions:\n"
            f"{nulls[nulls > 0]}"
        )

    # ---------- 2) Consistent counts ----------
    counts = (
        preds_df
        .groupby(["split_id", horizon_col, model_col, set_col], sort=False)
        .size()
        .reset_index(name="n_rows")
    )

    # For each (split,horizon,set), all models must have the same number of rows
    coverage = (
        counts
        .groupby(["split_id", horizon_col, set_col], sort=False)["n_rows"]
        .nunique()
        .reset_index(name="n_unique_counts")
    )

    bad = coverage[coverage["n_unique_counts"] > 1]
    if len(bad) > 0:
        raise ValueError(
            "Inconsistent number of predictions across models "
            "for the same split/horizon/set:\n"
            f"{bad}"
        )

    # ---------- 3) No time leakage ----------
    df = preds_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df[train_end_col] = pd.to_datetime(df[train_end_col], errors="raise")

    leaked = df[(df[set_col] == "test") & (df[date_col] <= df[train_end_col])]
    if len(leaked) > 0:
        raise ValueError(
            "Temporal leakage detected: test observations at or before train_end.\n"
            f"Examples:\n{leaked[[date_col, train_end_col]].head()}"
        )

    print("✓ Basic prediction validation passed.")

def build_horizon_results_table(
    metrics_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Build a tidy per-horizon results table with:
    model | split_year | train_RMSE | test_RMSE | test_MAE | test_MASE

    Ordered by split_year (then model).
    """

    df = metrics_df.query("horizon == @horizon").copy()
    df["split_year"] = pd.to_datetime(df["train_end"]).dt.year

    # TRAIN metrics
    train = (
        df.query("set == 'train'")
          .loc[:, ["split_year", "model", "RMSE"]]
          .rename(columns={"RMSE": "train_RMSE"})
    )

    # TEST metrics
    test = (
        df.query("set == 'test'")
          .loc[:, ["split_year", "model", "RMSE", "MAE", "MASE"]]
          .rename(columns={
              "RMSE": "test_RMSE",
              "MAE": "test_MAE",
              "MASE": "test_MASE",
          })
    )

    # Merge and order by split_year
    out = (
        train
        .merge(
            test,
            on=["split_year", "model"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["split_year", "model"])
        .reset_index(drop=True)
    )

    return out

def summarize_metrics_mean_std(
    metrics_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Summarize TEST performance for a given horizon.
    
    Rows: models
    Columns: RMSE, MAE, MASE
    Values: mean ± std across splits
    """

    df = (
        metrics_df
        .query("set == 'test' and horizon == @horizon")
        .loc[:, ["model", "RMSE", "MAE", "MASE"]]
    )

    summary = (
        df
        .groupby("model")
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MASE_mean=("MASE", "mean"),
            MASE_std=("MASE", "std"),
        )
    )

    # format as mean ± std
    out = pd.DataFrame({
        "RMSE": summary.apply(
            lambda r: f"{r.RMSE_mean:.3f} ± {r.RMSE_std:.3f}", axis=1
        ),
        "MAE": summary.apply(
            lambda r: f"{r.MAE_mean:.3f} ± {r.MAE_std:.3f}", axis=1
        ),
        "MASE": summary.apply(
            lambda r: f"{r.MASE_mean:.3f} ± {r.MASE_std:.3f}", axis=1
        ),
    })

    return out.sort_index()


def plot_stability_across_splits(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "MASE",
    aggregation: str = "monthly",
    exclude_models: tuple[str, ...] = ("Seasonal Naive",),
):
    """
    Boxplots of split-level performance across learned models,
    faceted by horizon (one figure per aggregation).
    """

    df = (
        metrics_df
        .query("set == 'test'")
        .copy()
    )

    if exclude_models:
        df = df[~df["model"].isin(exclude_models)]

    # stable categorical order
    df["model"] = df["model"].astype("category")

    g = sns.catplot(
        data=df,
        x="model",
        y=metric,
        col="horizon",
        kind="box",
        col_wrap=3,      # adjust if needed
        sharey=True,
        height=4,
        aspect=1.1,
    )

    g.set_titles("Horizon h = {col_name}")
    g.set_axis_labels("Model", metric)
    g.fig.suptitle(
        f"{aggregation.capitalize()} forecasting performance across splits",
        y=1.05,
        fontsize=14,
    )

    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

def plot_fit_and_forecast(
    preds_df: pd.DataFrame,
    model: str,
    *,
    split_idx: int = 0,
    horizon: int = 3,
    state: str | None = None,
):
    """
    Diagnostic plot of actual vs fitted/forecast values.

    - model: model name
    - split_idx: which split to visualize (0 = first, chronological order)
    - horizon: forecast horizon
    - state:
        - None  -> national aggregation (mean across states)
        - str   -> plot a single state
    """

    df = preds_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["train_end"] = pd.to_datetime(df["train_end"])

    # --- select split ---
    train_ends = sorted(df["train_end"].unique())
    if split_idx < 0 or split_idx >= len(train_ends):
        raise ValueError(f"split_idx must be in [0, {len(train_ends) - 1}]")

    train_end = train_ends[split_idx]

    # --- filter slice ---
    df = df[
        (df["train_end"] == train_end) &
        (df["horizon"] == horizon) &
        (df["model"] == model)
    ]

    if state is not None:
        df = df[df["State"] == state]
        if df.empty:
            raise ValueError(f"No data found for state='{state}'.")

    if df.empty:
        raise ValueError("No data found for the given arguments.")

    # --- aggregation ---
    if state is None:
        # national mean
        actual = (
            df.groupby("Date", as_index=False)["y_true"]
              .mean()
              .rename(columns={"y_true": "actual"})
              .sort_values("Date")
        )

        pred = (
            df.groupby(["set", "Date"], as_index=False)["y_pred"]
              .mean()
              .rename(columns={"y_pred": "pred"})
        )

        title_loc = "National mean"
    else:
        # single state (no aggregation)
        actual = (
            df[["Date", "y_true"]]
            .drop_duplicates(subset=["Date"])
            .rename(columns={"y_true": "actual"})
            .sort_values("Date")
        )

        pred = (
            df[["set", "Date", "y_pred"]]
            .rename(columns={"y_pred": "pred"})
        )

        title_loc = f"State: {state}"

    fitted = pred[pred["set"] == "train"].sort_values("Date")
    forecast = pred[pred["set"] == "test"].sort_values("Date")

    # --- plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(actual["Date"], actual["actual"], label="Actual", linewidth=2)
    plt.plot(fitted["Date"], fitted["pred"], label="Fitted (train)", alpha=0.9)
    plt.plot(forecast["Date"], forecast["pred"], label="Forecast (test)", alpha=0.9, color='red')

    plt.axvline(train_end, linestyle="--", color="black", label="Train end")

    plt.title(f"{title_loc} | {model} | h = {horizon} | split_idx = {split_idx}")
    plt.xlabel("Date")
    plt.ylabel("HospRisk")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residuals_distribution(
    preds_df: pd.DataFrame,
    model: str,
    *,
    split_idx: int = 0,
    horizon: int = 3,
    state: str | None = None,
    bins: int = 40,
):
    """
    Plot residual distribution (y_true - y_pred) for a given model/split/horizon.
    - state=None -> national (mean across states per date)
    - state=str  -> single state
    Plots train and test residual histograms.
    """

    df = preds_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["train_end"] = pd.to_datetime(df["train_end"])

    # select split
    train_ends = sorted(df["train_end"].unique())
    if split_idx < 0 or split_idx >= len(train_ends):
        raise ValueError(f"split_idx must be in [0, {len(train_ends) - 1}]")
    train_end = train_ends[split_idx]

    # filter slice
    df = df[
        (df["train_end"] == train_end) &
        (df["horizon"] == horizon) &
        (df["model"] == model)
    ]
    if state is not None:
        df = df[df["State"] == state]
        if df.empty:
            raise ValueError(f"No data found for state='{state}'.")
    if df.empty:
        raise ValueError("No data found for the given arguments.")

    # aggregate if national
    if state is None:
        # mean across states per (set, Date)
        agg = (
            df.groupby(["set", "Date"], as_index=False)
              .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
        )
        title_loc = "National mean"
    else:
        # single state (dedupe by set/date just in case)
        agg = (
            df.groupby(["set", "Date"], as_index=False)
              .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
        )
        title_loc = f"State: {state}"

    agg["resid"] = agg["y_true"] - agg["y_pred"]

    res_train = agg.loc[agg["set"] == "train", "resid"].to_numpy()
    res_test  = agg.loc[agg["set"] == "test",  "resid"].to_numpy()

    # common x-limits for comparability
    all_res = np.concatenate([res_train, res_test]) if len(res_train) and len(res_test) else (res_train if len(res_train) else res_test)
    if len(all_res) == 0:
        raise ValueError("No residuals available to plot.")
    lo, hi = np.percentile(all_res, [1, 99])  # robust limits

    plt.figure(figsize=(10, 4))
    plt.hist(res_train, bins=bins, range=(lo, hi), alpha=0.6, label="Train residuals")
    plt.hist(res_test,  bins=bins, range=(lo, hi), alpha=0.6, label="Test residuals")
    plt.axvline(0.0, linestyle="--", color="black", linewidth=1)

    plt.title(f"Residual distribution | {title_loc} | {model} | h={horizon} | split_idx={split_idx}")
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # quick numeric summary (useful while debugging)
    print("Train resid: mean={:.4g}, std={:.4g}".format(np.mean(res_train) if len(res_train) else np.nan,
                                                       np.std(res_train) if len(res_train) else np.nan))
    print("Test  resid: mean={:.4g}, std={:.4g}".format(np.mean(res_test) if len(res_test) else np.nan,
                                                       np.std(res_test) if len(res_test) else np.nan))

def plot_residuals_over_time(
    preds_df: pd.DataFrame,
    model: str,
    *,
    split_idx: int = 0,
    horizon: int = 3,
    state: str | None = None,
):
    """
    Residuals (y_true - y_pred) over time for a given model/split/horizon.

    - state=None -> national aggregation (mean across states per (set, Date))
    - state=str  -> single state
    Plots train and test residuals as two lines, with train_end marked.
    """

    df = preds_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["train_end"] = pd.to_datetime(df["train_end"])

    # select split
    train_ends = sorted(df["train_end"].unique())
    if split_idx < 0 or split_idx >= len(train_ends):
        raise ValueError(f"split_idx must be in [0, {len(train_ends) - 1}]")
    train_end = train_ends[split_idx]

    # filter slice
    df = df[
        (df["train_end"] == train_end) &
        (df["horizon"] == horizon) &
        (df["model"] == model)
    ]

    if state is not None:
        df = df[df["State"] == state]
        if df.empty:
            raise ValueError(f"No data found for state='{state}'.")

    if df.empty:
        raise ValueError("No data found for the given arguments.")

    # aggregate to national or keep single state (still aggregate by set/date to be safe)
    if state is None:
        title_loc = "National mean"
    else:
        title_loc = f"State: {state}"

    ts = (
        df.groupby(["set", "Date"], as_index=False)
          .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
          .sort_values("Date")
    )
    ts["resid"] = ts["y_true"] - ts["y_pred"]

    tr = ts[ts["set"] == "train"][["Date", "resid"]].sort_values("Date")
    te = ts[ts["set"] == "test"][["Date", "resid"]].sort_values("Date")

    plt.figure(figsize=(10, 4))
    if not tr.empty:
        plt.plot(tr["Date"], tr["resid"], label="Train residuals")
    if not te.empty:
        plt.plot(te["Date"], te["resid"], label="Test residuals")

    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.axvline(train_end, linestyle="--", color="black", linewidth=1, label="Train end")

    plt.title(f"Residuals over time | {title_loc} | {model} | h={horizon} | split_idx={split_idx}")
    plt.xlabel("Date")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residuals_acf_national(
    preds_df: pd.DataFrame,
    model: str,
    *,
    split_idx: int = 0,
    horizon: int = 3,
    nlags: int = 12,
):
    """
    Plot ACF of residuals (y_true - y_pred) at national level (mean across states per date),
    for a given model/split/horizon.

    Produces two ACF plots: train residuals and test residuals.
    """

    df = preds_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["train_end"] = pd.to_datetime(df["train_end"])

    # select split
    train_ends = sorted(df["train_end"].unique())
    if split_idx < 0 or split_idx >= len(train_ends):
        raise ValueError(f"split_idx must be in [0, {len(train_ends) - 1}]")
    train_end = train_ends[split_idx]

    # filter slice
    df = df[
        (df["train_end"] == train_end) &
        (df["horizon"] == horizon) &
        (df["model"] == model)
    ]
    if df.empty:
        raise ValueError("No data found for the given (model, split_idx, horizon).")

    # national aggregation per (set, Date)
    ts = (
        df.groupby(["set", "Date"], as_index=False)
          .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
          .sort_values("Date")
    )
    ts["resid"] = ts["y_true"] - ts["y_pred"]

    def _plot_one_acf(residuals: np.ndarray, title: str):
        residuals = residuals[np.isfinite(residuals)]
        n = len(residuals)
        if n < (nlags + 2):
            plt.figure(figsize=(7, 3))
            plt.title(f"{title} (too few points: n={n})")
            plt.axis("off")
            plt.show()
            return

        vals = acf(residuals, nlags=nlags, fft=True)  # includes lag 0
        lags = np.arange(len(vals))

        # approx 95% CI under white-noise assumption
        conf = 1.96 / np.sqrt(n)

        plt.figure(figsize=(8, 3.5))
        plt.stem(lags, vals, basefmt=" ")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.axhline(conf, linestyle="--", linewidth=1)
        plt.axhline(-conf, linestyle="--", linewidth=1)
        plt.xlim(0, nlags)
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.title(f"{title} | n={n} | 95% band ≈ ±{conf:.3f}")
        plt.tight_layout()
        plt.show()

    # train/test residual vectors (already aggregated to one value per date)
    res_train = ts.loc[ts["set"] == "train", "resid"].to_numpy()


    _plot_one_acf(res_train, f"ACF residuals (TRAIN) | National mean | {model} | h={horizon} | split_idx={split_idx}")

def plot_rmse_per_fold(
    metrics_df: pd.DataFrame,
    *,
    horizon: int,
    exclude_models: tuple[str, ...] = ("Seasonal Naive",),
    title: str | None = None,
):
    """
    Line plot of TEST RMSE per fold (train_end) for a given horizon.
    Excludes baseline models by default.
    """
    df = metrics_df.query("set == 'test' and horizon == @horizon").copy()
    df["train_end"] = pd.to_datetime(df["train_end"])

    if exclude_models:
        df = df[~df["model"].isin(exclude_models)]

    if df.empty:
        raise ValueError("No rows left after filtering (check horizon/model names).")

    # pivot: rows = folds (train_end), cols = models, values = RMSE
    piv = (
        df.pivot_table(index="train_end", columns="model", values="RMSE", aggfunc="mean")
          .sort_index()
    )

    plt.figure(figsize=(10, 4))
    for model in piv.columns:
        plt.plot(piv.index, piv[model], marker="o", label=model)

    plt.xlabel("Fold (train_end)")
    plt.ylabel("RMSE (test)")
    plt.title(title or f"Test RMSE per fold (h={horizon})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_delta_rmse_to_best_per_fold(
    metrics_df: pd.DataFrame,
    *,
    horizon: int,
    exclude_models: tuple[str, ...] = ("Seasonal Naive",),
    title: str | None = None,
):
    df = metrics_df.query("set == 'test' and horizon == @horizon").copy()
    df["train_end"] = pd.to_datetime(df["train_end"])
    if exclude_models:
        df = df[~df["model"].isin(exclude_models)]

    piv = (
        df.pivot_table(index="train_end", columns="model", values="RMSE", aggfunc="mean")
          .sort_index()
    )
    delta = piv.sub(piv.min(axis=1), axis=0)

    plt.figure(figsize=(10, 4))
    for model in delta.columns:
        plt.plot(delta.index, delta[model], marker="o", label=model)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Fold (train_end)")
    plt.ylabel("ΔRMSE to best model (test)")
    plt.title(title or f"ΔRMSE to best per fold (h={horizon})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ytrue_vs_ypred(
    preds_df,
    *,
    model: str,
    horizon: int,
    set_name: str = "test",
):
    """
    Scatter plot of y_true vs y_pred for a given model and horizon.
    """

    df = preds_df.query(
        "model == @model and horizon == @horizon and set == @set_name"
    )

    if df.empty:
        raise ValueError("No data found for the given model/horizon/set.")

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, c=df['split_id'], alpha=0.4)
    plt.colorbar(label="Split index")
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("Observed value")
    plt.ylabel("Predicted value")
    plt.title(f"{model} | h={horizon} | {set_name}")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.tight_layout()
    plt.show()

#permutation importance

def summarize_perm_importance(
    perm_df: pd.DataFrame,
    *,
    model: str,
    horizon: int,
    top_k: int = 10,
):
    """
    Summarize permutation importance across splits
    for a fixed model and horizon.
    """

    df = (
        perm_df
        .query("model == @model and horizon == @horizon")
        .copy()
    )

    summary = (
        df
        .groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            pos_frac=("importance", lambda x: (x > 0).mean()),
        )
        .sort_values("mean_importance", ascending=False)
    )

    return summary.head(top_k)


def plot_perm_importance_bar(summary_df: pd.DataFrame, title: str):
    plt.figure(figsize=(8, 0.35 * len(summary_df) + 1))
    plt.barh(
        summary_df.index[::-1],
        summary_df["mean_importance"].iloc[::-1],
        xerr=summary_df["std_importance"].iloc[::-1],
    )
    plt.xlabel("Permutation importance (ΔRMSE)")
    plt.title(title)
    plt.tight_layout()
    plt.show()



#RQ2

def run_local(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    splits: list[dict],
    horizons: list[int],
    *,
    group_col: str = "State",
    date_col: str = "Date",
    seasonal_period: int = 12,
    use_linear: bool = True,
    use_tree: bool = True,
    compute_perm_importance: bool = False,
    min_train: int = 12,
    savedir: str | Path | None = None,
    run_name: str = "run_local",
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Local (no pooling): for each split x horizon, fit one model per state on that state's training slice,
    then predict that state's train/test slices.
    """

    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError("X, y, meta must have same length and be row-aligned.")
    if date_col not in meta.columns or group_col not in meta.columns:
        raise ValueError(f"meta must contain {date_col} and {group_col}")

    meta = meta.copy()
    meta[date_col] = pd.to_datetime(meta[date_col], errors="raise")

    # order splits + validate increasing train sizes
    splits = sorted(splits, key=lambda s: pd.to_datetime(s["train_end"]))
    validate_splits_increasing(splits)

    out_dir = (Path(savedir) / run_name) if savedir is not None else None

    neg_rmse_scorer = make_scorer(rmse, greater_is_better=False)
    preds_parts: list[pd.DataFrame] = []
    perm_parts: list[pd.DataFrame] = [] if compute_perm_importance else []

    # loop CV splits
    for split_id, split in enumerate(splits, start=1):
        train_end = pd.to_datetime(split["train_end"])

        if (split["train_mask"] & split["test_mask"]).any():
            raise RuntimeError(f"Split {split_id}: overlap between train and test masks.")

        # precompute for this split (used for baseline + MASE scale)
        # NOTE: y_h depends on horizon, so we compute inside horizon loop.
        for h in horizons:
            # direct target: y(t+h) aligned at time t
            y_h = y.groupby(meta[group_col], sort=False).shift(-h)
            valid = y_h.notna()

            train_mask = split["train_mask"] & valid
            test_mask = split["test_mask"] & valid

            # baseline predictions (test only), pooled computation but per-state values
            mase_scale_all = mase_scale_per_state(
                y=y, meta=meta, group_col=group_col, train_mask=split["train_mask"], seasonal_period=seasonal_period
            )
            y_naive = seasonal_naive_direct(y, meta, group_col, seasonal_period, h)
            test_mask_naive = split["test_mask"] & valid & y_naive.notna()
            if test_mask_naive.any():
                base = meta.loc[test_mask_naive, [date_col, group_col]].copy()
                base.rename(columns={date_col: "Date", group_col: "State"}, inplace=True)
                base["split_id"] = split_id
                base["train_end"] = train_end
                base["horizon"] = h
                base["set"] = "test"
                base["model"] = "Seasonal Naive"
                base["y_true"] = y_h.loc[test_mask_naive].values
                base["y_pred"] = y_naive.loc[test_mask_naive].values
                base["mase_scale"] = mase_scale_all.loc[test_mask_naive].values
                preds_parts.append(base)

            # local fitting: one model per state
            # (iterate only over states that appear in train or test for this split/h)
            states = pd.Index(meta.loc[train_mask | test_mask, group_col].unique())

            for state in states:
                state_mask = (meta[group_col] == state)

                st_train_mask = train_mask & state_mask
                st_test_mask = test_mask & state_mask

                # skip if no data
                if st_train_mask.sum() < min_train or st_test_mask.sum() == 0:
                    continue

                X_train, y_train = X.loc[st_train_mask], y_h.loc[st_train_mask]
                X_test, y_test = X.loc[st_test_mask], y_h.loc[st_test_mask]

                # Guard against degenerate targets
                if y_train.nunique(dropna=True) < 2:
                    # still allow predictions? simplest: skip models for this state/h/split
                    continue

                # pick model set based on *local* training size
                model_configs = get_model_configs(
                    n_samples=len(X_train),
                    use_linear=use_linear,
                    use_tree=use_tree,
                )
                models = instantiate_models(model_configs)

                # fit each model on the state's data
                # (re-use your global fitter if it just loops and .fit()'s; otherwise fit here)
                fitted = fit_global_models(models=models, X_train=X_train, y_train=y_train)

                # MASE scale (state-specific slice) via the precomputed per-row scale
                # (mase_scale_per_state returns a vector aligned to rows; selecting masks is enough)
                mase_scale_train = mase_scale_all.loc[st_train_mask].values
                mase_scale_test = mase_scale_all.loc[st_test_mask].values

                for model_name, model in fitted.items():
                    # train fitted values
                    yhat_train = model.predict(X_train)
                    tr = meta.loc[st_train_mask, [date_col, group_col]].copy()
                    tr.rename(columns={date_col: "Date", group_col: "State"}, inplace=True)
                    tr["split_id"] = split_id
                    tr["train_end"] = train_end
                    tr["horizon"] = h
                    tr["set"] = "train"
                    tr["model"] = model_name
                    tr["y_true"] = y_train.values
                    tr["y_pred"] = yhat_train
                    tr["mase_scale"] = mase_scale_train
                    preds_parts.append(tr)

                    # test forecasts
                    yhat_test = model.predict(X_test)
                    te = meta.loc[st_test_mask, [date_col, group_col]].copy()
                    te.rename(columns={date_col: "Date", group_col: "State"}, inplace=True)
                    te["split_id"] = split_id
                    te["train_end"] = train_end
                    te["horizon"] = h
                    te["set"] = "test"
                    te["model"] = model_name
                    te["y_true"] = y_test.values
                    te["y_pred"] = yhat_test
                    te["mase_scale"] = mase_scale_test
                    preds_parts.append(te)

                    if compute_perm_importance:
                        pi = permutation_importance(
                            model, X_test, y_test,
                            scoring=neg_rmse_scorer,
                            n_repeats=5,
                            random_state=0,
                            n_jobs=-1,
                        )
                        imp = pd.DataFrame({
                            "feature": X_test.columns,
                            "importance": pi.importances_mean,
                            "importance_std": pi.importances_std,
                        })
                        imp["split_id"] = split_id
                        imp["train_end"] = train_end
                        imp["horizon"] = h
                        imp["model"] = model_name
                        imp["State"] = state
                        perm_parts.append(imp)

    preds_df = pd.concat(preds_parts, ignore_index=True) if preds_parts else pd.DataFrame()
    metrics_df = compute_metrics_from_preds(preds_df) if len(preds_df) else pd.DataFrame()
    perm_df = pd.concat(perm_parts, ignore_index=True) if compute_perm_importance and perm_parts else None

    if out_dir is not None:
        save_outputs_csv(metrics_df, perm_df, out_dir)

    return preds_df, metrics_df, perm_df
