from __future__ import annotations

import itertools
from typing import Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing


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

def results_to_df(results_list: list[dict]) -> pd.DataFrame:
    """
    Convert a list of model results dictionaries into a clean DataFrame.
    Expected keys: model_name, train_metrics, test_metrics.
    """
    rows = []
    for res in results_list:
        row = {
            "model": res["model_name"],
            "train_MAE": res["train_metrics"]["MAE"],
            "train_RMSE": res["train_metrics"]["RMSE"],
            "test_MAE": res["test_metrics"]["MAE"],
            "test_RMSE": res["test_metrics"]["RMSE"],
        }
        rows.append(row)

    return pd.DataFrame(rows)

def save_results_csv(
    df: pd.DataFrame,
    save_dir: Path,
    filename: str = "results.csv"
):
    """
    Save the evaluation results to a user-specified directory with a custom CSV filename.

    Parameters
    ----------
    df : pd.DataFrame
        Results table to save.
    save_dir : Path
        Directory where the file should be saved.
    filename : str
        Name of the CSV file (e.g., "sarima_results.csv").
    """

    # Ensure directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Construct file path
    out_path = save_dir / filename

    # Save
    df.to_csv(out_path, index=False)

    print(f"Saved results to: {out_path}")


# --------------------------------------------------------
# Step 1 — Global monthly aggregation
# --------------------------------------------------------

def aggregate_monthly_global(
    df: pd.DataFrame,
    date_col: str = "EventDate",
    target_cols: list[str] = ["Hospitalized", "Amputation"],
    freq: str = "MS",
    agg: str = "sum",
) -> pd.DataFrame:
    """
    Aggregate the incident-level data to a global monthly time series.

    Parameters
    ----------
    df : pd.DataFrame
        Original event-level dataframe.
    date_col : str
        Name of the date column.
    target_cols : list[str]
        Columns to aggregate (e.g. ["Hospitalized", "Amputation"]).
    freq : str
        Resampling frequency, default "MS" (month start).
    agg : str
        Aggregation function to apply within each period ("sum" or "mean").

    Returns
    -------
    monthly_df : pd.DataFrame
        Dataframe indexed by monthly Periods (freq=freq),
        with one column per target.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Daily aggregation first (in case there are multiple rows per day)
    daily = (
        df.groupby(date_col)[target_cols]
          .agg(agg)
          .sort_index()
    )

    # Monthly resampling
    monthly = daily.resample(freq).agg(agg)

    # Make sure we have a proper frequency set
    monthly = monthly.asfreq(freq)

    return monthly

# --------------------------------------------------------
# Order search for ARIMA / SARIMA
# --------------------------------------------------------

def search_arima_orders(
    y: pd.Series,
    p_range=range(0, 2),
    d_range=range(0, 2),
    q_range=range(0, 2),
    seasonal: bool = False,
    P_range=range(0, 2),
    D_range=range(0, 2),
    Q_range=range(0, 2),
    m: int = 12,
    ic: str = "aic",
    max_models: Optional[int] = None,
) -> pd.DataFrame:
    """
    Grid search over ARIMA / SARIMA orders using SARIMAX, ranked by IC.
    """
    warnings.filterwarnings("ignore")

    # 1. Generate all parameter combinations upfront using itertools
    pdq_combos = list(itertools.product(p_range, d_range, q_range))
    
    if seasonal:
        seasonal_combos = list(itertools.product(P_range, D_range, Q_range))
        # Add the period 'm' to the tuple
        seasonal_orders = [(x[0], x[1], x[2], m) for x in seasonal_combos]
    else:
        # If not seasonal, use a dummy list with None so the loop runs once per pdq
        seasonal_orders = [None]

    records: List[dict] = []
    models_tried = 0

    # 2. Flattened Loop: Cartesian product of non-seasonal and seasonal orders
    # We iterate through every combination of (order, seasonal_order)
    all_combinations = itertools.product(pdq_combos, seasonal_orders)

    for order, seasonal_order in all_combinations:
        
        # 3. Check max_models condition BEFORE fitting
        if max_models is not None and models_tried >= max_models:
            break

        try:
            # 4. Unified Model Fitting (DRY Principle)
            # SARIMAX handles seasonal_order=None gracefully if we pass it correctly,
            # but usually it expects (0,0,0,0) or explicit exclusion.
            # However, simpler logic is to just conditionally pass the argument.
            
            model_args = {
                "order": order,
                "enforce_stationarity": False,
                "enforce_invertibility": False
            }
            
            if seasonal_order:
                model_args["seasonal_order"] = seasonal_order

            model = sm.tsa.statespace.SARIMAX(y, **model_args)
            fitted = model.fit(disp=False)

            records.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": fitted.aic,
                "bic": fitted.bic,
            })
            
            models_tried += 1

        except Exception:
            # Skip models that fail to converge (LinAlgError, ValueError, etc.)
            continue

    results_df = pd.DataFrame(records)

    if results_df.empty:
        raise RuntimeError("No valid ARIMA/SARIMA models were successfully fitted.")

    # Sort and reset index
    results_df.sort_values(by=ic, ascending=True, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    
    return results_df

# Wrapper functions remain mostly the same, just ensure imports match
def find_best_arima(
    y: pd.Series,
    ic: str = "aic",
    p_range=range(0, 4),
    d_range=range(0, 3),
    q_range=range(0, 4),
) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
    
    results_df = search_arima_orders(
        y=y,
        p_range=p_range, d_range=d_range, q_range=q_range,
        seasonal=False,
        ic=ic,
    )
    best_order = results_df.loc[0, "order"]
    return results_df, best_order

def find_best_sarima(
    y: pd.Series,
    ic: str = "aic",
    p_range=range(0, 3),
    d_range=range(0, 2),
    q_range=range(0, 3),
    P_range=range(0, 2),
    D_range=range(0, 2),
    Q_range=range(0, 2),
    m: int = 12,
) -> Tuple[pd.DataFrame, Tuple[int, int, int], Tuple[int, int, int, int]]:
    
    results_df = search_arima_orders(
        y=y,
        p_range=p_range, d_range=d_range, q_range=q_range,
        seasonal=True,
        P_range=P_range, D_range=D_range, Q_range=Q_range, m=m,
        ic=ic,
    )
    best_order = results_df.loc[0, "order"]
    best_seasonal = results_df.loc[0, "seasonal_order"]
    return results_df, best_order, best_seasonal



def train_test_split_ts(
    y: pd.Series, 
    train_end: str | pd.Timestamp, 
    test_end: str | pd.Timestamp = "2024-12-30",
):
    """
    Train: up to and including train_end.
    Test: strictly after train_end, up to and including test_end.

    This creates:
    - y_train: all observations ≤ train_end
    - y_test: all observations > train_end and ≤ test_end
    """
    y = y.sort_index()

    # Train
    y_train = y.loc[:train_end]

    # Test (includes train_end, we remove it below)
    y_test = y.loc[train_end:test_end]

    # Remove the train_end boundary from test
    if train_end in y_test.index:
        y_test = y_test.iloc[1:]

    return y_train, y_test


def rolling_1step_sarimax(
    y: pd.Series,
    test_index: pd.DatetimeIndex,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> pd.Series:
    """
    Optimized expanding-window forecast using Warm Starts.
    """
    preds = []
    # We maintain the last fitted parameters to speed up the next fit (Warm Start)
    last_params = None

    for t in test_index:
        # Robust slicing: strictly less than current time 't'
        y_train = y.loc[y.index < t]

        try:
            model = sm.tsa.statespace.SARIMAX(
                y_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )
            
            # WARM START: Use previous parameters to start the optimizer
            if last_params is not None:
                fit = model.fit(start_params=last_params, disp=False)
            else:
                fit = model.fit(disp=False)
                
            last_params = fit.params
            preds.append(fit.forecast(steps=1).iloc[0])

        except Exception:
            # Fallback for convergence failures
            preds.append(y_train.iloc[-1] if len(y_train) > 0 else 0)

    return pd.Series(preds, index=test_index, name="forecast")

def evaluate_sarimax(
    y: pd.Series,
    train_end: str,
    order: Tuple[int, int, int],
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    strategy: Literal["rolling", "multistep"] = "rolling",
) -> dict:
    """
    One function to rule them all:
    - Handles ARIMA (if seasonal_order is None) and SARIMA.
    - Handles 'rolling' (1-step expanding) and 'multistep' (fixed origin).
    """
    # Use helper
    y_train, y_test = train_test_split_ts(y, train_end)

    # Normalization: ARIMA is just SARIMA with 0s
    s_order = seasonal_order if seasonal_order is not None else (0, 0, 0, 0)
    
    # 1. Fit on Train (Common to both strategies)
    model_train = sm.tsa.statespace.SARIMAX(
        y_train,
        order=order,
        seasonal_order=s_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit_train = model_train.fit(disp=False)
    
    # In-sample metrics
    train_pred = fit_train.get_prediction(dynamic=False).predicted_mean
    train_pred = train_pred.loc[y_train.index]  # Align
    train_metrics = mae_rmse(y_train, train_pred)

    # 2. Predict on Test (Strategy Specific)
    if strategy == "multistep":
        # Forecast h steps into the future at once
        test_pred = fit_train.get_forecast(steps=len(y_test)).predicted_mean
        test_pred.index = y_test.index
    else:
        # Rolling 1-step updates
        test_pred = rolling_1step_sarimax(
            y=y,
            test_index=y_test.index,
            order=order,
            seasonal_order=s_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

    test_metrics = mae_rmse(y_test, test_pred)

    return {
        "model_name": f"SARIMA_{order}_{s_order}_{strategy}",
        "fit": fit_train,  # Return the initial fit object
        "train_true": y_train, "train_pred": train_pred,
        "test_true": y_test,   "test_pred": test_pred,
        "train_metrics": train_metrics, "test_metrics": test_metrics,
    }

def evaluate_ets(
    y: pd.Series,
    train_end: str,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
    seasonal_periods: int = 12,
    strategy: Literal["rolling", "multistep"] = "rolling",
) -> dict:
    
    # Use helper
    y_train, y_test = train_test_split_ts(y, train_end)

    # 1. Fit on Train
    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    fit_train = model.fit(optimized=True)
    
    # Align fitted values to y_train index (and keep shape consistent)
    train_pred = fit_train.fittedvalues
    train_pred = train_pred.loc[y_train.index]
    train_metrics = mae_rmse(y_train, train_pred)

    # 2. Predict on Test
    if strategy == "multistep":
        test_pred = fit_train.forecast(steps=len(y_test))
        test_pred.index = y_test.index
    else:
        # ETS Rolling Loop: refit at each step
        preds = []
        for t in y_test.index:
            y_window = y.loc[y.index < t]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = ExponentialSmoothing(
                        y_window,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods,
                    )
                    f = m.fit(optimized=True)
                    preds.append(f.forecast(1).iloc[0])
            except Exception:
                preds.append(np.nan)
        
        test_pred = pd.Series(preds, index=y_test.index, name="forecast")

    test_metrics = mae_rmse(y_test, test_pred)

    return {
        "model_name": f"ETS_{trend}_{seasonal}_{strategy}",
        "fit": fit_train,
        "train_true": y_train, "train_pred": train_pred,
        "test_true": y_test,   "test_pred": test_pred,
        "train_metrics": train_metrics, "test_metrics": test_metrics,
    }

def evaluate_naive_persistence(
    y: pd.Series,
    train_end: str,
) -> dict:
    """
    Naive persistence model: ŷ_t = y_{t-1}

    Works for any frequency.
    Uses previous *observed* value as the forecast.
    """
    y_all = y.sort_index()
    y_train, y_test = train_test_split_ts(y_all, train_end)

    # One-step-ahead prediction via shift
    y_pred_full = y_all.shift(1)

    # Train predictions: aligned with y_train index
    train_pred = y_pred_full.loc[y_train.index]
    train_true = y_train.copy()

    # Drop first point where we have no lagged value (NaN)
    mask_train = train_pred.notna()
    train_pred = train_pred[mask_train]
    train_true = train_true[mask_train]

    train_metrics = mae_rmse(train_true, train_pred)

    # Test predictions from the same shifted series
    test_pred = y_pred_full.loc[y_test.index]
    test_true = y_test.copy()

    # Just in case, drop NaNs here too
    mask_test = test_pred.notna()
    test_pred = test_pred[mask_test]
    test_true = test_true[mask_test]

    test_metrics = mae_rmse(test_true, test_pred)

    return {
        "model_name": "NaivePersistence",
        "fit": None,  # for API consistency with other evaluators
        "train_true": train_true,
        "train_pred": train_pred,
        "test_true": test_true,
        "test_pred": test_pred,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
