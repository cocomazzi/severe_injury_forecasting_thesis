from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Iterable, Optional, Union
import calendar
from scipy.stats import shapiro, anderson, skew, kurtosis
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss


# -------------------------------------------------------------------
# Helpers to prepare series
# -------------------------------------------------------------------

def prepare_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: Optional[str] = None,
    agg: str = "sum",
) -> pd.Series:
    """
    Prepare a univariate time series from a DataFrame.

    - Ensures datetime index
    - Sorts by date
    - Optionally resamples to a given frequency (e.g. 'M', 'MS', 'W')

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    date_col : str
        Column with datetime information.
    value_col : str
        Column with the target values (e.g. Hospitalized counts).
    freq : str, optional
        Pandas offset alias for resampling (e.g. 'M', 'W').
    agg : str, default='sum"
        Aggregation for resampling ('sum', 'mean', etc.).

    Returns
    -------
    pd.Series
        Time series indexed by datetime.
    """
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts.set_index(date_col, inplace=True)
    ts.sort_index(inplace=True)

    if freq is not None:
        ts = getattr(ts.resample(freq), agg)()

    # Return as Series
    return ts[value_col]


def plot_monthly_seasonality(
    df: pd.DataFrame,
    date_col: str = "Date",
    value_col: Optional[str] = None,
    agg: str = "sum",
    years: Optional[Union[int, Iterable[int]]] = None,
) -> pd.DataFrame:
    """
    Plot seasonal pattern at a monthly resolution, optionally for
    a single year or multiple years.

    If value_col is None, counts the number of rows (events) per month.
    Otherwise, aggregates the given value_col with `agg` (e.g. 'sum', 'mean').

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a datetime column.
    date_col : str, default="Date"
        Name of the datetime column.
    value_col : str or None, default=None
        Column to aggregate. If None, row counts are used.
        For counts of Hospitalized / Amputation, pass the column and use agg='sum'.
    agg : str, default="sum"
        Aggregation function ('count', 'sum', 'mean', ...).
    years : int or iterable of int or None, default=None
        - None  → use all available years
        - int   → plot only that year
        - iterable[int] → plot those years on the same figure

    Returns
    -------
    monthly : pd.DataFrame
        DataFrame with columns ['Year', 'Month', 'value'] used for plotting.
    """
    df_ = df.copy()
    df_[date_col] = pd.to_datetime(df_[date_col])

    df_["Year"] = df_[date_col].dt.year
    df_["Month"] = df_[date_col].dt.month

    # --------- Aggregate ---------
    if value_col is None:
        monthly = (
            df_.groupby(["Year", "Month"])
               .size()
               .reset_index(name="value")
        )
        y_label = "Number of events"
    else:
        monthly = (
            df_.groupby(["Year", "Month"])[value_col]
               .agg(agg)
               .reset_index(name="value")
        )
        y_label = f"{agg.capitalize()} of {value_col}"

    # --------- Filter by year(s) if requested ---------
    if years is not None:
        if isinstance(years, (int, np.integer)):
            years = [int(years)]
        else:
            years = list(years)
        monthly = monthly[monthly["Year"].isin(years)]

    # If filtering removed everything, warn early
    if monthly.empty:
        raise ValueError("No data available for the specified year(s).")

    # --------- Plot ---------
    fig, ax = plt.subplots(figsize=(10, 6))

    for year, group in monthly.groupby("Year"):
        group_sorted = group.sort_values("Month")
        ax.plot(
            group_sorted["Month"],
            group_sorted["value"],
            marker="o",
            alpha=0.8,
            label=str(year),
        )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(calendar.month_abbr[1:13])
    ax.set_xlabel("Month")
    ax.set_ylabel(y_label)

    # Dynamic title depending on years
    if years is None:
        title_years = "all years"
    else:
        title_years = ", ".join(str(y) for y in sorted(set(years)))
    ax.set_title(f"Seasonal pattern at monthly resolution ({title_years})", fontsize=14)

    ax.grid(alpha=0.3)
    ax.legend(title="Year", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()

    return monthly


def _resolve_freq(agg_level: str) -> str:
    """
    Map a human-friendly aggregation level to a pandas frequency string.

    Parameters
    ----------
    agg_level : str
        Either 'monthly' or 'weekly' (case-insensitive).

    Returns
    -------
    str
        Pandas offset alias (e.g. 'M' for month-end, 'W' for weekly).
    """
    level = agg_level.lower()
    if level in {"monthly", "month", "m"}:
        return "M"   # month-end
    if level in {"weekly", "week", "w"}:
        return "W"   # weekly, default anchor (Sun)
    raise ValueError(f"Unsupported agg_level: {agg_level!r}. Use 'monthly' or 'weekly'.")


def plot_target_timeseries(
    df: pd.DataFrame,
    date_col: str = "EventDate",
    targets: Iterable[str] = ("Hospitalized", "Amputation"),
    agg_level: str = "monthly",
    agg: str = "sum",
    per_state: bool = False,
    state_col: str = "State",
) -> None:
    """
    Plot aggregated target time series at monthly or weekly level (counts).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing a datetime column and numeric targets.
    date_col : str, default="EventDate"
        Name of the datetime column.
    targets : iterable of str, default=("Hospitalized", "Amputation")
        Target variable names to analyze (count variables).
    agg_level : str, default="monthly"
        Aggregation level: 'monthly' or 'weekly'.
    agg : str, default="sum"
        Aggregation function for resampling ('sum', 'mean', etc.).
        For counts, 'sum' is typically appropriate.
    per_state : bool, default=False
        If False, aggregate over all states.
        If True, aggregate per state and plot one line per state.
    state_col : str, default="State"
        Column name for the state identifier when per_state=True.
    """
    freq = _resolve_freq(agg_level)

    df_ = df.copy()
    df_[date_col] = pd.to_datetime(df_[date_col])
    df_.set_index(date_col, inplace=True)

    # ------------------------ global aggregation ------------------------
    if not per_state:
        agg_df = (
            df_[list(targets)]
            .resample(freq)
            .agg(agg)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        agg_df.plot(ax=ax)
        ax.set_title(f"{agg_level.capitalize()} {agg} targets over time", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # ------------------------ per-state aggregation ------------------------
    if state_col not in df_.columns:
        raise KeyError(f"state_col '{state_col}' not found in DataFrame.")

    agg_df = (
        df_
        .groupby(state_col)[list(targets)]
        .resample(freq)
        .agg(agg)
        .reset_index()
    )

    # For per_state, make one figure per target, lines = states
    for target in targets:
        fig, ax = plt.subplots(figsize=(12, 6))

        for state, group in agg_df.groupby(state_col):
            ax.plot(group[date_col], group[target], label=state, alpha=0.6)

        ax.set_title(
            f"{agg_level.capitalize()} {agg} {target} per {state_col}",
            fontsize=16,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{target} (count)")
        ax.grid(alpha=0.3)

        # With 50 states the legend can get big; feel free to comment this out.
        ax.legend(title=state_col, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()


def _normality_tests(series: pd.Series) -> dict:
    """
    Run normality tests on a 1D numeric series (e.g. monthly counts).
    Returns a dictionary with results.
    """
    s = series.dropna()

    # Shapiro-Wilk
    sh_w, sh_p = shapiro(s)

    # Anderson-Darling
    ad_result = anderson(s)
    ad_stat = ad_result.statistic
    ad_crit = ad_result.critical_values
    ad_sig = ad_result.significance_level

    return {
        "shapiro_W": sh_w,
        "shapiro_p": sh_p,
        "anderson_stat": ad_stat,
        "anderson_crit": list(zip(ad_sig, ad_crit)),
        "skewness": skew(s),
        "kurtosis": kurtosis(s),
    }


import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_histograms(
    df: pd.DataFrame,
    date_col: str = "EventDate",
    targets: tuple[str, ...] = ("Hospitalized", "Amputation"),
    agg_level: str | None = "monthly",   # if None -> assume already aggregated
    agg: str = "sum",
    per_state: bool = False,
    state_col: str = "State",
    bins: int = 30,
) -> None:
    """
    Plot histograms (with KDE) for target variables.

    Supports two modes:
    1) Event-level mode (default): resamples by time (and optionally by state).
       - Provide agg_level ('monthly' or 'weekly'), agg ('sum', 'mean', etc.)
    2) Aggregated-panel mode: if agg_level is None, assumes df already has one row
       per (state, period) or per period; no resampling is performed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    date_col : str
        Name of datetime column.
    targets : tuple[str, ...]
        One or more target columns to plot (e.g., ("HospRisk",)).
    agg_level : {"monthly","weekly"} or None
        If None, skip resampling and plot the given columns as-is.
    agg : str
        Aggregation function if resampling is used.
    per_state : bool
        If True and resampling is used, aggregates within each state then pools values.
        If agg_level is None, per_state just means "pool all rows" (no effect).
    state_col : str
        State column name.
    bins : int
        Histogram bins.
    """
    # ----------------------------
    # Validate inputs
    # ----------------------------
    if isinstance(targets, (list, set)):
        targets = tuple(targets)

    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns: {missing_targets}")

    df_ = df.copy()

    # ----------------------------
    # Build agg_df
    # ----------------------------
    if agg_level is None:
        # Already aggregated: use as-is
        agg_df = df_[list(targets)].copy()

    else:
        freq = _resolve_freq(agg_level)

        if date_col not in df_.columns:
            raise KeyError(f"date_col '{date_col}' not found in DataFrame.")

        df_[date_col] = pd.to_datetime(df_[date_col])
        df_.set_index(date_col, inplace=True)

        if not per_state:
            agg_df = df_[list(targets)].resample(freq).agg(agg)
        else:
            if state_col not in df_.columns:
                raise KeyError(f"state_col '{state_col}' not found in DataFrame.")
            agg_df = (
                df_
                .groupby(state_col)[list(targets)]
                .resample(freq)
                .agg(agg)
                .reset_index()
            )

    # ----------------------------
    # Plot layout (any #targets)
    # ----------------------------
    n = len(targets)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = [axes] if n == 1 else axes.flatten()

    for i, target in enumerate(targets):
        ax = axes[i]

        series = agg_df[target].dropna()

        sns.histplot(series, bins=bins, kde=True, ax=ax)
        ax.set_title(f"Distribution of {target}", fontsize=13)

        xlabel = f"{target}"
        if agg_level is not None:
            xlabel += f" ({agg_level} aggregated, {agg})"
        ax.set_xlabel(xlabel)

        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

        # Normality tests (optional: keep for HospRisk, but note it's usually non-normal)
        results = _normality_tests(series)

        print("\n" + "=" * 60)
        print(f"Normality tests for: {target}")
        print("=" * 60)
        print(f"Shapiro-Wilk W = {results['shapiro_W']:.4f}, p = {results['shapiro_p']:.4f}")
        print("→ Reject normality (p < 0.05)" if results["shapiro_p"] < 0.05 else "→ Fail to reject normality (p ≥ 0.05)")
        print("\nAnderson-Darling statistic =", f"{results['anderson_stat']:.4f}")
        print("Critical values (sig_level %, critical_value):")
        for sig, crit in results["anderson_crit"]:
            print(f"  {sig}%  →  {crit:.4f}")
        print("\nSkewness  =", f"{results['skewness']:.4f}")
        print("Kurtosis =", f"{results['kurtosis']:.4f}")
        print("=" * 60)

    # Turn off unused axes if n is odd
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()



# -------------------------------------------------------------------
# Core diagnostics for a single series
# -------------------------------------------------------------------

def plot_ts_with_rolling(
    series: pd.Series,
    window: int = 12,
    title: Optional[str] = None,
) -> None:
    """
    Plot time series with rolling mean and rolling std.

    Parameters
    ----------
    series : pd.Series
        Time series with a DatetimeIndex.
    window : int
        Rolling window size (in number of observations).
    title : str, optional
        Plot title.
    """
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(series.index, series.values, label="Series")
    ax[0].plot(roll_mean.index, roll_mean.values, linestyle="--", label=f"Rolling mean ({window})")
    ax[0].set_ylabel(series.name or "value")
    ax[0].legend()
    if title:
        ax[0].set_title(title)

    ax[1].plot(roll_std.index, roll_std.values, label=f"Rolling std ({window})")
    ax[1].set_ylabel("Rolling std")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def stl_decompose(
    df: pd.DataFrame,
    date_col: str = "Date",
    value_col: str = "Hospitalized",
    agg_level: str = "monthly",   # "monthly" or "weekly"
    agg: str = "sum",
    period: int | None = None,    # e.g. 12 for monthly data
    plot: bool = True,
):
    """
    Perform STL decomposition at the global level, typically on
    aggregated counts (e.g. monthly Hospitalized).

    Parameters
    ----------
    df : DataFrame
        Full dataset.
    date_col : str
        Date column.
    value_col : str
        Target column to decompose (e.g. 'Hospitalized' counts).
    agg_level : str
        'monthly' or 'weekly'.
    agg : str
        Resampling aggregation: 'sum', 'mean', etc.
        For counts, 'sum' is appropriate.
    period : int or None
        Seasonal period (12 = monthly seasonality; 52 = weekly).
        If None, inferred automatically from agg_level.
    plot : bool
        If True, plots the decomposition.

    Returns
    -------
    result : STL object
    """
    freq = _resolve_freq(agg_level)

    # Resample via helper
    ts = prepare_series(
        df=df,
        date_col=date_col,
        value_col=value_col,
        freq=freq,
        agg=agg,
    )

    # Set default period
    if period is None:
        if agg_level == "monthly":
            period = 12
        elif agg_level == "weekly":
            period = 52
        else:
            raise ValueError("Cannot infer period; specify it manually.")

    stl = STL(ts, period=period, robust=True)
    result = stl.fit()

    if plot:
        fig = result.plot()
        fig.set_size_inches(10, 8)
        fig.suptitle(f"STL Decomposition of {value_col} ({agg_level})", fontsize=14)
        plt.tight_layout()
        plt.show()

    return result


# -------------------------------------------------------------------
# Statistical testing for Stationarity
# -------------------------------------------------------------------

def stationarity_tests(
    series: pd.Series,
    alpha: float = 0.05,
    kpss_regression: str = "c",   # "c" = level-stationary, "ct" = trend-stationary
) -> pd.DataFrame:
    """
    Run ADF and KPSS tests on a series and return a tidy summary.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex (NaNs are dropped),
        e.g. monthly counts of Hospitalized.
    alpha : float
        Significance level for 'stationary' vs 'non-stationary' label.
    kpss_regression : {"c", "ct"}
        Type of stationarity for KPSS test.

    Returns
    -------
    pd.DataFrame with index ["ADF", "KPSS"] and columns:
        ["statistic", "pvalue", "lags", "nobs", "critical_values", "verdict"]
    """
    s = series.dropna()

    results = []

    # ADF
    try:
        adf_res = adfuller(s, autolag="AIC")
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adf_res
        adf_verdict = "stationary" if adf_p < alpha else "non-stationary"
        results.append({
            "test": "ADF",
            "statistic": adf_stat,
            "pvalue": adf_p,
            "lags": adf_lags,
            "nobs": adf_nobs,
            "critical_values": adf_crit,
            "verdict": adf_verdict,
        })
    except Exception as e:
        results.append({
            "test": "ADF",
            "statistic": np.nan,
            "pvalue": np.nan,
            "lags": np.nan,
            "nobs": len(s),
            "critical_values": {},
            "verdict": f"error: {e}",
        })

    # KPSS
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression=kpss_regression, nlags="auto")
        kpss_verdict = "stationary" if kpss_p > alpha else "non-stationary"
        results.append({
            "test": "KPSS",
            "statistic": kpss_stat,
            "pvalue": kpss_p,
            "lags": kpss_lags,
            "nobs": len(s),
            "critical_values": kpss_crit,
            "verdict": kpss_verdict,
        })
    except Exception as e:
        results.append({
            "test": "KPSS",
            "statistic": np.nan,
            "pvalue": np.nan,
            "lags": np.nan,
            "nobs": len(s),
            "critical_values": {},
            "verdict": f"error: {e}",
        })

    return pd.DataFrame(results).set_index("test")


def apply_differencing(
    series: pd.Series,
    d: int = 0,
    D: int = 0,
    season_period: int | None = None,
) -> pd.Series:
    """
    Apply non-seasonal (d) and seasonal (D) differencing to a series.

    Parameters
    ----------
    series : pd.Series
        Input time series (e.g. monthly counts).
    d : int
        Order of non-seasonal differencing.
    D : int
        Order of seasonal differencing.
    season_period : int, optional
        Seasonal period, e.g. 12 for monthly series with yearly seasonality.

    Returns
    -------
    pd.Series
        Differenced series (with NaNs at the beginning).
    """
    s = series.copy()

    # Seasonal differencing first (common convention)
    if D > 0 and season_period is not None:
        for _ in range(D):
            s = s.diff(season_period)

    # Non-seasonal differencing
    if d > 0:
        for _ in range(d):
            s = s.diff()

    return s


def suggest_differencing(
    series: pd.Series,
    season_period: int | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Suggest (d, D) differencing orders based on ADF + KPSS.

    We test:
      - (d=0, D=0)
      - (d=1, D=0)
      - (d=0, D=1) if season_period is not None
      - (d=1, D=1) if season_period is not None

    Classification:
      'stationary' if ADF p < alpha and KPSS p > alpha.

    Returns
    -------
    dict with keys:
      - 'recommended': {'d': int, 'D': int, 'season_period': int | None}
      - 'variants': dict of all tested variants and their test results
    """
    variants = {}

    def eval_variant(name: str, d: int, D: int):
        s_diff = apply_differencing(series, d=d, D=D, season_period=season_period)
        tests = stationarity_tests(s_diff, alpha=alpha)
        adf_p = tests.loc["ADF", "pvalue"]
        kpss_p = tests.loc["KPSS", "pvalue"]
        stationary = (
            pd.notna(adf_p) and pd.notna(kpss_p)
            and adf_p < alpha and kpss_p > alpha
        )
        variants[name] = {
            "d": d,
            "D": D,
            "adf_pvalue": adf_p,
            "kpss_pvalue": kpss_p,
            "stationary": stationary,
        }

    # Always test original and plain diff
    eval_variant("original", d=0, D=0)
    eval_variant("diff1", d=1, D=0)

    # If seasonal period is provided, also test seasonal variants
    if season_period is not None:
        eval_variant("seasonal_diff", d=0, D=1)
        eval_variant("diff1_seasonal", d=1, D=1)

    # Choose recommended: smallest total differencing that is stationary
    candidates = [
        v for v in variants.values() if v["stationary"]
    ]
    if candidates:
        # sort by total differencing order (d + D), then by D (prefer non-seasonal first)
        candidates_sorted = sorted(
            candidates,
            key=lambda x: (x["d"] + x["D"], x["D"])
        )
        best = candidates_sorted[0]
    else:
        # Fallback: if nothing is clearly stationary, recommend some conservative choice
        if season_period is not None:
            best = {"d": 1, "D": 1}
        else:
            best = {"d": 1, "D": 0}
        # merge with dummy p-values / flags
        best |= {
            "adf_pvalue": np.nan,
            "kpss_pvalue": np.nan,
            "stationary": False,
        }

    recommended = {
        "d": best["d"],
        "D": best["D"],
        "season_period": season_period,
    }

    return {
        "recommended": recommended,
        "variants": variants,
    }


def suggest_differencing_by_state(
    df: pd.DataFrame,
    state_col: str,
    date_col: str,
    value_col: str,
    freq: str | None = None,
    season_period: int | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run suggest_differencing() for each state and return a summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    state_col : str
        Column with state identifiers.
    date_col : str
        Datetime column.
    value_col : str
        Target column (e.g. monthly counts).
    freq : str or None
        Frequency for resampling (e.g. 'M' for monthly).
    season_period : int or None
        Seasonal period (e.g. 12 for monthly data).
    alpha : float
        Significance level for stationarity tests.

    Returns
    -------
    pd.DataFrame
        Table with columns ['State', 'd', 'D', 'season_period'].
    """
    rows = []
    for state in sorted(df[state_col].dropna().unique()):
        df_state = df[df[state_col] == state]
        series = prepare_series(df_state, date_col=date_col, value_col=value_col, freq=freq)
        res = suggest_differencing(series, season_period=season_period, alpha=alpha)
        rec = res["recommended"]
        rows.append({
            "State": state,
            "d": rec["d"],
            "D": rec["D"],
            "season_period": rec["season_period"],
        })
    return pd.DataFrame(rows)
