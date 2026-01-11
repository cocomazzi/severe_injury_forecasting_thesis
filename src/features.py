from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

def compute_naics_mix(
        df: pd.DataFrame,
        naics_col: str = "Primary NAICS",
        date_col: str = "EventDate",
        state_col: str = "State",
        freq: str = "MS",
        top_k: int = 10,
        out_date_col: str = "Date",
    ) -> pd.DataFrame:
    """
    Compute NAICS2 mix features per (state, period) based on ALL events.

    For each (state, period) we compute, for the top-K most frequent NAICS2 codes:
        share_NAICS2_<code> = (# reports in NAICS2=<code>) / (total reports in that state-period)

    All NAICS2 codes not in the global top-K (or missing) are grouped into "Other":
        share_NAICS2_Other

    When a state-period has zero reports, all shares are 0.

    Parameters
    ----------
    df : pd.DataFrame
        Event-level data (one row per severe injury report).
    state_col : str, default "State"
        Column identifying the state.
    date_col : str, default "EventDate"
        Column with the event date (one per report).
    naics_col : str, default "Primary NAICS"
        Column with the NAICS code (string or numeric).
    freq : str, default "MS"
        Aggregation frequency for the panel, e.g. "MS" for monthly.
        This must match your panel aggregation.
    top_k : int, default 10
        Number of most frequent 2-digit NAICS codes to keep as separate
        categories; all others are grouped into "Other".
    out_date_col : str, default "Date"
        Name of the date column in the output, so it can be merged
        directly with your panel (which uses e.g. "Date").

    Returns
    -------
    naics_mix : pd.DataFrame
        Columns:
          [state_col, out_date_col, share_NAICS2_<code>, ..., share_NAICS2_Other]
        One row per (state, period).
    """
    if state_col not in df.columns:
        raise KeyError(f"state_col '{state_col}' not found in df.")
    if date_col not in df.columns:
        raise KeyError(f"date_col '{date_col}' not found in df.")
    if naics_col not in df.columns:
        raise KeyError(f"naics_col '{naics_col}' not found in df.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    
    data = df[[state_col, date_col, naics_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')

    if freq == 'MS':
        period = data[date_col].dt.to_period("M")
        agg_date = period.dt.to_timestamp()
    else:
        period = data[date_col].dt.to_period(freq)
        agg_date = period.dt.to_timestamp()
    
    data[out_date_col] = agg_date

    naics_raw = data[naics_col].astype(str).str.extract(r'(\d+)')[0]
    naics2 = naics_raw.str[:2]

    freq_naics2 = naics2.value_counts(dropna=True)
    top_codes = freq_naics2.index[:top_k].tolist()

    naics2_reduced = naics2.where(naics2.isin(top_codes), other="Other").fillna("Other")
    data["NAICS2_reduced"] = naics2_reduced

    counts = (
        data.groupby([state_col, out_date_col, "NAICS2_reduced"])
        .size()
        .reset_index(name="n_reports")
    )

    # 6) Total reports per (state, period)
    totals = (
        counts.groupby([state_col, out_date_col])["n_reports"]
        .sum()
        .reset_index(name="total_reports")
    )

    counts = counts.merge(totals, on=[state_col, out_date_col], how="left")

    # 7) Compute shares
    counts["share"] = counts["n_reports"] / counts["total_reports"].where(
        counts["total_reports"] > 0, other=pd.NA
    )

    # 8) Pivot to wide format: one row per (state, period)
    counts["col_name"] = "share_NAICS2_" + counts["NAICS2_reduced"].astype(str)

    mix = (
        counts.pivot_table(
            index=[state_col, out_date_col],
            columns="col_name",
            values="share",
            aggfunc="first",
        )
        .fillna(0.0)  # where total_reports was 0, shares become 0
        .reset_index()
    )

    mix.columns.name = None
    id_cols = [state_col, out_date_col]
    feature_cols = [c for c in mix.columns if c not in id_cols]
    mix = mix[id_cols + feature_cols]

    return mix

def compute_label_mix(
    df: pd.DataFrame,
    label_col: str,
    date_col: str = "EventDate",
    state_col: str = "State",
    freq: str = "MS",
    top_k: int = 10,
    out_date_col: str = "Date",
    prefix: str | None = None,
    end_date: str | pd.Timestamp | None = None,   # <-- NEW
) -> pd.DataFrame:
    """
    Compute share_<prefix>_<label> features per (state, period) based on ALL events.

    Top-K label set is learned globally; if end_date is provided, top-K is
    learned only from rows with date <= end_date (e.g., through 2023-12-31)
    to avoid regime breaks / leakage. Shares are still computed for the full df.

    Others + missing -> "Other".
    """
    if state_col not in df.columns:
        raise KeyError(f"state_col '{state_col}' not found in df.")
    if date_col not in df.columns:
        raise KeyError(f"date_col '{date_col}' not found in df.")
    if label_col not in df.columns:
        raise KeyError(f"label_col '{label_col}' not found in df.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if prefix is None:
        prefix = label_col

    data = df[[state_col, date_col, label_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    # Aggregate timestamp aligned with panel frequency
    if freq == "MS":
        agg_date = data[date_col].dt.to_period("M").dt.to_timestamp()
    else:
        agg_date = data[date_col].dt.to_period(freq).dt.to_timestamp()
    data[out_date_col] = agg_date

    # --- learn top-K labels on restricted data (if end_date provided)
    labels_all = data[label_col].astype("string").fillna("Other").str.strip()

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        labels_for_topk = labels_all[data[date_col].notna() & (data[date_col] <= end_ts)]
    else:
        labels_for_topk = labels_all

    top_labels = (
        labels_for_topk.value_counts(dropna=True)
        .head(top_k)
        .index
        .tolist()
    )

    # Apply fixed category set to ALL rows
    data["_label_reduced"] = labels_all.where(labels_all.isin(top_labels), other="Other")

    # Counts per (state, period, label)
    counts = (
        data.groupby([state_col, out_date_col, "_label_reduced"], dropna=False)
            .size()
            .reset_index(name="n_events")
    )

    # Totals per (state, period)
    totals = (
        counts.groupby([state_col, out_date_col], dropna=False)["n_events"]
              .sum()
              .reset_index(name="total_events")
    )

    counts = counts.merge(totals, on=[state_col, out_date_col], how="left")
    counts["share"] = counts["n_events"] / counts["total_events"].where(counts["total_events"] > 0, other=pd.NA)

    counts["col_name"] = "share_" + prefix + "_" + counts["_label_reduced"].astype(str)

    mix = (
        counts.pivot_table(
            index=[state_col, out_date_col],
            columns="col_name",
            values="share",
            aggfunc="first",
        )
        .fillna(0.0)
        .reset_index()
    )
    mix.columns.name = None
    return mix

FREQ_PRESETS = {
    "MS": {
        "lags": (1, 2, 3, 6, 12),
        "rolling_windows": (3, 6, 12),
        "ewm_spans": (3, 6, 12),
        "add_month_cycle": True,
        "add_week_cycle": False,
        "include_weekofyear": False,  # optional (you can drop weekofyear entirely)
    },
    "W-MON": {
        "lags": (1, 2, 4, 13, 26, 52),
        "rolling_windows": (4, 13, 26, 52),
        "ewm_spans": (4, 13, 26, 52),
        "add_month_cycle": False,  # keep clean; can set True if you ever want both
        "add_week_cycle": True,
        "include_weekofyear": False,  # we’ll not keep raw weekofyear
    },
}



def series_to_national_panel(
    y: pd.Series,
    target: str = "Hospitalized",
    group_col: str = "State",
    date_col: str = "Date",
    group_value: str = "NATIONAL",
) -> pd.DataFrame:
    """
    Convert a single national time series into the panel_df format expected by build_panel_features().
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a DatetimeIndex.")
    if y.name is None:
        y = y.rename(target)

    return pd.DataFrame(
        {
            group_col: group_value,
            date_col: y.index,
            target: y.to_numpy(dtype=float),
        }
    )

def build_panel_features(
    panel_df: pd.DataFrame,
    target: str = "Hospitalized",
    group_col: str = "State",
    date_col: str = "Date",
    freq: str = "MS",
    # time-based feature blocks
    add_calendar: bool = True,
    add_lags: bool = True,
    add_rolling: bool = True,
    add_ewm: bool = True,
    # autoregressive feature hyperparams
    lags: Sequence[int] | None = None,
    rolling_windows: Sequence[int] | None = None,
    ewm_spans: Sequence[int] | None = None,

    # categorical encodings
    state_encoding: Literal["none", "dummy"] = "none",
    naics_mix_cols: Sequence[str] | None = None,
    label_mix_cols: Sequence[str] | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build a feature matrix for global panel forecasting from an aggregated panel.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Aggregated panel with at least [group_col, date_col, target].
        May also contain exogenous features such as NAICS mix shares
        (e.g. share_NAICS2_23, ...) created by `compute_naics_mix`.
    target : str, default "Hospitalized"
        Name of the target column to be predicted.
    group_col : str, default "State"
        Name of the column identifying cross-sectional units.
    date_col : str, default "Date"
        Name of the date column. Must be convertible to datetime.
    freq : str, default "MS"
        Frequency string (e.g. "MS" for monthly, "W-MON" for weekly).
        Currently only used for sanity / potential future behavior.

    add_calendar, add_lags, add_rolling, add_ewm :
        Toggles for time-based feature blocks.

    lags, rolling_windows, ewm_spans :
        Hyperparameters for autoregressive features (in periods).

    state_encoding : {"none", "dummy"}, default "none"
        - "none": no explicit state encoding is added to X.
        - "dummy": one-hot encode the group_col and append dummies.

    naics_mix_cols : sequence of str or None, default None
        Column names in panel_df corresponding to NAICS mix shares
        (e.g. ["share_NAICS2_23", ...]). For each such column, a
        lag-1 feature is created per state:
            <col>_lag1 = value of <col> at t-1

    dropna : bool, default True
        If True, drops all rows that contain NaN in any constructed feature
        or in the target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with one row per observation.
    y : pd.Series
        Target series aligned with X.
    meta : pd.DataFrame
        Metadata aligned with X/y, containing at least [group_col, date_col].
    """
    # ------------------------------------------------------------------
    # Basic validation and setup
    # ------------------------------------------------------------------
    if target not in panel_df.columns:
        raise KeyError(f"Target column '{target}' not found in panel_df.")

    if group_col not in panel_df.columns:
        raise KeyError(f"group_col '{group_col}' not found in panel_df.")

    if date_col not in panel_df.columns:
        raise KeyError(f"date_col '{date_col}' not found in panel_df.")

    df = panel_df.copy()

    # Ensure datetime type
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    # Sort by group and time to ensure well-ordered operations
    df.sort_values([group_col, date_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Target and meta
    y = df[target].astype(float)
    meta = df[[group_col, date_col]].copy()

    # Feature container
    X = pd.DataFrame(index=df.index)

    preset = FREQ_PRESETS.get(freq, FREQ_PRESETS["MS"])

    if lags is None:
        lags = preset["lags"]
    if rolling_windows is None:
        rolling_windows = preset["rolling_windows"]
    if ewm_spans is None:
        ewm_spans = preset["ewm_spans"]

    # ------------------------------------------------------------------
    # 1. Calendar features
    # ------------------------------------------------------------------
    if add_calendar:
        dt = df[date_col].dt
        X["year"] = dt.year
        X["month"] = dt.month
        X["quarter"] = dt.quarter

        # Month cycle (monthly models)
        if preset.get("add_month_cycle", True):
            X["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
            X["month_cos"] = np.cos(2 * np.pi * dt.month / 12)

        # Week-of-year cycle (weekly models)
        if preset.get("add_week_cycle", False):
            w = dt.isocalendar().week.astype(int).clip(upper=52)
            X["week_sin"] = np.sin(2 * np.pi * w / 52)
            X["week_cos"] = np.cos(2 * np.pi * w / 52)


    # ------------------------------------------------------------------
    # 2. Lag features
    # ------------------------------------------------------------------
    if add_lags:
        for lag in lags:
            col_name = f"{target}_lag{lag}"
            X[col_name] = df.groupby(group_col)[target].shift(lag)

    # ------------------------------------------------------------------
    # 3. Rolling mean features (past window, excluding current)
    # ------------------------------------------------------------------
    if add_rolling:
        for window in rolling_windows:
            col_name = f"{target}_rollmean{window}"
            s = df.groupby(group_col)[target].shift(1)
            X[col_name] = (
                s.groupby(df[group_col])
                .rolling(window=window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # ------------------------------------------------------------------
    # 4. EWMA features (based on past values)
    # ------------------------------------------------------------------
    if add_ewm:
        for span in ewm_spans:
            col_name = f"{target}_ewm{span}"
            s = df.groupby(group_col)[target].shift(1)
            X[col_name] = (
            s.groupby(df[group_col])
            .ewm(span=span, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
            )

    # ------------------------------------------------------------------
    # 5. State dummy encoding
    # ------------------------------------------------------------------
    if state_encoding == "dummy":
        # One-hot encode the panel unit
        state_dummies = pd.get_dummies(
            df[group_col].astype("category"),
            prefix=group_col,
            drop_first=False,  # keep all; ridge will handle collinearity
        )
        # Align index and concat
        X = pd.concat([X, state_dummies], axis=1)
    elif state_encoding == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported state_encoding='{state_encoding}'. Use 'none' or 'dummy'."
        )

    # ------------------------------------------------------------------
    # 6. NAICS 2-digit dummy encoding
    # ------------------------------------------------------------------

    if naics_mix_cols is not None:
        for col in naics_mix_cols:
            if col not in df.columns:
                raise KeyError(
                    f"NAICS mix column '{col}' not found in panel_df. "
                    "Make sure to merge compute_naics_mix output first."
                )

            # past-only series (exclude current t)
            s = df.groupby(group_col)[col].shift(1)

            # leakage-safe rolling mean over the previous 12 periods
            X[f"{col}_rollmean12_lag1"] = (
                s.groupby(df[group_col])
                .rolling(window=12, min_periods=12)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # optional: also keep the plain lag-1
            # X[f"{col}_lag1"] = s


    # ------------------------------------------------------------------
    # 6b. Nature/Event mix shares 
    # ------------------------------------------------------------------
    if label_mix_cols is not None:
        for col in label_mix_cols:
            if col not in df.columns:
                raise KeyError(
                    f"Label mix column '{col}' not found in panel_df. "
                    "Make sure to merge compute_label_mix output first."
                )

            s = df.groupby(group_col)[col].shift(1)

            X[f"{col}_rollmean12_lag1"] = (
                s.groupby(df[group_col])
                .rolling(window=12, min_periods=12)
                .mean()
                .reset_index(level=0, drop=True)
            )


    # ------------------------------------------------------------------
    # 7. Final checks and NA handling
    # ------------------------------------------------------------------
    if X.shape[1] == 0:
        raise ValueError(
            "No features were created. Enable at least one of "
            "add_calendar, add_lags, add_rolling, add_ewm, state_encoding='dummy', "
            "or naics_encoding=True."
        )

    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        meta = meta.loc[mask].reset_index(drop=True)

    return X, y, meta

