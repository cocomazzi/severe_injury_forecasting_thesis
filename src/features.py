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
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    rolling_windows: Sequence[int] = (3, 6, 12),
    ewm_spans: Sequence[int] = (3, 6, 12),
    # categorical encodings
    state_encoding: Literal["none", "dummy"] = "none",
    naics_mix_cols: Sequence[str] | None = None,
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

    # ------------------------------------------------------------------
    # 1. Calendar features
    # ------------------------------------------------------------------
    if add_calendar:
        dt = df[date_col].dt
        X["year"] = dt.year
        X["month"] = dt.month
        X["quarter"] = dt.quarter
        X["weekofyear"] = dt.isocalendar().week.astype(int)
        X["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
        X["month_cos"] = np.cos(2 * np.pi * dt.month / 12)

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
            X[col_name] = df.groupby(group_col)[target].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
            )

    # ------------------------------------------------------------------
    # 4. EWMA features (based on past values)
    # ------------------------------------------------------------------
    if add_ewm:
        for span in ewm_spans:
            col_name = f"{target}_ewm{span}"
            X[col_name] = df.groupby(group_col)[target].transform(
                lambda s, sp=span: s.ewm(span=sp, adjust=False).mean().shift(1)
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
            lag_col = f"{col}_lag1"
            X[lag_col] = df.groupby(group_col)[col].shift(1)
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

