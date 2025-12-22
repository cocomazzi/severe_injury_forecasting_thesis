# aggregation.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import pandas as pd


def ensure_datetime(
    df: pd.DataFrame,
    date_col: str,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Return a copy of df where `date_col` is converted to pandas datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the date column in df.
    dayfirst : bool, default False
        Passed to pd.to_datetime for ambiguous formats (e.g. 01/02/2015).

    Notes
    -----
    - Tries to handle integer-like dates (e.g. 20150101) and ISO strings.
    - Does NOT modify the input df in place.
    """
    df_copy = df.copy()

    # If already datetime-like, just return
    if pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        return df_copy

    # Convert to string first to be robust to int64 columns like 20150101
    df_copy[date_col] = pd.to_datetime(
        df_copy[date_col].astype(str),
        dayfirst=dayfirst,
        errors="coerce",
    )

    if df_copy[date_col].isna().any():
        n_na = df_copy[date_col].isna().sum()
        raise ValueError(
            f"ensure_datetime: could not parse {n_na} dates in column '{date_col}'."
        )

    return df_copy


def aggregate_panel(
    df: pd.DataFrame,
    date_col: str,
    group_col: str = "State",
    target_cols: Sequence[str] = ("Hospitalized",),
    freq: str = "MS",
    agg: str = "sum",
    extra_aggs: Mapping[str, str] | None = None,
    complete_panel: bool = True,
) -> pd.DataFrame:
    """
    Aggregate raw event-level data to a panel with one row per (group, period).

    Parameters
    ----------
    df : pd.DataFrame
        Raw data.
    date_col : str
        Name of the date column in df.
    group_col : str, default "State"
        Column identifying cross-sectional units (e.g. states).
    target_cols : sequence of str, default ("Hospitalized",)
        Columns to aggregate as targets (e.g. hospitalization counts).
    freq : str, default "MS"
        Resampling frequency (e.g. "MS" for month start, "W-MON" for weekly).
    agg : str, default "sum"
        Aggregation function applied to all target_cols if extra_aggs is None.
    extra_aggs : mapping, optional
        Additional aggregations for other columns,
        e.g. {"Employment": "mean", "Amputation": "sum"}.
        If provided, it is merged with the default mapping for target_cols.
    complete_panel : bool, default True
        If True, ensures a balanced panel by:
          - Building the full Cartesian product of all groups and all periods
          - Filling missing combinations with zeros for target_cols.

    Returns
    -------
    panel_df : pd.DataFrame
        Aggregated panel with at least [group_col, date_col, *target_cols].

    Notes
    -----
    - The function does NOT modify the input df.
    - The output is sorted by group_col then date_col.
    """
    if not isinstance(target_cols, Sequence) or isinstance(target_cols, (str, bytes)):
        raise TypeError("target_cols must be a sequence of column names.")

    # Make sure date_col is datetime
    df_dt = ensure_datetime(df, date_col=date_col)

    # Build aggregation mapping
    agg_map: dict[str, str] = {col: agg for col in target_cols}
    if extra_aggs is not None:
        # User-provided aggs override defaults if keys clash
        agg_map.update(extra_aggs)

    # Group by (group, period) using pd.Grouper
    grouper = pd.Grouper(key=date_col, freq=freq)
    if isinstance(freq, str) and freq.startswith("W-"):
        grouper = pd.Grouper(key=date_col, freq=freq, label="right", closed="right")

    grouped = (
        df_dt
        .groupby([group_col, grouper], dropna=False)
        .agg(agg_map)
        .reset_index()
        .rename(columns={date_col: "Date"})
    )

    panel_df = grouped.sort_values([group_col, "Date"]).reset_index(drop=True)

    if not complete_panel:
        return panel_df

    # ------------------------------------------------------------------
    # Build a complete balanced panel (all groups x all periods)
    # ------------------------------------------------------------------
    all_groups = panel_df[group_col].dropna().unique()
    all_dates = pd.date_range(
        start=panel_df["Date"].min(),
        end=panel_df["Date"].max(),
        freq=freq,
    )

    full_index = pd.MultiIndex.from_product(
        [all_groups, all_dates],
        names=[group_col, "Date"],
    )

    panel_complete = (
        panel_df
        .set_index([group_col, "Date"])
        .reindex(full_index)
        .reset_index()
    )

    # Fill missing targets with zeros (counts), leave other columns as is
    for col in target_cols:
        if col in panel_complete.columns:
            panel_complete[col] = panel_complete[col].fillna(0)

    # For any extra aggregated columns (e.g., means), we leave NaN as-is
    # so downstream code can choose how to handle them.

    return panel_complete.sort_values([group_col, "Date"]).reset_index(drop=True)


def check_panel_balance(
    panel_df: pd.DataFrame,
    group_col: str = "State",
    date_col: str = "Date",
) -> dict:
    """
    Check whether the panel is balanced: each group has the same number of periods.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Aggregated panel, typically output of aggregate_panel.
    group_col : str, default "State"
        Column identifying cross-sectional units.
    date_col : str, default "Date"
        Column with time periods.

    Returns
    -------
    info : dict
        Dictionary with:
          - "is_balanced": bool
          - "n_groups": int
          - "n_periods": int
          - "rows_expected": int
          - "rows_actual": int
          - "periods_per_group": pd.Series
          - "missing_combinations": pd.DataFrame or None
                (group_col, date_col) pairs missing if not balanced.
    """
    if panel_df.empty:
        raise ValueError("check_panel_balance: panel_df is empty.")

    groups = panel_df[group_col].dropna().unique()
    periods = panel_df[date_col].dropna().unique()

    n_groups = len(groups)
    n_periods = len(periods)
    rows_expected = n_groups * n_periods
    rows_actual = len(panel_df)

    periods_per_group = (
        panel_df
        .dropna(subset=[date_col])
        .groupby(group_col)[date_col]
        .nunique()
        .sort_values()
    )

    is_balanced = (rows_expected == rows_actual) and (periods_per_group.nunique() == 1)

    missing_combinations = None
    if not is_balanced:
        full_index = pd.MultiIndex.from_product(
            [groups, periods],
            names=[group_col, date_col],
        )
        current_index = panel_df.set_index([group_col, date_col]).index
        missing_index = full_index.difference(current_index)

        if len(missing_index) > 0:
            missing_combinations = (
                missing_index.to_frame(index=False).sort_values([group_col, date_col])
            )

    return {
        "is_balanced": is_balanced,
        "n_groups": n_groups,
        "n_periods": n_periods,
        "rows_expected": rows_expected,
        "rows_actual": rows_actual,
        "periods_per_group": periods_per_group,
        "missing_combinations": missing_combinations,
    }
