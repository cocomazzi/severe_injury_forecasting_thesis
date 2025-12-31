from __future__ import annotations

import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from pathlib import Path
import re

def preprocess_federal_data(
    df: pd.DataFrame,
    state_col: str = "State",
    federal_col: str = "FederalState",
    exclude_states: list = None,
) -> pd.DataFrame:
    """
    Preprocess OSHA dataset by keeping only federal states and optionally
    excluding custom states.

    Parameters
    ----------
    df : pd.DataFrame
        Input OSHA dataset.
    state_col : str, default="State"
        Column indicating the state name or code.
    federal_col : str, default="FederalState"
        Column indicating whether a state is federal (1) or not (0).
    exclude_states : list of str, optional
        List of states to exclude from the final dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset containing only federal states, excluding any
        specified states.
    """
    # Keep only federal states
    df_fed = df[df[federal_col] == 1].copy()

    # Exclude custom states if specified
    if exclude_states:
        df_fed = df_fed[~df_fed[state_col].isin(exclude_states)]

    print(f"✅ Remaining records: {len(df_fed)}")
    print(f"✅ Unique states: {df_fed[state_col].nunique()}")
    return df_fed

def fetch_bls_employment(series_ids, start_year, end_year):

    BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-type": "application/json"}
    data = {
        "seriesid": list(series_ids.values()),
        "startyear": str(start_year),
        "endyear": str(end_year),
    }

    response = requests.post(BLS_API_URL, json=data, headers=headers)
    json_data = response.json()

    records = []
    for series in json_data["Results"]["series"]:
        sid = series["seriesID"]
        state = [k for k, v in series_ids.items() if v == sid][0]
        for item in series["data"]:
            year = int(item["year"])
            if item["period"].startswith("M"):
                month = int(item["period"][1:])
                value = float(item["value"])
                date = f"{year}-{month:02d}-01"
                records.append({"State": state, "Date": date, "Employees": value})

    df = pd.DataFrame(records)
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    return df

#monthly ML panel

def make_monthly_ml_panel(
    panel_df: pd.DataFrame,
    exclude_lags: list[int] | None = None,
    exclude_diffs: list[int] | None = None,
    exclude_rolling: list[int] | None = None,
    exclude_seasonal: bool = False,
    include_covid_flag: bool = True,
) -> pd.DataFrame:
    """
    Prepare ML-ready monthly panel for supervised learning.

    Default: include calendar, seasonal, lags [1,3,6,12],
    diffs [1,12], rolling [3,6], EWMA over [3,6], and a CovidFlag
    for the core pandemic period (2020-03 to 2021-09).

    Calendar features (Year, Month, Quarter, DaysInMonth) are always included.
    Seasonal features (Month_sin, Month_cos) can be excluded.
    Rolling features include both mean and standard deviation of past values.
    EWMA features use past-only exponentially weighted means.

    IMPORTANT: This function drops rows with insufficient history. With default
    settings (lags up to 12, diffs up to 12, rolling up to 6), approximately
    the first 13 months for each State will be dropped. All lag, differencing,
    rolling, and EWMA features use only past information to prevent data leakage.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel dataset with ["State", "Date", "HospRisk", "AmpRisk"].
    exclude_lags : list[int] or None
        Lags to exclude, e.g. [12] to drop lag 12.
    exclude_diffs : list[int] or None
        Differencing periods to exclude, e.g. [1] to drop diff1.
    exclude_rolling : list[int] or None
        Rolling windows to exclude, e.g. [3] to drop roll3 (and its std/ewm).
    exclude_seasonal : bool
        If True, drop Month_sin and Month_cos.
    include_covid_flag : bool
        If True, add CovidFlag = 1 between 2020-03-01 and 2021-09-30, else 0.

    Returns
    -------
    pd.DataFrame
        ML-ready panel dataframe with all features and rows with insufficient
        history removed.
    """
    df = panel_df.copy()
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)

    # Optional safety:
    # df["Date"] = pd.to_datetime(df["Date"])

    # ------------------------------------------------
    # 0. Defaults (everything included unless excluded)
    # ------------------------------------------------
    default_lags = [1, 3, 6, 12]
    default_diffs = [1, 12]
    default_rolling = [3, 6]

    if exclude_lags is None:
        exclude_lags = []
    if exclude_diffs is None:
        exclude_diffs = []
    if exclude_rolling is None:
        exclude_rolling = []

    lags_to_use = [lag for lag in default_lags if lag not in exclude_lags]
    diffs_to_use = [d for d in default_diffs if d not in exclude_diffs]
    rolling_to_use = [w for w in default_rolling if w not in exclude_rolling]

    # -------------------------
    # 1. Calendar features (always included)
    # -------------------------
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["DaysInMonth"] = df["Date"].dt.days_in_month

    # -------------------------
    # 2. Seasonal features (optional)
    # -------------------------
    if not exclude_seasonal:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # -------------------------
    # 3. Covid flag (limited period)
    # -------------------------
    if include_covid_flag:
        covid_start = pd.Timestamp("2020-03-01")
        covid_end = pd.Timestamp("2021-09-30")
        df["CovidFlag"] = (
            (df["Date"] >= covid_start) & (df["Date"] <= covid_end)
        ).astype(int)

    # Pre-compute grouped series once
    grpH = df.groupby("State")["HospRisk"]
    grpA = df.groupby("State")["AmpRisk"]

    # -------------------------
    # 4. Lags (past values only)
    # -------------------------
    if lags_to_use:
        for lag in lags_to_use:
            df[f"HospRisk_lag{lag}"] = grpH.shift(lag)
            df[f"AmpRisk_lag{lag}"] = grpA.shift(lag)

    # -------------------------
    # 5. Differencing (past values only; no leakage)
    # -------------------------
    if diffs_to_use:
        for d in diffs_to_use:
            df[f"HospRisk_diff{d}"] = grpH.shift(1).diff(d)
            df[f"AmpRisk_diff{d}"] = grpA.shift(1).diff(d)

    # -------------------------
    # 6. Rolling windows + EWMA (past-only)
    #     - mean:  HospRisk_roll{w}, AmpRisk_roll{w}
    #     - std:   HospRisk_roll_std{w}, AmpRisk_roll_std{w}
    #     - ewma:  HospRisk_ewm{w}, AmpRisk_ewm{w}
    # -------------------------
    if rolling_to_use:
        grpH_shift = grpH.shift(1)
        grpA_shift = grpA.shift(1)

        for w in rolling_to_use:
            # Rolling mean/std (need full window)
            rollH = grpH_shift.rolling(w, min_periods=w)
            rollA = grpA_shift.rolling(w, min_periods=w)

            df[f"HospRisk_roll{w}"] = rollH.mean()
            df[f"AmpRisk_roll{w}"] = rollA.mean()

            df[f"HospRisk_roll_std{w}"] = rollH.std()
            df[f"AmpRisk_roll_std{w}"] = rollA.std()

            # EWMA (exponentially weighted moving average), span = w
            df[f"HospRisk_ewm{w}"] = grpH_shift.ewm(span=w, adjust=False).mean()
            df[f"AmpRisk_ewm{w}"] = grpA_shift.ewm(span=w, adjust=False).mean()

    # -------------------------
    # 7. Drop rows with missing values
    # -------------------------
    df = df.dropna().reset_index(drop=True)

    return df




def make_weekly_ml_panel(
    panel_df: pd.DataFrame,
    exclude_lags: list[int] | None = None,
    exclude_diffs: list[int] | None = None,
    exclude_rolling: list[int] | None = None,
    exclude_seasonal: bool = False,
    include_covid_flag: bool = True,
) -> pd.DataFrame:
    """
    Prepare ML-ready weekly panel for supervised learning.

    Default: include calendar, seasonal, lags [1,4,8,24,52],
    diffs [1,24], rolling [4,8], EWMA over [4,8], and a CovidFlag
    for the core pandemic period (2020-03 to 2021-09).

    Calendar features (Year, Month, Week, Quarter, DayOfWeek) are always
    included. Seasonal features (Week_sin, Week_cos) can be excluded.
    Rolling features include both mean and standard deviation of past values.
    EWMA features use past-only exponentially weighted means.

    IMPORTANT: This function drops rows with insufficient history. With default
    settings (lags up to 52, diffs up to 24, rolling up to 8), the first part
    of each State's time series is removed. All lag, differencing, rolling, and
    EWMA features use only past information to prevent data leakage.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Weekly panel with at least ["State", "Date", "HospRisk", "AmpRisk"].
    exclude_lags : list[int] or None
        Lags to exclude, e.g. [52] to drop lag 52.
    exclude_diffs : list[int] or None
        Differencing periods to exclude, e.g. [1] to drop diff1.
    exclude_rolling : list[int] or None
        Rolling windows (in weeks) to exclude, e.g. [4] to drop roll4.
    exclude_seasonal : bool
        If True, drop Week_sin and Week_cos.
    include_covid_flag : bool
        If True, add CovidFlag = 1 between 2020-03-01 and 2021-09-30, else 0.

    Returns
    -------
    pd.DataFrame
        ML-ready weekly panel dataframe with all features and rows with
        insufficient history removed.
    """
    df = panel_df.copy()
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)

    # Optional safety:
    # df["Date"] = pd.to_datetime(df["Date"])

    # ------------------------------------------------
    # 0. Defaults (everything included unless excluded)
    # ------------------------------------------------
    default_lags = [1, 4, 8, 24, 52]
    default_diffs = [1, 24]
    default_rolling = [4, 8]

    if exclude_lags is None:
        exclude_lags = []
    if exclude_diffs is None:
        exclude_diffs = []
    if exclude_rolling is None:
        exclude_rolling = []

    lags_to_use = [lag for lag in default_lags if lag not in exclude_lags]
    diffs_to_use = [d for d in default_diffs if d not in exclude_diffs]
    rolling_to_use = [w for w in default_rolling if w not in exclude_rolling]

    # -------------------------
    # 1. Calendar features (always included)
    # -------------------------
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.weekday  # Monday=0

    # -------------------------
    # 2. Seasonal features (optional, based on week-of-year)
    # -------------------------
    if not exclude_seasonal:
        df["Week_sin"] = np.sin(2 * np.pi * df["Week"] / 52)
        df["Week_cos"] = np.cos(2 * np.pi * df["Week"] / 52)

    # -------------------------
    # 3. Covid flag (limited period)
    # -------------------------
    if include_covid_flag:
        covid_start = pd.Timestamp("2020-03-01")
        covid_end = pd.Timestamp("2021-09-30")
        df["CovidFlag"] = (
            (df["Date"] >= covid_start) & (df["Date"] <= covid_end)
        ).astype(int)

    # Pre-compute grouped series once
    grpH = df.groupby("State")["HospRisk"]
    grpA = df.groupby("State")["AmpRisk"]

    # -------------------------
    # 4. Lags (past values only)
    # -------------------------
    if lags_to_use:
        for lag in lags_to_use:
            df[f"HospRisk_lag{lag}"] = grpH.shift(lag)
            df[f"AmpRisk_lag{lag}"] = grpA.shift(lag)

    # -------------------------
    # 5. Differencing (past values only; no leakage)
    # -------------------------
    if diffs_to_use:
        for d in diffs_to_use:
            df[f"HospRisk_diff{d}"] = grpH.shift(1).diff(d)
            df[f"AmpRisk_diff{d}"] = grpA.shift(1).diff(d)

    # -------------------------
    # 6. Rolling windows + EWMA (past-only)
    #     - mean:  HospRisk_roll{w}, AmpRisk_roll{w}
    #     - std:   HospRisk_roll_std{w}, AmpRisk_roll_std{w}
    #     - ewma:  HospRisk_ewm{w}, AmpRisk_ewm{w}
    # -------------------------
    if rolling_to_use:
        grpH_shift = grpH.shift(1)
        grpA_shift = grpA.shift(1)

        for w in rolling_to_use:
            # Rolling mean/std (need full window)
            rollH = grpH_shift.rolling(w, min_periods=w)
            rollA = grpA_shift.rolling(w, min_periods=w)

            df[f"HospRisk_roll{w}"] = rollH.mean()
            df[f"AmpRisk_roll{w}"] = rollA.mean()

            df[f"HospRisk_roll_std{w}"] = rollH.std()
            df[f"AmpRisk_roll_std{w}"] = rollA.std()

            # EWMA (span in weeks)
            df[f"HospRisk_ewm{w}"] = grpH_shift.ewm(span=w, adjust=False).mean()
            df[f"AmpRisk_ewm{w}"] = grpA_shift.ewm(span=w, adjust=False).mean()

    # -------------------------
    # 7. Drop rows with missing values
    # -------------------------
    df = df.dropna().reset_index(drop=True)

    return df

# src/feature_inspection.py


from typing import Dict, List

def inspect_feature_set(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Inspect and categorize the generated features in an ML panel dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Output of make_monthly_ml_panel or make_weekly_ml_panel.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping category → list of feature names.
    """
    columns = list(df.columns)

    categories = {
        "calendar": [],
        "seasonal": [],
        "lags": [],
        "diffs": [],
        "rolling_mean": [],
        "rolling_std": [],
        "ewma": [],
        "covid": [],
        "other": [],
    }

    for col in columns:
        # Skip structural columns
        if col in ("State", "Date"):
            continue

        # Calendar features (always included)
        if col in ("Year", "Month", "Quarter", "DaysInMonth"):
            categories["calendar"].append(col)
            continue

        # Seasonal
        if col in ("Month_sin", "Month_cos", "Week_sin", "Week_cos"):
            categories["seasonal"].append(col)
            continue

        # Covid flag
        if col == "CovidFlag":
            categories["covid"].append(col)
            continue

        # Lags: *_lagX
        if re.search(r"_lag\d+$", col):
            categories["lags"].append(col)
            continue

        # Differencing: *_diffX
        if re.search(r"_diff\d+$", col):
            categories["diffs"].append(col)
            continue

        # Rolling means: *_rollX
        if re.search(r"_roll\d+$", col):
            categories["rolling_mean"].append(col)
            continue

        # Rolling std: *_roll_stdX
        if re.search(r"_roll_std\d+$", col):
            categories["rolling_std"].append(col)
            continue

        # EWMA: *_ewmX
        if re.search(r"_ewm\d+$", col):
            categories["ewma"].append(col)
            continue

        # Everything else
        categories["other"].append(col)

    # Pretty printing
    print("\n==================== Feature Summary ====================")
    print(f"Total features (excluding State/Date): {len(columns) - 2}\n")
    for key, feats in categories.items():
        print(f"{key:15s}: {len(feats):3d}  {feats}")
    print("=========================================================\n")

    return categories


