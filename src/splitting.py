from __future__ import annotations

import warnings
from typing import Any
from datetime import datetime
import pandas as pd


def _to_datetime(value: Any) -> pd.Timestamp:
    """Helper to convert input to a standardized Pandas Timestamp."""
    if isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value)


def temporal_panel_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    date_col: str = "Date",
    train_end: str | datetime | pd.Timestamp | None = None,
    test_size: int | None = None,
    test_start: str | datetime | pd.Timestamp | None = None,
) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    """
    Temporal train / test split for panel data.
    
    Logic:
    1. TRAIN: All periods <= train_end.
    2. TEST: The next 'test_size' available periods immediately after train_end.
    3. DROP: Any periods remaining after the test window are excluded.
    """
    
    # ---------------------------------------------------------
    # 1. Input Validation
    # ---------------------------------------------------------
    if not (len(X) == len(y) == len(meta)):
        raise ValueError("X, y, and meta must have the same number of rows.")

    # Ensure indices align (crucial for .loc masking later)
    if not (X.index.equals(meta.index) and y.index.equals(meta.index)):
        # Optional: You could choose to reset_index here, but raising is safer
        # to prevent accidental misalignment of data.
        raise ValueError("Indices of X, y, and meta do not align.")

    if date_col not in meta.columns:
        raise KeyError(f"date_col '{date_col}' not found in meta DataFrame.")

    if train_end is None:
        raise ValueError("train_end must be provided (e.g., '2023-12-01').")
    
    if test_size is None or test_size <= 0:
        raise ValueError("test_size must be a positive integer.")

    # ---------------------------------------------------------
    # 2. Date Processing
    # ---------------------------------------------------------
    # Convert meta column to datetime objects
    # We keep the index of 'dates' matching 'meta' for creating the masks later
    dates = pd.to_datetime(meta[date_col], errors="raise")
    
    # Get unique, sorted periods to determine split boundaries
    unique_dates = pd.Index(sorted(dates.unique()))
    train_end_ts = _to_datetime(train_end)

    if train_end_ts < unique_dates[0] or train_end_ts > unique_dates[-1]:
        raise ValueError(f"train_end {train_end_ts} is outside the range of available dates.")

    # ---------------------------------------------------------
    # 3. Define Train Scope
    # ---------------------------------------------------------
    train_dates = unique_dates[unique_dates <= train_end_ts]
    
    if len(train_dates) == 0:
        raise ValueError("No data found for the training set (check train_end).")

    # ---------------------------------------------------------
    # 4. Define Test Scope
    # ---------------------------------------------------------
    # Logic: Start immediately after train_end, or at specific test_start
    if test_start is not None:
        test_start_ts = _to_datetime(test_start)
        if test_start_ts <= train_end_ts:
            raise ValueError(f"test_start {test_start_ts} must be strictly after train_end {train_end_ts}.")
    else:
        # Default: The first observed period strictly after train_end
        future_dates = unique_dates[unique_dates > train_end_ts]
        if len(future_dates) == 0:
            raise ValueError("No periods available after train_end to create a test set.")
        test_start_ts = future_dates[0]

    # Gather all possible dates starting from test_start
    candidate_test_dates = unique_dates[unique_dates >= test_start_ts]

    if len(candidate_test_dates) == 0:
         raise ValueError(f"No dates found starting from {test_start_ts}.")

    # CHECK: Warn if we don't have enough data to fill 'test_size'
    if len(candidate_test_dates) < test_size:
        warnings.warn(
            f"Requested test_size={test_size}, but only {len(candidate_test_dates)} "
            "periods exist after train_end. The test set will be truncated."
        )

    # SELECT: Take exactly the first N periods. 
    # Any period after this slice is implicitly dropped.
    test_dates = candidate_test_dates[:test_size]

    # ---------------------------------------------------------
    # 5. Masking and Splitting
    # ---------------------------------------------------------
    # Create boolean masks. Since 'dates' has the same index as X/y/meta, this aligns.
    train_mask = dates.isin(train_dates)
    test_mask = dates.isin(test_dates)

    # Sanity check for overlap
    if (train_mask & test_mask).any():
        raise RuntimeError("CRITICAL: Overlap detected between train and test sets.")

    def _subset(mask: pd.Series) -> dict[str, Any]:
        return {
            "X": X.loc[mask].reset_index(drop=True),
            "y": y.loc[mask].reset_index(drop=True),
            "meta": meta.loc[mask].reset_index(drop=True),
        }

    return {
        "train": _subset(train_mask),
        "test": _subset(test_mask),
    }