# src/plotting.py
"""
Plotting utilities for the master_thesis project.

- Global ggplot-style theme
- Unified figure saving
- Generic plot helpers (time series, boxplots)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .config import FIGURES_DIR


# ---------------------------------------------------------------------------
# Colors & palettes
# ---------------------------------------------------------------------------

COLOR_PRIMARY: Final[str] = "#006094"    # blue
COLOR_SECONDARY: Final[str] = "#946110"  # gold / brown

DEFAULT_PALETTE: Final[list[str]] = [
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#CCB974",
    "#64B5CD",
]


# ---------------------------------------------------------------------------
# Global ggplot style
# ---------------------------------------------------------------------------

def set_plot_style(font_scale: float = 0.9) -> None:
    """
    Apply a global ggplot-style Matplotlib theme.
    Automatically applies to all Seaborn plots as well.
    """
    plt.style.use("ggplot")  # <-- MOST IMPORTANT: ggplot theme globally

    sns.set_palette(DEFAULT_PALETTE)
    sns.set_context("notebook", font_scale=font_scale)

    # Additional Matplotlib rcParams to refine the ggplot appearance
    mpl.rcParams.update(
        {
            # Figure
            "figure.figsize": (8, 5),
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,

            # Axes
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": True,
            "legend.frameon": False,

            # Remove top and right spines (ggplot-like)
            "axes.spines.top": False,
            "axes.spines.right": False,

            # Fonts
            "font.size": 11,
        }
    )


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    import re
    name = name.strip()
    name = re.sub(r"[^\w\s\-\.]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name or "figure"


def save_figure(
    fig: plt.Figure,
    filename: str,
    base_dir: Path | str | None = None,
    subfolder: str | None = None,
    ext: str = "png",
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> Path:
    """
    Unified figure saving utility.
    """
    if base_dir is None:
        out_dir = FIGURES_DIR
    else:
        out_dir = Path(base_dir)
        if not out_dir.is_absolute():
            out_dir = FIGURES_DIR / out_dir

    if subfolder:
        out_dir = out_dir / subfolder

    out_dir.mkdir(parents=True, exist_ok=True)

    fname = _slugify(filename)
    if not fname.lower().endswith(f".{ext.lower()}"):
        fname = f"{fname}.{ext}"

    outpath = out_dir / fname
    fig.savefig(outpath, dpi=dpi, bbox_inches=bbox_inches)
    return outpath


# ---------------------------------------------------------------------------
# Generic plotting helpers
# ---------------------------------------------------------------------------

def plot_timeseries(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Generic time-series line plot using ggplot style.
    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)

    if not legend and ax.legend_ is not None:
        ax.legend_.remove()

    return ax


def plot_metric_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Generic boxplot for RMSE/MAE/model comparisons.
    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)

    if hue and ax.legend_ is not None:
        ax.legend_.set_title(hue)
        ax.legend_.set_bbox_to_anchor((1.05, 1), loc="upper left")

    return ax



def plot_forecast_results(results: dict, title: str = None, figsize=(12, 6), burn_in: int = 1):
    """
    Plots training data, test data, and model predictions.
    Automatically detects if the strategy was 'rolling' or 'multistep' based on model_name.
    """
    # 1. Extract Data
    y_train = results["train_true"].copy()
    train_pred = results["train_pred"].copy()
    y_test = results["test_true"].copy()
    test_pred = results["test_pred"].copy()
    
    model_name = results.get("model_name", "Unknown Model")

    # 2. Handle Burn-in (Drop first N points of training fit which are usually unstable)
    if burn_in > 0 and len(y_train) > burn_in:
        y_train = y_train.iloc[burn_in:]
        train_pred = train_pred.loc[y_train.index]

    # 3. Determine Label based on Strategy
    if "multistep" in model_name.lower():
        test_label = "Test Forecast (Multistep)"
        # Multistep looks better as a solid line usually
        test_style = "-" 
    elif "rolling" in model_name.lower():
        test_label = "Test Forecast (Rolling 1-step)"
        # Rolling often looks better with a marker or slightly different style
        test_style = "-" 
    else:
        test_label = "Test Forecast"
        test_style = "-"

    # 4. Plotting
    fig, ax = plt.subplots(figsize=figsize)

    # Historical Data
    ax.plot(y_train.index, y_train.values, label="Train", color="black", alpha=0.6)
    ax.plot(y_test.index, y_test.values, label="Test", color="black", linestyle="--", alpha=0.8)

    # Model Predictions
    ax.plot(train_pred.index, train_pred.values, label="Train Fit", color="tab:blue", alpha=0.8)
    ax.plot(test_pred.index, test_pred.values, label=test_label, color="tab:red", linestyle=test_style, linewidth=2)

    # Vertical Line at Split
    if not y_test.empty:
        split_date = y_test.index[0]
        ax.axvline(split_date, color="grey", linestyle=":", alpha=0.6, label="Train/Test Split")

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Auto-format date on X-axis (prevents overlapping labels)
    fig.autofmt_xdate()

    if title is None:
        title = model_name
    ax.set_title(title)

    plt.show()

def plot_top_k_feature_importances(
    importances: pd.DataFrame,
    model_names: Sequence[str] | None = None,
    k: int = 20,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Plot horizontal bar charts of the top-k features for each model.

    Parameters
    ----------
    importances : pd.DataFrame
        Output of compute_feature_importances, with:
          - index  = feature names
          - columns = model names
    model_names : sequence of str or None, default None
        Models to plot. If None, all columns in `importances` are used.
    k : int, default 20
        Number of top features to display per model (by importance, descending).
    figsize : tuple(float, float) or None, default None
        Figure size passed to matplotlib. If None, a reasonable default is used
        based on the number of models.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : sequence of matplotlib.axes.Axes
        The axes for each model subplot (one per model).
    """
    if model_names is None:
        model_names = list(importances.columns)
    else:
        # Filter to those that are actually present
        model_names = [m for m in model_names if m in importances.columns]

    if not model_names:
        raise ValueError("No valid model names provided for plotting.")

    n_models = len(model_names)

    if figsize is None:
        # simple heuristic: width grows with number of models, height with k
        width = max(5.0, 4.0 * n_models)
        height = min(12.0, max(4.0, 0.35 * k))
        figsize = (width, height)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=figsize,
        squeeze=False,
    )
    axes = axes[0]  # since we created a single row

    for ax, model in zip(axes, model_names):
        # Get top-k features for this model
        s = importances[model].dropna().sort_values(ascending=False).head(k)
        # Reverse order so the most important is at the top in barh
        s = s.iloc[::-1]

        ax.barh(s.index, s.values)
        ax.set_title(model)
        ax.set_xlabel("Importance")

        # Improve layout: small font for y labels if many features
        if k > 20:
            ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    return fig, axes
