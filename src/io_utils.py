# io_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def build_predictions_long_df(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    set_names: tuple[str, ...] = ("train", "test"),
) -> pd.DataFrame:
    """
    Convert the nested `results` dict into a long-format DataFrame
    with predictions and metadata for all models and splits.

    Expected `results` structure (from train_and_predict_global_models):
    {
      model_name: {
        "train": {
           "y_true": pd.Series,
           "y_pred": pd.Series,
           "meta":  pd.DataFrame,
        },
        "test": {
           "y_true": pd.Series,
           "y_pred": pd.Series,
           "meta":  pd.DataFrame,
        },
      },
      ...
    }

    Returns
    -------
    df : pd.DataFrame
        Columns:
          - "model"
          - "set"          (train/test)
          - "y_true"
          - "y_pred"
          - all columns from meta (e.g. State, Date, ...)
    """
    rows = []

    for model_name, model_res in results.items():
        for set_name in set_names:
            if set_name not in model_res:
                continue

            container = model_res[set_name]
            y_true = container["y_true"].reset_index(drop=True)
            y_pred = container["y_pred"].reset_index(drop=True)
            meta = container["meta"].reset_index(drop=True)

            df_tmp = meta.copy()
            df_tmp["y_true"] = y_true
            df_tmp["y_pred"] = y_pred
            df_tmp["model"] = model_name
            df_tmp["set"] = set_name

            rows.append(df_tmp)

    if not rows:
        raise ValueError("No predictions found in results for the requested sets.")

    df = pd.concat(rows, axis=0, ignore_index=True)
    # Optional: order columns
    cols = ["model", "set"] + [c for c in df.columns if c not in ("model", "set")]
    df = df[cols]

    return df


def export_panel_results(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: Dict[str, Dict[str, pd.DataFrame]] | None,
    output_dir: str | Path,
    experiment_name: str = "experiment",
    save_predictions: bool = True,
    save_metrics: bool = True,
) -> Dict[str, Path]:
    """
    Export predictions and metrics to CSV files.

    Parameters
    ----------
    results : dict
        Output of train_and_predict_global_models.
    metrics : dict or None
        Output of compute_panel_metrics, or None if you only want to
        export predictions.
        Expected structure:
        {
          "global": {
             "train": pd.DataFrame,
             "test":  pd.DataFrame,
          },
          "by_group": {
             "train": pd.DataFrame,
             "test":  pd.DataFrame,
          },
        }
    output_dir : str or Path
        Directory where CSV files will be saved. Created if it does not exist.
    experiment_name : str, default "experiment"
        Base name to prefix all CSV filenames.
    save_predictions : bool, default True
        Whether to save predictions_long CSV.
    save_metrics : bool, default True
        Whether to save metrics CSVs.

    Returns
    -------
    paths : dict
        Mapping from logical name to file Path, e.g.:
        {
          "predictions": Path(...),
          "metrics_global_train": Path(...),
          ...
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    # ---------------------------------------------------------
    # 1) Predictions
    # ---------------------------------------------------------
    if save_predictions:
        preds_df = build_predictions_long_df(results)
        preds_path = output_dir / f"{experiment_name}_predictions_long.csv"
        preds_df.to_csv(preds_path, index=False)
        paths["predictions"] = preds_path

    # ---------------------------------------------------------
    # 2) Metrics
    # ---------------------------------------------------------
    if save_metrics and metrics is not None:
        # Global metrics
        global_train = metrics["global"]["train"]
        global_test = metrics["global"]["test"]

        global_train_path = output_dir / f"{experiment_name}_metrics_global_train.csv"
        global_test_path = output_dir / f"{experiment_name}_metrics_global_test.csv"

        global_train.to_csv(global_train_path)
        global_test.to_csv(global_test_path)

        paths["metrics_global_train"] = global_train_path
        paths["metrics_global_test"] = global_test_path

        # By-group metrics (e.g., per State)
        by_group_train = metrics["by_group"]["train"]
        by_group_test = metrics["by_group"]["test"]

        by_group_train_path = output_dir / f"{experiment_name}_metrics_by_group_train.csv"
        by_group_test_path = output_dir / f"{experiment_name}_metrics_by_group_test.csv"

        by_group_train.to_csv(by_group_train_path)
        by_group_test.to_csv(by_group_test_path)

        paths["metrics_by_group_train"] = by_group_train_path
        paths["metrics_by_group_test"] = by_group_test_path

    return paths
