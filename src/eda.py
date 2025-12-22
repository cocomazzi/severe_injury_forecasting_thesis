from __future__ import annotations

import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

from .plotting import save_figure


def inspect_dataset(df: pd.DataFrame, head=10) -> None:
    print("+++ Dataset Overview +++")
    display(df.head(head))
    print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns\n")

    print("General Info")
    print(df.info(), "\n")

    print('+++ Column Types +++')
    print(df.dtypes.value_counts(), "\n")

    print("+++ Numerical features +++")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(numerical_cols)


def find_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    missing_perc = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "missing values": missing,
        "percentage missing": missing_perc
    }).sort_values(by="percentage missing", ascending=False)
    return missing_df


def year_overview(df: pd.DataFrame, date_col: str = "Date", year: int = None):
    df["Year"] = df[date_col].dt.year

    if year is not None:
        df_year = df[df["Year"] == year]
    else:
        print(f"Please select a year")
        return None

    print(f"+++ General time-related stats for {year} +++")
    print(f"First date: {df_year[date_col].min()}")
    print(f"Last date: {df_year[date_col].max()}")
    print(f"Time span: {df_year[date_col].max() - df_year[date_col].min()} days")
    print(f"Unique dates: {df_year[date_col].nunique()}")

    # --- Weekday distribution ---
    weekday_counts = df_year[date_col].dt.day_name().value_counts()
    busiest_day = df_year[date_col].dt.date.value_counts().idxmax()
    busiest_count = df_year[date_col].dt.date.value_counts().max()

    print("\nWeekday distribution:")
    print(weekday_counts)

    print(f"\nBusiest day in {year}: {busiest_day} ({busiest_count} events)")
    
    print(f"+++ Stats for {year} +++")
    monthly = df_year.groupby(df_year[date_col].dt.month).size()
    daily = df_year.groupby(df_year[date_col].dt.date).size()
    
    print(f"Monthly reports: {monthly}")
    print(f"Daily counts: {daily}")

    return {
        "weekday": weekday_counts,
        "busiest_day" : (busiest_day, busiest_count),
        "monthly" : monthly, 
        "daily": daily}


def plot_year_overview(
    stats: dict,
    year: int,
    color1: str = "#006094",
    color2: str = "#946110",
    save: bool = True,
    save_dir: str | None = None,
):
    """
    Visualize the output of year_overview() with seaborn styling.
    If save=True, figures are written to the shared figure directory (or save_dir).
    """
    if stats is None:
        print("No stats to plot.")
        return
    
    # --- Weekday distribution ---
    fig = plt.figure(figsize=(8, 4))
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = stats["weekday"].reindex(order)
    weekday_palette = sns.color_palette("blend:" + color1 + "," + color2, n_colors=len(weekday))
    sns.barplot(
        x=weekday.index,
        y=weekday.values,
        hue=weekday.index,
        palette=weekday_palette,
        legend=False
    )
    plt.title(f"Events per Weekday ({year})", fontsize=14)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    if save:
        save_figure(fig, f"events_per_weekday_{year}.png", base_dir=save_dir)
    plt.show()
    
    # --- Monthly totals ---
    fig = plt.figure(figsize=(8, 4))
    monthly = stats["monthly"]
    monthly_palette = sns.color_palette("blend:" + color1 + "," + color2, n_colors=len(monthly))
    sns.barplot(
        x=monthly.index,
        y=monthly.values,
        hue=monthly.index,
        palette=monthly_palette,
        legend=False
    )
    plt.title(f"Monthly Event Counts ({year})", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Count")
    if save:
        save_figure(fig, f"monthly_event_counts_{year}.png", base_dir=save_dir)
    plt.show()
    
    # --- Daily counts ---
    fig = plt.figure(figsize=(12, 4))
    sns.lineplot(
        x=stats["daily"].index,
        y=stats["daily"].values,
        color=color1
    )
    plt.title(f"Daily Event Counts ({year})", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.tight_layout()
    if save:
        save_figure(fig, f"daily_event_counts_{year}.png", base_dir=save_dir)
    plt.show()


def plot_weekday_month_heatmap(
    df: pd.DataFrame,
    year: int,
    column: str = "Date",
    save: bool = True,
    save_dir: str | None = None,
):
    """
    Heatmap of events by weekday and month for a given year.
    Rows = months in calendar order, columns = weekdays (Mon–Sun).
    """
    df_year = df[df[column].dt.year == year]
    if df_year.empty:
        print(f"No data available for {year}")
        return
    
    pivot = df_year.pivot_table(
        index=df_year[column].dt.month,
        columns=df_year[column].dt.day_name(),
        values=column,
        aggfunc="count"
    ).fillna(0)

    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(columns=weekday_order)

    pivot.index = pivot.index.map(
        lambda m: pd.to_datetime(str(m), format="%m").strftime("%B")
    )

    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={'label': 'Event count'})
    plt.title(f"Events by Weekday and Month ({year})", fontsize=14)
    plt.ylabel("Month")
    plt.xlabel("Weekday")
    if save:
        save_figure(fig, f"weekday_month_heatmap_{year}.png", base_dir=save_dir)
    plt.show()


def compare_weekday_delta(
    df: pd.DataFrame,
    year1: int,
    year2: int,
    column: str = "Date",
    save: bool = True,
    save_dir: str | None = None,
):
    """
    Show the difference in weekday event counts between two years (year2 - year1).
    """
    df1 = df[df[column].dt.year == year1]
    df2 = df[df[column].dt.year == year2]

    if df1.empty or df2.empty:
        print("No data available for one of the years")
        return

    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    counts1 = df1[column].dt.day_name().value_counts().reindex(order, fill_value=0)
    counts2 = df2[column].dt.day_name().value_counts().reindex(order, fill_value=0)

    delta = counts2 - counts1
    delta_df = pd.DataFrame({"Weekday": order, "Delta": delta.values})

    fig = plt.figure(figsize=(10, 6))
    sns.barplot(
        data=delta_df,
        x="Weekday", y="Delta",
        palette=["#946110" if v < 0 else "#006094" for v in delta_df["Delta"]]
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Change in Weekday Event Counts: {year2} – {year1}", fontsize=14)
    plt.ylabel("Δ Events")
    plt.xticks(rotation=45)
    if save:
        save_figure(fig, f"weekday_delta_{year1}_to_{year2}.png", base_dir=save_dir)
    plt.show()

    return delta_df


def year_injuries_overview(
    df: pd.DataFrame, 
    date_col: str = "Date", 
    year: int = None,
    metrics: list = ["Hospitalized", "Amputation"]
):
    df["Year"] = df[date_col].dt.year

    if year is not None:
        df_year = df[df["Year"] == year]
    else:
        print("Please select a year")
        return None
    
    print(f"+++ Injury stats for {year} +++")
    print(f"First date: {df_year[date_col].min()}")
    print(f"Last date: {df_year[date_col].max()}")
    print(f"Time span: {df_year[date_col].max() - df_year[date_col].min()}")
    print(f"Unique dates: {df_year[date_col].nunique()}")

    weekday_injuries = df_year.groupby(df_year[date_col].dt.day_name())[metrics].sum()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_injuries = weekday_injuries.reindex(order)

    busiest_day = (
        df_year.groupby(df_year[date_col].dt.date)[metrics]
        .sum()
        .sum(axis=1)
        .idxmax()
    )
    busiest_count = (
        df_year.groupby(df_year[date_col].dt.date)[metrics]
        .sum()
        .sum(axis=1)
        .max()
    )

    monthly_injuries = df_year.groupby(df_year[date_col].dt.month)[metrics].sum()
    daily_injuries = df_year.groupby(df_year[date_col].dt.date)[metrics].sum()

    return {
        "weekday_injuries": weekday_injuries,
        "busiest_day": (busiest_day, busiest_count),
        "monthly_injuries": monthly_injuries,
        "daily_injuries": daily_injuries,
    }


def plot_year_injuries_overview(
    stats: dict,
    year: int,
    color1: str = "#006094",
    color2: str = "#946110",
    save: bool = True,
    save_dir: str | None = None,
):
    """
    Visualize the output of injury-focused year_overview() with seaborn styling.
    If save=True, figures are written to the shared figure directory (or save_dir).
    """
    if stats is None:
        print("No stats to plot.")
        return
    
    injury_palette = {"hospitalized": color1, "amputation": color2}
    
    # --- Weekday injuries ---
    fig = plt.figure(figsize=(8, 4))
    weekday = stats["weekday_injuries"].reset_index()
    id_col = weekday.columns[0]
    weekday = weekday.melt(id_vars=id_col, var_name="Injury", value_name="Count")
    weekday.rename(columns={id_col: "Weekday"}, inplace=True)
    weekday["Injury"] = weekday["Injury"].str.lower()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.barplot(
        data=weekday,
        x="Weekday", y="Count", hue="Injury",
        palette=injury_palette,
        order=order
    )
    plt.title(f"Injuries per Weekday ({year})", fontsize=14)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Injury type")
    if save:
        save_figure(fig, f"injuries_per_weekday_{year}.png", base_dir=save_dir)
    plt.show()
    
    # --- Monthly injuries ---
    fig = plt.figure(figsize=(8, 4))
    monthly = stats["monthly_injuries"].reset_index()
    id_col = monthly.columns[0]
    monthly = monthly.melt(id_vars=id_col, var_name="Injury", value_name="Count")
    monthly.rename(columns={id_col: "Month"}, inplace=True)
    monthly["Injury"] = monthly["Injury"].str.lower()

    sns.barplot(
        data=monthly,
        x="Month", y="Count", hue="Injury",
        palette=injury_palette
    )
    plt.title(f"Monthly Injuries ({year})", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.legend(title="Injury type")
    if save:
        save_figure(fig, f"monthly_injuries_{year}.png", base_dir=save_dir)
    plt.show()
    
    # --- Daily injuries ---
    fig = plt.figure(figsize=(12, 4))
    daily = stats["daily_injuries"].reset_index()
    id_col = daily.columns[0]
    daily = daily.melt(id_vars=id_col, var_name="Injury", value_name="Count")
    daily.rename(columns={id_col: "Date"}, inplace=True)
    daily["Injury"] = daily["Injury"].str.lower()

    sns.lineplot(
        data=daily,
        x="Date", y="Count", hue="Injury",
        palette=injury_palette
    )
    plt.title(f"Daily Injuries ({year})", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(title="Injury type")
    plt.tight_layout()
    if save:
        save_figure(fig, f"daily_injuries_{year}.png", base_dir=save_dir)
    plt.show()


def plot_category_distribution(
    series: pd.Series,
    color1: str = "#006094",
    color2: str = "#946110",
    title: str = "",
    save: bool = True,
    save_dir: str | None = None,
):
    """
    Plot a horizontal barplot of category distribution and optionally save it.

    Parameters
    ----------
    series : pd.Series
        Categorical series to visualize.
    color1, color2 : str
        Hex codes for the color palette.
    title : str, optional
        Plot title.
    save : bool, default=True
        Whether to save the figure.
    save_dir : str or Path, optional
        Directory where the figure will be saved.
    """

    distribution = series.value_counts().sort_values(ascending=False)

    fig = plt.figure(figsize=(8, max(4, len(distribution) * 0.3)))
    sns.barplot(
        x=distribution.values,
        y=distribution.index,
        hue=distribution.index,
        palette=sns.color_palette("blend:" + color1 + "," + color2, n_colors=len(distribution)),
        legend=False,
        orient="h"
    )
    plt.title(title or f"Distribution of {series.name or 'Category'}", fontsize=14)
    plt.xlabel("Count")
    plt.ylabel(series.name if series.name else "Category")
    plt.tight_layout()

    if save:
        filename = (title or series.name or "category_distribution").lower().replace(" ", "_")
        save_figure(fig, filename, base_dir=save_dir)

    plt.show()
