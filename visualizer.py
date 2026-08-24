"""
visualizer.py
-------------
All chart-drawing functions live here, kept separate from the app UI
and the analysis logic. Every function returns a Matplotlib `Figure`
object, which Streamlit can display directly using `st.pyplot(fig)`.

Chart types provided (as required by the project spec):
- Bar chart
- Line chart
- Pie chart
- Histogram
- Scatter plot
- Correlation heatmap
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# A calm, consistent style for all charts.
sns.set_theme(style="whitegrid")


def _new_figure(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def bar_chart(df: pd.DataFrame, category_col: str, value_col: str, top_n: int = 10):
    """
    Bar chart of `value_col` totals, grouped by `category_col`.
    Shows only the top_n categories so the chart stays readable.
    """
    grouped = (
        df.groupby(category_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = _new_figure()
    sns.barplot(x=grouped.values, y=grouped.index, ax=ax, orient="h",
                hue=grouped.index, palette="viridis", legend=False)
    ax.set_xlabel(value_col)
    ax.set_ylabel(category_col)
    ax.set_title(f"{value_col} by {category_col} (Top {top_n})")
    fig.tight_layout()
    return fig


def line_chart(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Line chart, typically used for a date column (x_col) against a
    numeric column (y_col) to show a trend over time.
    """
    plot_df = df[[x_col, y_col]].copy()

    # Try to parse the x-axis as a date; if it fails, keep it as-is.
    parsed_dates = pd.to_datetime(plot_df[x_col], errors="coerce")
    if parsed_dates.notnull().sum() > 0:
        plot_df[x_col] = parsed_dates
        plot_df = plot_df.dropna(subset=[x_col]).sort_values(x_col)
        plot_df = plot_df.groupby(x_col)[y_col].sum().reset_index()

    fig, ax = _new_figure()
    sns.lineplot(data=plot_df, x=x_col, y=y_col, ax=ax, marker="o")
    ax.set_title(f"{y_col} over {x_col}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def pie_chart(df: pd.DataFrame, category_col: str, top_n: int = 8):
    """
    Pie chart showing the share of each category (by row count).
    """
    counts = df[category_col].value_counts().head(top_n)

    fig, ax = _new_figure((6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Share of records by {category_col}")
    fig.tight_layout()
    return fig


def histogram(df: pd.DataFrame, column: str, bins: int = 20):
    """
    Histogram showing the distribution of a numeric column.
    """
    fig, ax = _new_figure()
    sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    fig.tight_layout()
    return fig


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None):
    """
    Scatter plot to visually inspect the relationship between two
    numeric columns. An optional categorical column can be used to
    color the points.
    """
    fig, ax = _new_figure()
    sns.scatterplot(
        data=df, x=x_col, y=y_col,
        hue=df[hue_col] if hue_col else None,
        ax=ax, alpha=0.8,
    )
    ax.set_title(f"{y_col} vs {x_col}")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """
    Heatmap of the correlation matrix across all numeric columns.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = _new_figure((7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, square=True)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig
