"""
data_analyzer.py
-----------------
This module contains all the "brain" logic of the AI Data Analysis Agent.

It is kept completely separate from the Streamlit UI (app.py) so that:
1. The logic can be tested independently of the UI.
2. A beginner can read this file top-to-bottom and understand exactly
   how every number shown on the dashboard was calculated.

IMPORTANT DESIGN RULE (as required by the project spec):
- Every number returned by this file comes directly from Pandas
  calculations on the real dataset.
- Nothing is hard-coded and nothing is invented.
"""

import pandas as pd
import numpy as np


# =====================================================================
# 1. LOADING DATA
# =====================================================================

def load_dataset(uploaded_file):
    """
    Load a CSV or Excel file into a Pandas DataFrame.

    Parameters
    ----------
    uploaded_file : a file-like object coming from Streamlit's
                    st.file_uploader (has a .name attribute)

    Returns
    -------
    (df, error_message)
        df            : pandas.DataFrame if successful, else None
        error_message : str if something went wrong, else None

    We return a tuple instead of raising an exception so that the
    Streamlit app can show a friendly message instead of crashing.
    """
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Please upload a .csv or .xlsx/.xls file."

        if df.empty:
            return None, "The uploaded file is empty. Please upload a dataset with data in it."

        return df, None

    except Exception as e:
        # Catching all exceptions here keeps the app from crashing on a
        # malformed file, and shows the real reason to the user instead.
        return None, f"Could not read the file. Details: {e}"


# =====================================================================
# 2. BASIC DATASET OVERVIEW
# =====================================================================

def get_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Returns simple facts about the dataset: shape, column names,
    data types, and a preview of the first 5 rows.
    """
    overview = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(5),
    }
    return overview


# =====================================================================
# 3. DATA CLEANING SUMMARY
# =====================================================================

def get_cleaning_summary(df: pd.DataFrame) -> dict:
    """
    Inspects the dataset for common data-quality issues WITHOUT
    modifying or deleting anything. The project requirement is that
    we must inform the user instead of silently deleting data.

    Returns a dictionary with:
    - missing_values   : per-column count of missing values
    - duplicate_rows   : number of fully duplicated rows
    - numeric_columns   : list of columns detected as numeric
    - categorical_columns : list of columns detected as categorical/text
    - constant_columns  : columns that only contain one unique value
                          (often not useful for analysis)
    """
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].to_dict()

    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    constant_columns = [
        col for col in df.columns if df[col].nunique(dropna=True) <= 1
    ]

    return {
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "constant_columns": constant_columns,
    }


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = False,
                   fill_missing_numeric: bool = False) -> pd.DataFrame:
    """
    Performs OPTIONAL cleaning actions, only when the user explicitly
    asks for them (e.g. by ticking a checkbox in the UI). This keeps
    us aligned with the rule: "Do not delete important data
    automatically without informing the user."

    Parameters
    ----------
    drop_duplicates      : if True, removes fully duplicated rows
    fill_missing_numeric  : if True, fills missing numeric values with
                            the column median (a safe, common default)
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if fill_missing_numeric:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_value)

    return cleaned_df


# =====================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

def get_column_statistics(df: pd.DataFrame, column: str) -> dict:
    """
    Returns descriptive statistics for a single column.
    Works differently depending on whether the column is numeric
    or categorical/text.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        stats = {
            "type": "numeric",
            "count": int(series.count()),
            "missing": int(series.isnull().sum()),
            "mean": round(series.mean(), 2) if series.count() else None,
            "median": round(series.median(), 2) if series.count() else None,
            "min": series.min(),
            "max": series.max(),
            "std_dev": round(series.std(), 2) if series.count() else None,
            "unique_values": int(series.nunique()),
        }
    else:
        value_counts = series.value_counts().head(10)
        stats = {
            "type": "categorical",
            "count": int(series.count()),
            "missing": int(series.isnull().sum()),
            "unique_values": int(series.nunique()),
            "most_common": value_counts.to_dict(),
        }

    return stats


def get_full_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns df.describe() for all numeric columns as a DataFrame,
    ready to display in a Streamlit table.
    """
    return df.describe(include=[np.number]).round(2)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the correlation matrix for all numeric columns.
    Used to power the correlation heatmap and correlation-related
    natural-language questions.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr().round(2)


# =====================================================================
# 5. NATURAL-LANGUAGE QUESTION HANDLING (RULE-BASED, NOT A FULL AI AGENT)
# =====================================================================
#
# As per the project requirement, the first version implements common
# questions using plain Python/Pandas logic (keyword matching) instead
# of a complex autonomous agent. If an LLM is plugged in later, it
# should only be used to (a) map the free-text question to one of
# these handlers and (b) phrase the final explanation in plain
# English -- the NUMBERS themselves always come from Pandas.

def _find_column(df: pd.DataFrame, keyword: str):
    """Helper: find a column whose name contains the given keyword
    (case-insensitive). Returns the column name or None."""
    keyword = keyword.lower()
    for col in df.columns:
        if keyword in col.lower():
            return col
    return None


def answer_question(df: pd.DataFrame, question: str) -> dict:
    """
    Very small rule-based "question answering" engine.

    Returns a dictionary shaped like:
        {
            "answer": "<plain language answer>",
            "data": <supporting pandas object, e.g. a Series/DataFrame, or None>,
            "chart_hint": "<suggested chart type, or None>"
        }

    This function purposefully covers only the common question
    patterns listed in the project brief. Add more `elif` branches
    here to extend the agent later.
    """
    q = question.lower().strip()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ---- 1. "average / mean of X" --------------------------------------
    if "average" in q or "mean" in q:
        target_col = None
        for col in numeric_cols:
            if col.lower() in q:
                target_col = col
                break
        if target_col is None and numeric_cols:
            target_col = numeric_cols[0]  # fallback: first numeric column

        if target_col:
            avg = round(df[target_col].mean(), 2)
            return {
                "answer": f"The average {target_col} is {avg}.",
                "data": None,
                "chart_hint": None,
            }

    # ---- 2. "highest / maximum / top" -----------------------------------
    if "highest" in q or "maximum" in q or "max " in q or q.startswith("max"):
        # Try to find "which <category> has highest <numeric>" pattern
        target_numeric = None
        for col in numeric_cols:
            if col.lower() in q:
                target_numeric = col
                break
        if target_numeric is None and numeric_cols:
            target_numeric = numeric_cols[0]

        group_col = None
        for col in categorical_cols:
            if col.lower() in q:
                group_col = col
                break

        if target_numeric and group_col:
            grouped = df.groupby(group_col)[target_numeric].sum().sort_values(ascending=False)
            top_item = grouped.index[0]
            top_value = round(grouped.iloc[0], 2)
            return {
                "answer": f"'{top_item}' has the highest total {target_numeric} "
                          f"({top_value}).",
                "data": grouped,
                "chart_hint": "bar",
            }
        elif target_numeric:
            idx = df[target_numeric].idxmax()
            row = df.loc[idx]
            return {
                "answer": f"The highest {target_numeric} value is {row[target_numeric]}.",
                "data": row,
                "chart_hint": None,
            }

    # ---- 3. "top N" -------------------------------------------------------
    if "top 5" in q or "top5" in q or "top " in q:
        target_numeric = numeric_cols[0] if numeric_cols else None
        group_col = categorical_cols[0] if categorical_cols else None
        for col in numeric_cols:
            if col.lower() in q:
                target_numeric = col
        for col in categorical_cols:
            if col.lower() in q:
                group_col = col

        if target_numeric and group_col:
            grouped = df.groupby(group_col)[target_numeric].sum().sort_values(ascending=False).head(5)
            return {
                "answer": f"Here are the top 5 {group_col} values by total {target_numeric}.",
                "data": grouped,
                "chart_hint": "bar",
            }

    # ---- 4. "most records / most common category" ------------------------
    if "most records" in q or "most common" in q or "count" in q:
        group_col = None
        for col in categorical_cols:
            if col.lower() in q:
                group_col = col
                break
        if group_col is None and categorical_cols:
            group_col = categorical_cols[0]

        if group_col:
            counts = df[group_col].value_counts()
            top_category = counts.index[0]
            top_count = int(counts.iloc[0])
            return {
                "answer": f"'{top_category}' has the most records ({top_count} rows) "
                          f"in the '{group_col}' column.",
                "data": counts,
                "chart_hint": "bar",
            }

    # ---- 5. "trend" -------------------------------------------------------
    if "trend" in q:
        date_col = None
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break
        target_numeric = numeric_cols[0] if numeric_cols else None
        for col in numeric_cols:
            if col.lower() in q:
                target_numeric = col

        if date_col and target_numeric:
            trend_df = df[[date_col, target_numeric]].copy()
            trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
            trend_df = trend_df.dropna(subset=[date_col]).sort_values(date_col)
            trend_series = trend_df.groupby(date_col)[target_numeric].sum()
            return {
                "answer": f"Here is the {target_numeric} trend over time, based on the '{date_col}' column.",
                "data": trend_series,
                "chart_hint": "line",
            }

    # ---- 6. "missing values" ----------------------------------------------
    if "missing" in q:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return {
                "answer": "No columns contain missing values in this dataset.",
                "data": None,
                "chart_hint": None,
            }
        cols_list = ", ".join(missing.index.tolist())
        return {
            "answer": f"These columns contain missing values: {cols_list}.",
            "data": missing,
            "chart_hint": "bar",
        }

    # ---- 7. "correlation between X and Y" ----------------------------------
    if "correlation" in q:
        mentioned = [col for col in numeric_cols if col.lower() in q]
        if len(mentioned) >= 2:
            col_a, col_b = mentioned[0], mentioned[1]
            corr_value = round(df[col_a].corr(df[col_b]), 3)
            return {
                "answer": f"The correlation between {col_a} and {col_b} is {corr_value}.",
                "data": None,
                "chart_hint": "scatter",
            }
        else:
            corr_matrix = get_correlation_matrix(df)
            return {
                "answer": "Here is the full correlation matrix for all numeric columns.",
                "data": corr_matrix,
                "chart_hint": "heatmap",
            }

    # ---- Fallback -----------------------------------------------------------
    return {
        "answer": (
            "I couldn't match this question to a known pattern yet. "
            "Try asking about average, highest, top 5, most common, trend, "
            "missing values, or correlation -- and mention a column name if possible."
        ),
        "data": None,
        "chart_hint": None,
    }
