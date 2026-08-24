"""
app.py
------
This is the main Streamlit application file. It builds a simple,
beginner-friendly dashboard with the following sections:

1. Dataset upload
2. Dataset preview (rows, columns, dtypes, first 5 rows)
3. Data-cleaning summary
4. Exploratory Data Analysis (EDA) statistics
5. Data visualization
6. Natural-language question & answer

Run this app with:
    streamlit run app.py

All the "heavy lifting" (calculations) happens in data_analyzer.py and
visualizer.py. This file is only responsible for laying out the UI and
calling those functions -- it does not do any of its own math.
"""

import streamlit as st
import pandas as pd

import data_analyzer as da
import visualizer as viz


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Data Analysis Agent")
st.caption(
    "Upload a dataset, explore it, visualize it, and ask basic questions "
    "in plain English -- all calculations are done with real Pandas code, "
    "nothing is invented."
)


# ---------------------------------------------------------------------
# Session state: keep the dataframe around between reruns
# ---------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None


# =======================================================================
# SECTION 1: DATASET UPLOAD
# =======================================================================
st.header("1. Upload your dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

use_sample = st.checkbox("Or use the included sample dataset (data/sample_sales.csv)")

if uploaded_file is not None:
    df, error = da.load_dataset(uploaded_file)
    if error:
        st.error(error)
    else:
        st.session_state.df = df
        st.success(f"Loaded '{uploaded_file.name}' successfully.")
elif use_sample:
    try:
        df = pd.read_csv("data/sample_sales.csv")
        st.session_state.df = df
        st.success("Loaded sample dataset: data/sample_sales.csv")
    except Exception as e:
        st.error(f"Could not load sample dataset. Details: {e}")


# Stop here until we actually have a dataframe to work with.
if st.session_state.df is None:
    st.info("Upload a file (or check the sample-dataset box) to get started.")
    st.stop()

df = st.session_state.df


# =======================================================================
# SECTION 2: DATASET PREVIEW
# =======================================================================
st.header("2. Dataset preview")

overview = da.get_dataset_overview(df)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", overview["num_rows"])
col2.metric("Columns", overview["num_columns"])
col3.metric("Missing cells", int(df.isnull().sum().sum()))

with st.expander("Column names and data types"):
    dtype_df = pd.DataFrame(
        {"Column": overview["dtypes"].keys(), "Data Type": overview["dtypes"].values()}
    )
    st.dataframe(dtype_df, use_container_width=True)

st.subheader("First 5 rows")
st.dataframe(overview["head"], use_container_width=True)


# =======================================================================
# SECTION 3: DATA CLEANING SUMMARY
# =======================================================================
st.header("3. Data cleaning summary")

cleaning = da.get_cleaning_summary(df)

col1, col2 = st.columns(2)

with col1:
    st.write("**Missing values per column**")
    if cleaning["missing_values"]:
        st.dataframe(
            pd.DataFrame(
                {"Column": cleaning["missing_values"].keys(),
                 "Missing Count": cleaning["missing_values"].values()}
            ),
            use_container_width=True,
        )
    else:
        st.write("No missing values found. ✅")

with col2:
    st.write("**Duplicate rows**")
    st.write(f"{cleaning['duplicate_rows']} fully duplicated row(s) found.")
    st.write("**Numeric columns**:", ", ".join(cleaning["numeric_columns"]) or "None")
    st.write("**Categorical columns**:", ", ".join(cleaning["categorical_columns"]) or "None")
    if cleaning["constant_columns"]:
        st.warning(f"These columns have only one unique value: {', '.join(cleaning['constant_columns'])}")

st.write("---")
st.write("**Optional cleaning actions** (nothing is changed automatically):")
drop_dupes = st.checkbox("Remove fully duplicated rows")
fill_missing = st.checkbox("Fill missing numeric values with the column median")

if drop_dupes or fill_missing:
    df = da.clean_dataset(df, drop_duplicates=drop_dupes, fill_missing_numeric=fill_missing)
    st.success("Cleaning actions applied for this session. (Original upload is unchanged.)")
    st.dataframe(df.head(5), use_container_width=True)


# =======================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS
# =======================================================================
st.header("4. Exploratory Data Analysis (EDA)")

st.subheader("Overall summary statistics (numeric columns)")
summary_df = da.get_full_summary_statistics(df)
if not summary_df.empty:
    st.dataframe(summary_df, use_container_width=True)
else:
    st.write("No numeric columns found in this dataset.")

st.subheader("Explore a single column")
selected_column = st.selectbox("Choose a column", options=df.columns)

if selected_column:
    stats = da.get_column_statistics(df, selected_column)
    st.json(stats)


# =======================================================================
# SECTION 5: DATA VISUALIZATION
# =======================================================================
st.header("5. Data visualization")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

chart_type = st.selectbox(
    "Choose a chart type",
    ["Bar chart", "Line chart", "Pie chart", "Histogram", "Scatter plot", "Correlation heatmap"],
)

try:
    if chart_type == "Bar chart":
        cat_col = st.selectbox("Category column", categorical_cols, key="bar_cat")
        val_col = st.selectbox("Value column", numeric_cols, key="bar_val")
        if cat_col and val_col:
            fig = viz.bar_chart(df, cat_col, val_col)
            st.pyplot(fig)

    elif chart_type == "Line chart":
        x_col = st.selectbox("X-axis column (often a date)", df.columns, key="line_x")
        y_col = st.selectbox("Y-axis column (numeric)", numeric_cols, key="line_y")
        if x_col and y_col:
            fig = viz.line_chart(df, x_col, y_col)
            st.pyplot(fig)

    elif chart_type == "Pie chart":
        cat_col = st.selectbox("Category column", categorical_cols, key="pie_cat")
        if cat_col:
            fig = viz.pie_chart(df, cat_col)
            st.pyplot(fig)

    elif chart_type == "Histogram":
        num_col = st.selectbox("Numeric column", numeric_cols, key="hist_col")
        if num_col:
            fig = viz.histogram(df, num_col)
            st.pyplot(fig)

    elif chart_type == "Scatter plot":
        x_col = st.selectbox("X-axis (numeric)", numeric_cols, key="scatter_x")
        y_col = st.selectbox("Y-axis (numeric)", numeric_cols, key="scatter_y")
        hue_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols, key="scatter_hue")
        if x_col and y_col:
            fig = viz.scatter_plot(df, x_col, y_col, hue_col if hue_col != "None" else None)
            st.pyplot(fig)

    elif chart_type == "Correlation heatmap":
        if len(numeric_cols) >= 2:
            fig = viz.correlation_heatmap(df)
            st.pyplot(fig)
        else:
            st.write("Need at least 2 numeric columns for a correlation heatmap.")

except Exception as e:
    st.error(f"Could not draw this chart with the selected columns. Details: {e}")


# =======================================================================
# SECTION 6: NATURAL-LANGUAGE QUESTIONS
# =======================================================================
st.header("6. Ask a question about your data")

st.write(
    "Try questions like: *'What is the average Revenue?'*, "
    "*'Which Product has the highest Revenue?'*, "
    "*'What are the top 5 Products?'*, *'Show the Revenue trend.'*, "
    "*'Which column contains missing values?'*, "
    "*'What is the correlation between Price and Revenue?'*"
)

question = st.text_input("Your question")

if question:
    result = da.answer_question(df, question)

    st.markdown(f"**Answer:** {result['answer']}")

    if result["data"] is not None:
        st.write("**Supporting data:**")
        st.dataframe(result["data"], use_container_width=True)

    # Draw a supporting chart if the answer suggests one.
    hint = result["chart_hint"]
    try:
        if hint == "bar" and isinstance(result["data"], pd.Series):
            fig, ax = None, None
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            result["data"].head(10).plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Supporting chart")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig)

        elif hint == "heatmap" and len(numeric_cols) >= 2:
            fig = viz.correlation_heatmap(df)
            st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not draw a supporting chart. Details: {e}")


st.write("---")
st.caption(
    "Built as a beginner-friendly Data Science project: "
    "Data Collection → Data Cleaning → EDA → Visualization → Question Analysis → Insights."
)
