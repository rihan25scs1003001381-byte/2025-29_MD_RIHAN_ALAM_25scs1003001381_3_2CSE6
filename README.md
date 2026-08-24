# AI Data Analysis Agent (Basic Version)

A beginner-friendly AI Data Analysis Agent built with **Python, Pandas, and Streamlit**.
Upload a CSV/Excel file, get an automatic data-cleaning summary, explore statistics,
generate charts, and ask basic questions about your data in plain English.

This is the **first, simple version** of the project — it uses rule-based
Python/Pandas logic to answer questions, not a complex autonomous AI agent.
It's designed to match a B.Tech CSE (AIML) / Data Science internship level.

---

## What it does

| Step | Feature |
|------|---------|
| 1 | Upload a CSV or Excel dataset |
| 2 | See rows, columns, data types, and a preview |
| 3 | Automatic data-cleaning summary (missing values, duplicates, column types) |
| 4 | Exploratory Data Analysis — mean, median, min, max, std dev, value counts |
| 5 | Charts — bar, line, pie, histogram, scatter, correlation heatmap |
| 6 | Ask natural-language questions (e.g. "Which product has the highest sales?") |

**Important rule followed throughout the code:** every number shown comes from
an actual Pandas calculation on your uploaded data. Nothing is hard-coded or invented.

---

## Project structure

```
ai-data-analysis-agent/
│
├── app.py                     # Streamlit dashboard (the UI)
├── data_analyzer.py           # Loading, cleaning, EDA, and question-answering logic
├── visualizer.py               # All chart-drawing functions (Matplotlib + Seaborn)
├── requirements.txt           # Python dependencies
├── README.md                  # You are here
│
├── data/
│   └── sample_sales.csv       # Sample dataset for testing (includes some
│                               # missing values and a duplicate row on purpose,
│                               # so you can see the cleaning summary in action)
│
└── notebooks/
    └── data_analysis.ipynb    # Jupyter notebook for initial, exploratory analysis
```

Each Python file has a single, clear responsibility:
- **`data_analyzer.py`** = the "brain" (all calculations)
- **`visualizer.py`** = the "artist" (all charts)
- **`app.py`** = the "face" (Streamlit layout, calls the other two files)

This separation makes the code easier to read, test, and extend later.

---

## How to run it

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
This will open the dashboard in your browser (usually at `http://localhost:8501`).

### 4. Try it out
- Check **"Or use the included sample dataset"** to try the app instantly, or
- Upload your own `.csv` / `.xlsx` file.

---

## Example: sample_sales.csv

The included sample dataset has these columns:

```
Date, Product, Category, Quantity, Price, Revenue
```

Try asking the app:
- *"What is the average Revenue?"*
- *"Which Product has the highest Revenue?"*
- *"What are the top 5 Products?"*
- *"Show the Revenue trend."*
- *"Which column contains missing values?"*
- *"What is the correlation between Price and Revenue?"*

For a question like "Which product generated the highest revenue?", the app will:
1. Calculate the answer using `groupby()` + `sum()` in Pandas.
2. Show the product name and revenue value.
3. Display a supporting bar chart.
4. Print a short, plain-language explanation.

---

## How the "AI" part works (and what it doesn't do)

The first version does **not** call any LLM API. Questions are matched to
one of a few common patterns (average, highest, top N, most common,
trend, missing values, correlation) using simple keyword matching in
`data_analyzer.answer_question()`.

If you later plug in an LLM API, it should only be used for two things:
1. Turning a free-text question into one of these analysis instructions.
2. Explaining the *already-calculated* Pandas result in plain language.

The LLM should **never** be asked to produce the numbers itself — this
avoids the model "hallucinating" statistics that don't match the real data.

---

## Notes on data cleaning

The app never deletes or modifies your data automatically. It only *reports*
what it finds (missing values, duplicate rows, constant columns). Two
optional checkboxes let you apply simple cleaning actions if you want to:
- Remove fully duplicated rows
- Fill missing numeric values with the column median

These actions only apply to the current session's working copy — your
original uploaded file is never overwritten.

---

## Extending this project later

Once this basic version works and is understood, natural next steps could include:
- Plugging in an LLM to handle more flexible, free-text questions.
- Adding more chart types or interactive filters.
- Saving generated insights to a PDF/Word report.
- Supporting multiple datasets / joins.

The project intentionally avoids RAG, multi-agent systems, vector databases,
or complex ML models in this first version — those can be added later once
the fundamentals are solid.
