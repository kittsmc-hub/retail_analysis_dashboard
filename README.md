# Online Retail — Customer Purchasing Behaviour & Sales Performance Analysis

Individual data science project for DSA 1080XA — Individual Programming for
Data Science**. This repository contains the full analysis of a real-world
online retail transaction dataset, covering data cleaning, exploratory data
analysis, visualization, statistical analysis, and a beginner-level machine
learning model predicting customer spend.

## 📁 Repository Structure

```
online-retail-analysis/
├── README.md                      # this file
├── retail_analysis.ipynb          # main Jupyter notebook — full analysis
├── data/
│   └── Online_Retail.xlsx         # raw dataset (see Dataset section below)
└── retail_dashboard/              # optional: interactive Streamlit dashboard
    ├── app.py
    ├── requirements.txt
    └── README.md
```

## 📊 Dataset

- **Name:** Online Retail
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/dataset/352/online+retail
- **Citation:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine
  Learning Repository.
- **Size:** 541,909 transaction records, 8 columns
- **Coverage:** 1 December 2010 – 9 December 2011, a UK-based online retailer
  selling all-occasion giftware, transactions across 37 countries

| Column | Description |
|---|---|
| InvoiceNo | 6-digit unique transaction number (starts with 'C' if cancelled) |
| StockCode | 5-digit unique product code |
| Description | Product name |
| Quantity | Units purchased in that transaction line |
| InvoiceDate | Date and time of the transaction |
| UnitPrice | Price per unit (£) |
| CustomerID | 5-digit unique customer identifier |
| Country | Customer's country of residence |

If `data/Online_Retail.xlsx` is not present in this repo (e.g. excluded for
size reasons), download it directly from the UCI link above and place it in
the `data/` folder before running the notebook.

## 🧪 What's in the Notebook

`retail_analysis.ipynb` is organised into the following sections, in order:

1. **Data Loading** — reading the raw Excel file into pandas
2. **Initial Inspection** — shape, dtypes, summary statistics, missing values, duplicates
3. **Data Cleaning** — removing cancellations, bad-debt adjustment rows, missing CustomerIDs, duplicates, a "Manual" adjustment entries, and one extreme outlier transaction (each with justification in markdown cells)
4. **Feature Engineering** — creating `TotalPrice` (Quantity × UnitPrice) and RFM (Recency, Frequency, Monetary) features per customer
5. **Exploratory Data Analysis** — groupby/aggregation-based investigation of top products, revenue by country, monthly trends, and customer order frequency
6. **Visualizations** — 6 charts: monthly sales trend, top 10 products, revenue by country, order value distribution, customer purchase frequency, and a correlation heatmap
7. **Statistical Analysis** — mean, median, mode, standard deviation, and correlation analysis
8. **Machine Learning Model** — a Linear Regression model predicting customer total spend from Recency and Frequency, evaluated with MAE, RMSE, and R²
9. **Conclusions** — key findings, business recommendations, limitations, and suggestions for future improvement

## ⚙️ How to Run the Notebook

**Requirements:** Python 3.9+ and Jupyter (via VS Code or `jupyter notebook`)

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/online-retail-analysis.git
   cd online-retail-analysis
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
   ```
4. Make sure `data/Online_Retail.xlsx` is present (see Dataset section above).
5. Open the notebook:
   ```bash
   jupyter notebook retail_analysis.ipynb
   ```
   or open the repository folder in VS Code and launch the notebook from there.
6. Run all cells in order (Kernel → Restart & Run All) to reproduce the full
   analysis from raw data to final model.

## 📈 Key Findings (Summary)

- Revenue shows strong seasonality, rising sharply from September and
  peaking in November — consistent with pre-Christmas wholesale ordering.
- The UK accounts for **81.9%** of total revenue; the next largest markets
  are the Netherlands, Ireland, Germany, and France.
- **34.5%** of customers made only a single purchase across the entire
  dataset period.
- Total spend correlates strongly with purchase **Quantity** (r = 0.84) but
  only weakly with **UnitPrice** (r = 0.13).
- A Linear Regression model using only Recency and Frequency explains
  **~41.6%** of the variance in customer spend (R² = 0.4164).

Full methodology, reasoning for each cleaning decision, and detailed
interpretation of every result are documented in the notebook itself and in
the accompanying project report.

## 🖥️ Optional: Interactive Dashboard

The `retail_dashboard/` folder contains a Streamlit web app version of this
analysis with interactive filters. See `retail_dashboard/README.md` for setup
and deployment instructions. This is a supplementary extension and is not a
substitute for the notebook.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine learning:** scikit-learn
- **Environment:** Jupyter Notebook / VS Code

## 👤 Author

Kitts Mark Makokha — 202603164
DSA 1080XA

## 📄 Academic Integrity Note

This project was completed individually as required by the course's academic
integrity policy. AI tools were used only for learning concepts and
debugging assistance during development; all analysis, code, and written
interpretation in the final submission reflect my own understanding and were
reviewed and rewritten by me before submission.