"""
Online Retail Customer Behaviour & Sales Performance Dashboard
----------------------------------------------------------------
Interactive Streamlit dashboard for the UCI "Online Retail" dataset.

Run locally:
    streamlit run app.py

Deploy:
    Push this folder to a public GitHub repo, then deploy on
    https://share.streamlit.io (Streamlit Community Cloud) pointing at app.py.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Page config 
st.set_page_config(
    page_title="Online Retail Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"
DEFAULT_RAW_PATH = f"{DATA_DIR}/Online_Retail.xlsx"
DEFAULT_CLEAN_PATH = f"{DATA_DIR}/online_retail_clean.csv"



# Data loading & cleaning (cached so it only runs once per session/data change)

@st.cache_data(show_spinner=False)
def load_raw(file) -> pd.DataFrame:
    """Load the raw Online Retail file (xlsx or csv), whatever is handed in."""
    if hasattr(file, "name"):
        name = file.name.lower()
    else:
        name = str(file).lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(file)
    return df


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply the full cleaning pipeline and return the cleaned dataframe plus
    a summary dict of how many rows were removed at each step (for transparency
    in the dashboard and for the report).
    """
    stats = {"rows_start": len(df)}

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Basic dtype safety
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["StockCode"] = df["StockCode"].fillna("").astype(str)

    # 1. Remove cancelled orders (InvoiceNo starts with 'C')
    before = len(df)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    stats["removed_cancellations"] = before - len(df)

    # 2. Remove bad-debt adjustment rows (InvoiceNo starts with 'A')
    before = len(df)
    df = df[~df["InvoiceNo"].str.startswith("A")]
    stats["removed_bad_debt"] = before - len(df)

    # 3. Drop rows with missing CustomerID (customer-behaviour focus needs this)
    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    stats["removed_missing_customerid"] = before - len(df)

    # 4. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    stats["removed_duplicates"] = before - len(df)

    # 5. Fix CustomerID dtype
    df["CustomerID"] = df["CustomerID"].astype(int)

    # 6. Create TotalPrice
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # 7. Remove generic "Manual" adjustment rows
    before = len(df)
    df = df[df["Description"].str.strip().str.lower() != "manual"]
    stats["removed_manual_rows"] = before - len(df)

    # 8. Remove extreme outlier transactions (top 0.01% by Quantity)
    #    Documented, defensible cutoff rather than a magic number.
    before = len(df)
    qty_cap = df["Quantity"].quantile(0.9999)
    df = df[df["Quantity"] <= qty_cap]
    stats["removed_extreme_outliers"] = before - len(df)

    stats["rows_final"] = len(df)
    return df, stats


@st.cache_data(show_spinner=False)
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    return rfm


@st.cache_data(show_spinner=False)
def train_model(rfm: pd.DataFrame):
    X = rfm[["Recency", "Frequency"]]
    y = rfm["Monetary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "coef_recency": model.coef_[0],
        "coef_frequency": model.coef_[1],
        "intercept": model.intercept_,
    }
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    return model, metrics, results_df



# Sidebar: data source selection

st.sidebar.title("🛒 Retail Dashboard")
st.sidebar.markdown("---")
st.sidebar.subheader("Data source")

uploaded_file = st.sidebar.file_uploader(
    "Upload Online Retail file (.xlsx or .csv)",
    type=["xlsx", "csv"],
    help="Uses the raw UCI 'Online Retail' file format. If you don't upload "
    "one, the app will try to load a bundled sample from the data/ folder.",
)


# Load data — with clear, user-facing error handling (no silent failures)

raw_df = None
load_error = None

try:
    if uploaded_file is not None:
        raw_df = load_raw(uploaded_file)
    else:
        import os

        if os.path.exists(DEFAULT_RAW_PATH):
            raw_df = load_raw(DEFAULT_RAW_PATH)
        elif os.path.exists(DEFAULT_CLEAN_PATH):
            raw_df = None  # signal to load the pre-cleaned CSV path below
        else:
            load_error = (
                "No data found. Upload an Online Retail .xlsx/.csv file "
                "in the sidebar to get started."
            )
except Exception as exc:  # noqa: BLE001 - surface any load error to the user
    load_error = f"Couldn't read that file: {exc}"

if load_error:
    st.warning(load_error)
    st.stop()

# Clean data (or load already-cleaned CSV if that's all that's available)
import os

if raw_df is not None:
    try:
        df_clean, clean_stats = clean_data(raw_df)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Something went wrong while cleaning the data: {exc}\n\n"
            "Check that the file has the expected Online Retail columns: "
            "InvoiceNo, StockCode, Description, Quantity, InvoiceDate, "
            "UnitPrice, CustomerID, Country."
        )
        st.stop()
elif os.path.exists(DEFAULT_CLEAN_PATH):
    df_clean = pd.read_csv(DEFAULT_CLEAN_PATH, parse_dates=["InvoiceDate"])
    clean_stats = None
else:
    st.stop()

if df_clean.empty:
    st.error("The cleaned dataset has zero rows — check the uploaded file's format.")
    st.stop()


# Sidebar filters (applied AFTER cleaning, on top of the full cleaned dataset)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

min_date = df_clean["InvoiceDate"].min().date()
max_date = df_clean["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

all_countries = sorted(df_clean["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries", options=all_countries, default=all_countries
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (
        (df_clean["InvoiceDate"].dt.date >= start_date)
        & (df_clean["InvoiceDate"].dt.date <= end_date)
        & (df_clean["Country"].isin(selected_countries))
    )
    view = df_clean.loc[mask]
else:
    view = df_clean.loc[df_clean["Country"].isin(selected_countries)]

if view.empty:
    st.warning("No data matches the current filters — widen your date range or country selection.")
    st.stop()


# Header + KPI row

st.title("Online Retail — Customer Behaviour & Sales Performance")
st.caption(
    "UCI 'Online Retail' dataset — a UK-based online gift retailer, "
    "Dec 2010 to Dec 2011."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"£{view['TotalPrice'].sum():,.0f}")
k2.metric("Orders", f"{view['InvoiceNo'].nunique():,}")
k3.metric("Customers", f"{view['CustomerID'].nunique():,}")
k4.metric("Avg Order Value", f"£{view.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.2f}")

st.markdown("---")


# Tabs

tab_overview, tab_products, tab_customers, tab_model, tab_data = st.tabs(
    ["Sales Overview", "Products & Geography", "Customer Behaviour", "Prediction Model", "Data Quality"]
)

# ---- Tab 1: Sales overview ----
with tab_overview:
    st.subheader("Monthly Sales Trend")
    monthly = (
        view.assign(YearMonth=view["InvoiceDate"].dt.to_period("M").astype(str))
        .groupby("YearMonth")["TotalPrice"]
        .sum()
        .reset_index()
    )
    fig = px.line(monthly, x="YearMonth", y="TotalPrice", markers=True)
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Revenue (£)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note: the final month may look artificially low if your selected "
        "date range ends mid-month — that's a partial-period effect, not a real drop."
    )

    st.subheader("Order Value Distribution")
    cap = st.slider("Zoom x-axis (max order line value £)", 20, 500, 100, step=10)
    fig2 = px.histogram(view[view["TotalPrice"] <= cap], x="TotalPrice", nbins=50)
    fig2.update_layout(xaxis_title="Order Line Value (£)", yaxis_title="Frequency")
    st.plotly_chart(fig2, use_container_width=True)

# ---- Tab 2: Products & geography ----
with tab_products:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Products by Quantity")
        top_products = (
            view.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        )
        fig3 = px.bar(top_products[::-1], orientation="h")
        fig3.update_layout(showlegend=False, xaxis_title="Total Quantity Sold", yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Top 10 Countries by Revenue (excl. UK)")
        country_sales = (
            view[view["Country"] != "United Kingdom"]
            .groupby("Country")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        fig4 = px.bar(country_sales[::-1], orientation="h", color_discrete_sequence=["seagreen"])
        fig4.update_layout(showlegend=False, xaxis_title="Total Revenue (£)", yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    uk_share = (
        view.loc[view["Country"] == "United Kingdom", "TotalPrice"].sum()
        / view["TotalPrice"].sum()
        * 100
        if view["TotalPrice"].sum() > 0
        else 0
    )
    st.info(f"UK share of total revenue in current selection: **{uk_share:.1f}%**")

# ---- Tab 3: Customer behaviour ----
with tab_customers:
    st.subheader("Customer Purchase Frequency")
    customer_orders = view.groupby("CustomerID")["InvoiceNo"].nunique()
    max_orders_shown = st.slider("Max orders shown (for readability)", 5, 50, 20)
    fig5 = px.histogram(customer_orders[customer_orders <= max_orders_shown], nbins=max_orders_shown)
    fig5.update_layout(xaxis_title="Number of Orders", yaxis_title="Number of Customers", showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

    one_time_pct = (customer_orders == 1).mean() * 100
    st.info(f"**{one_time_pct:.1f}%** of customers in the current selection made only one order.")

    st.subheader("Correlation Heatmap")
    corr = view[["Quantity", "UnitPrice", "TotalPrice"]].corr()
    fig6 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig6, use_container_width=True)

# ---- Tab 4: Prediction model (RFM + Linear Regression) ----
with tab_model:
    st.subheader("Predicting Customer Spend from Recency & Frequency")
    st.caption(
        "A Linear Regression model trained on RFM features to predict total "
        "customer spend (Monetary) from purchase Recency and Frequency."
    )

    if view["CustomerID"].nunique() < 20:
        st.warning("Not enough customers in the current filter selection to train a meaningful model. Widen your filters.")
    else:
        rfm = build_rfm(view)
        model, metrics, results_df = train_model(rfm)

        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metrics['r2']:.3f}")
        m2.metric("MAE", f"£{metrics['mae']:,.0f}")
        m3.metric("RMSE", f"£{metrics['rmse']:,.0f}")

        st.markdown(
            f"**Model equation:** `Monetary ≈ {metrics['intercept']:,.1f} "
            f"+ ({metrics['coef_recency']:.2f} × Recency) "
            f"+ ({metrics['coef_frequency']:.2f} × Frequency)`"
        )

        fig7 = px.scatter(
            results_df,
            x="Actual",
            y="Predicted",
            opacity=0.5,
            labels={"Actual": "Actual Monetary Spend (£)", "Predicted": "Predicted Monetary Spend (£)"},
        )
        max_val = float(results_df[["Actual", "Predicted"]].max().max())
        fig7.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig7, use_container_width=True)

        st.caption(
            "Points near the red dashed line are well-predicted. A small number "
            "of very high-spend outlier customers typically pull RMSE up — "
            "look at MAE for a more typical error picture."
        )

# ---- Tab 5: Data quality / cleaning transparency ----
with tab_data:
    st.subheader("Cleaning Summary")
    if clean_stats:
        st.json(clean_stats)
    else:
        st.info("Loaded a pre-cleaned CSV — no cleaning stats available for this run.")

    st.subheader("Sample of Cleaned Data")
    st.dataframe(view.head(100), use_container_width=True)

    st.download_button(
        "Download filtered data as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_retail_data.csv",
        mime="text/csv",
    )
