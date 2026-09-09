import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Walmart Sales Forecasting",
    layout="wide"
)

st.title("Walmart Sales Forecasting")

st.write(
    "Machine learning project for forecasting weekly Walmart store sales."
)

st.info(
    "This project uses historical Walmart sales data and machine learning "
    "to forecast weekly sales. The final model is a Random Forest Regressor "
    "using time-based and historical sales features."
)

path = r"C:\Users\gopin\.cache\kagglehub\competitions\walmart-recruiting-store-sales-forecasting"

train = pd.read_csv(path + r"\train.csv")

train["Date"] = pd.to_datetime(train["Date"])

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", f"{len(train):,}")

with col2:
    st.metric("Stores", train["Store"].nunique())

with col3:
    st.metric("Departments", train["Dept"].nunique())

st.header("Weekly Sales Trend")

sales_over_time = (
    train.groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
)

st.line_chart(
    sales_over_time,
    x="Date",
    y="Weekly_Sales"
)

st.header("Total Sales by Store")

sales_by_store = (
    train.groupby("Store")["Weekly_Sales"]
    .sum()
    .reset_index()
    .sort_values("Weekly_Sales", ascending=False)
)

st.bar_chart(
    sales_by_store,
    x="Store",
    y="Weekly_Sales"
)

st.header("Model Comparison")

model_results = pd.DataFrame({
    "Model": [
        "Lag-1 Baseline",
        "Random Forest",
        "Gradient Boosting"
    ],
    "RMSE": [
        3428.64,
        2916.46,
        3101.69
    ],
    "MAE": [
        1543.87,
        1439.50,
        1515.52
    ]
})

st.dataframe(model_results, hide_index=True)

st.subheader("RMSE Comparison")

st.bar_chart(
    model_results,
    x="Model",
    y="RMSE"
)

st.header("Final Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline RMSE", "3428.64")

with col2:
    st.metric("Final RMSE", "2916.46")

with col3:
    st.metric("RMSE Improvement", "14.94%")

st.write(
    "The Random Forest model reduced RMSE by 14.94% compared with "
    "the Lag-1 baseline."
)

st.header("Features Used")

st.write(
    """
    The model uses:
    - Historical sales: Lag-1, 4-week rolling mean, 8-week rolling mean
    - Time features: Year, Month, Week
    - Store information: Store, Department, Type, Size
    - Economic features: CPI, Unemployment, Fuel Price
    - Weather and promotion features: Temperature, Markdown variables
    - Holiday information
    """
)

st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("RMSE", "2916.46")

with col2:
    st.metric("MAE", "1439.50")

st.subheader("Model Used")

st.write(
    "Random Forest Regressor with 200 trees and max_features='sqrt'."
)