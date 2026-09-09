# Walmart Sales Forecasting

Machine learning project for forecasting weekly Walmart store sales using historical sales, store information, economic indicators, weather data, and promotional features.

## Project Overview

The goal of this project is to predict weekly sales for Walmart stores and departments using machine learning.

The project follows a time-based forecasting workflow:

1. Exploratory Data Analysis
2. Feature Engineering
3. Time-Based Train/Validation Split
4. Baseline Model
5. Machine Learning Model Comparison
6. Random Forest Optimization
7. Interactive Streamlit Dashboard

## Dataset

The project uses the Walmart Recruiting Store Sales Forecasting dataset from Kaggle.

The dataset contains:

- Historical weekly sales for Walmart stores and departments
- Store information such as store type and size
- Temperature and fuel prices
- CPI and unemployment data
- Markdown promotional information
- Holiday indicators

## Features

### Time Features
- Year
- Month
- Week

### Historical Sales Features
- Lag-1: previous week's sales for the same store and department
- 4-week rolling mean
- 8-week rolling mean

### Store Features
- Store
- Department
- Store Type
- Store Size

### External Features
- Temperature
- Fuel Price
- CPI
- Unemployment
- Markdown 1–5
- Holiday indicator

## Modeling

A chronological 80/20 train-validation split was used instead of randomly shuffling the data. This better reflects the real-world forecasting scenario, where future sales should be predicted using information available from the past.

### Models Compared

- Lag-1 Baseline
- Random Forest Regressor
- Gradient Boosting Regressor

### Final Model

The final model is a Random Forest Regressor configured with:

- 200 trees
- `max_features="sqrt"`
- `random_state=42`
- Parallel processing using `n_jobs=-1`

### Results

| Model | RMSE | MAE |
|---|---:|---:|
| Lag-1 Baseline | 3428.64 | 1543.87 |
| Random Forest | 2916.46 | 1439.50 |
| Gradient Boosting | 3101.69 | 1515.52 |

The Random Forest model achieved the lowest RMSE and MAE among the tested models.

Compared with the Lag-1 baseline, the final Random Forest model reduced RMSE by approximately **14.94%**.

## Project Structure

```text
sales-forecasting/
│
├── dashboard.py
├── data_preparation.py
├── download_data.py
├── eda.py
├── feature_engineering.py
├── inspect_data.py
└── README.md