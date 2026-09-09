import pandas as pd

from feature_engineering import create_features

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

train = create_features()

print("Before removing NaNs:", train.shape)

train = train[train["Rolling_Mean_8"].notna()].copy()

print("After removing NaNs:", train.shape)
print(train.isnull().sum())

train = train.sort_values("Date")

split_date = train["Date"].quantile(0.8)

train_data = train[train["Date"] < split_date].copy()
validation_data = train[train["Date"] >= split_date].copy()

print("Split date:", split_date)
print("Training data:", train_data.shape)
print("Validation data:", validation_data.shape)

print("Training period:", train_data["Date"].min(), "to", train_data["Date"].max())
print("Validation period:", validation_data["Date"].min(), "to", validation_data["Date"].max())

model_features = [
    "Store",
    "Dept",
    "IsHoliday",
    "Year",
    "Month",
    "Week",
    "Lag_1",
    "Rolling_Mean_4",
    "Rolling_Mean_8",
    "Type",
    "Size",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment"
]

X_train = train_data[model_features].copy()
y_train = train_data["Weekly_Sales"].copy()

X_val = validation_data[model_features].copy()
y_val = validation_data["Weekly_Sales"].copy()

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

X_train = pd.get_dummies(
    X_train,
    columns=["Type"],
    dtype=int
)

X_val = pd.get_dummies(
    X_val,
    columns=["Type"],
    dtype=int
)

X_val = X_val.reindex(
    columns=X_train.columns,
    fill_value=0
)

print("X_train after encoding:", X_train.shape)
print("X_val after encoding:", X_val.shape)
print(X_train.head())

print("Missing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in X_val:", X_val.isnull().sum().sum())

baseline_predictions = X_val["Lag_1"]

print("Baseline predictions:")
print(baseline_predictions.head())

baseline_rmse = np.sqrt(
    mean_squared_error(y_val, baseline_predictions)
)

baseline_mae = mean_absolute_error(
    y_val,
    baseline_predictions
)

print("Baseline RMSE:", baseline_rmse)
print("Baseline MAE:", baseline_mae)

model = RandomForestRegressor(
    n_estimators=200,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

rf_predictions = model.predict(X_val)

rf_rmse = np.sqrt(
    mean_squared_error(y_val, rf_predictions)
)

rf_mae = mean_absolute_error(
    y_val,
    rf_predictions
)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest MAE:", rf_mae)

gbr_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gbr_model.fit(X_train, y_train)

gbr_predictions = gbr_model.predict(X_val)

gbr_rmse = np.sqrt(
    mean_squared_error(y_val, gbr_predictions)
)

gbr_mae = mean_absolute_error(
    y_val,
    gbr_predictions
)

print("Gradient Boosting RMSE:", gbr_rmse)
print("Gradient Boosting MAE:", gbr_mae)

rmse_improvement = (
    (baseline_rmse - rf_rmse) / baseline_rmse
) * 100

mae_improvement = (
    (baseline_mae - rf_mae) / baseline_mae
) * 100

print("\nModel improvement over baseline:")
print("RMSE improvement:", rmse_improvement, "%")
print("MAE improvement:", mae_improvement, "%")