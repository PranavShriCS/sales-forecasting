import pandas as pd

from feature_engineering import create_features

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