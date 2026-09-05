import pandas as pd

path = r"C:\Users\gopin\.cache\kagglehub\competitions\walmart-recruiting-store-sales-forecasting"

train = pd.read_csv(path + r"\train.csv")

train["Date"] = pd.to_datetime(train["Date"])

# Sort chronologically within each store and department
train = train.sort_values(["Store", "Dept", "Date"])

# Calendar features
train["Year"] = train["Date"].dt.year
train["Month"] = train["Date"].dt.month
train["Week"] = train["Date"].dt.isocalendar().week.astype(int)

# Previous week's sales
train["Lag_1"] = (
    train.groupby(["Store", "Dept"])["Weekly_Sales"]
         .shift(1)
)

print(train.head(15))