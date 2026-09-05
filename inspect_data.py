import pandas as pd

path = r"C:\Users\gopin\.cache\kagglehub\competitions\walmart-recruiting-store-sales-forecasting"

train = pd.read_csv(path + r"\train.csv")
test = pd.read_csv(path + r"\test.csv")
features = pd.read_csv(path + r"\features.csv")
stores = pd.read_csv(path + r"\stores.csv")

print("\n--- TRAIN ---")
print(train.shape)
print(train.columns)
print(train.head())

print("\n--- TEST ---")
print(test.shape)
print(test.columns)
print(test.head())

print("\n--- FEATURES ---")
print(features.shape)
print(features.columns)
print(features.head())

print("\n--- STORES ---")
print(stores.shape)
print(stores.columns)
print(stores.head())

print("\n--- TRAIN INFO ---")
print(train.info())

print("\n--- MISSING VALUES ---")
print(train.isnull().sum())

print("\n--- DATE RANGE ---")
print(train["Date"].min())
print(train["Date"].max())

print("\n--- UNIQUE STORES ---")
print(train["Store"].nunique())

print("\n--- UNIQUE DEPARTMENTS ---")
print(train["Dept"].nunique())

print("\n--- SALES STATISTICS ---")
print(train["Weekly_Sales"].describe())