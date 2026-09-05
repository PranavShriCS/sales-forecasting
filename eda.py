import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\gopin\.cache\kagglehub\competitions\walmart-recruiting-store-sales-forecasting"

train = pd.read_csv(path + r"\train.csv")

train["Date"] = pd.to_datetime(train["Date"])

# Total weekly sales over time
weekly_sales = train.groupby("Date")["Weekly_Sales"].sum()

plt.figure(figsize=(12, 5))
plt.plot(weekly_sales)
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Total sales by store
store_sales = train.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 5))
plt.bar(store_sales.index.astype(str), store_sales.values)
plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Distribution of weekly sales
plt.figure(figsize=(10, 5))
plt.hist(train["Weekly_Sales"], bins=100)
plt.title("Distribution of Weekly Sales")
plt.xlabel("Weekly Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()