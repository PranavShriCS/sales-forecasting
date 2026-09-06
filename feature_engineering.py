import pandas as pd

def create_features():

    path = r"C:\Users\gopin\.cache\kagglehub\competitions\walmart-recruiting-store-sales-forecasting"

    train = pd.read_csv(path + r"\train.csv")

    stores = pd.read_csv(path + r"\stores.csv")

    features = pd.read_csv(path + r"\features.csv")

    train["Date"] = pd.to_datetime(train["Date"])

    features["Date"] = pd.to_datetime(features["Date"])

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

    train["Rolling_Mean_4"] = (
        train.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    train["Rolling_Mean_8"] = (
        train.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(8).mean())
    )

    train = train.merge(
        stores,
        on="Store",
        how="left"
    )

    train = train.merge(
        features,
        on=["Store", "Date"],
        how="left"
    )

    train = train.drop(columns=["IsHoliday_y"])
    train = train.rename(columns={"IsHoliday_x": "IsHoliday"})

    markdown_cols = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    train[markdown_cols] = train[markdown_cols].fillna(0)

    print(train.head())

    print(train.columns)

    return train;